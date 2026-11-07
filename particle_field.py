import argparse
import colorsys
import math
import os
import time

import numpy as np
import pygame

PALETTES = {
    "aurora": ["#0e3b4a", "#12607a", "#1c8f9a", "#2fc0a6", "#5ee3b4", "#9bf5cc", "#d4fbe8", "#ffffff"],
    "ember": ["#3a0f1e", "#6b1429", "#a3202b", "#d8431f", "#f27a1a", "#f9b23c", "#fcdf8a", "#fff6d8"],
    "ultraviolet": ["#1a1450", "#2c1b8a", "#4a22b8", "#7a2ed0", "#b042d6", "#e063c8", "#f5a3dc", "#fff0fa"],
    "mono": ["#2a2d38", "#454a59", "#646a7c", "#868ca0", "#a9aec0", "#c8ccda", "#e4e7f0", "#ffffff"],
}
PALETTE_KEYS = ["aurora", "ember", "ultraviolet", "mono"]
SHAPES = ["grid", "rings", "spiral"]
BG = (7, 8, 12)
BINS = 8


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def make_lut(name):
    lut = np.array([hex_to_rgb(c) for c in PALETTES[name]], dtype=np.int16)
    # lift the two slowest bins so particles at rest stay visible
    lut[0] = (lut[0] + lut[2]) // 2
    lut[1] = (lut[1] + lut[3]) // 2
    return lut


def make_custom_lut(hue, sat=1.0):
    cols = []
    for i in range(BINS):
        t = i / (BINS - 1)
        v = 0.30 + 0.70 * t ** 0.7
        s = sat * (1.0 - 0.85 * t ** 3)
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, s, v)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    lut = np.array(cols, dtype=np.int16)
    lut[0] = (lut[0] + lut[2]) // 2
    lut[1] = (lut[1] + lut[3]) // 2
    return lut


class Field:
    def __init__(self, w, h, n, shape):
        self.w, self.h, self.n, self.shape = w, h, 0, shape
        self.rng = np.random.default_rng()
        self.px = self.py = self.vx = self.vy = self.hx = self.hy = None
        self.resize_count(n)

    # ---- home shapes -------------------------------------------------
    def build_homes(self):
        n, w, h, rng = self.n, self.w, self.h, self.rng
        i = np.arange(n)
        if self.shape == "grid":
            cols = max(1, int(math.ceil(math.sqrt(n * w / h))))
            rows = int(math.ceil(n / cols))
            sx, sy = w / cols, h / rows
            hx = (i % cols + 0.5) * sx + (rng.random(n) - 0.5) * sx * 0.4
            hy = (i // cols + 0.5) * sy + (rng.random(n) - 0.5) * sy * 0.4
        elif self.shape == "rings":
            rings = 14
            r = (1 + (i % rings)) / rings * min(w, h) * 0.46
            a = rng.random(n) * math.tau
            hx = w / 2 + np.cos(a) * r
            hy = h / 2 + np.sin(a) * r
        else:  # spiral
            t = i / n
            arms = 3
            ang = t * 9 * math.tau + (i % arms) * (math.tau / arms)
            r = np.sqrt(t) * min(w, h) * 0.47
            hx = w / 2 + np.cos(ang) * r + (rng.random(n) - 0.5) * 3
            hy = h / 2 + np.sin(ang) * r + (rng.random(n) - 0.5) * 3
        self.hx = hx.astype(np.float32)
        self.hy = hy.astype(np.float32)

    def resize_count(self, n):
        old = self.n
        self.n = n
        self.build_homes()
        npx = self.hx.copy()
        npy = self.hy.copy()
        nvx = np.zeros(n, np.float32)
        nvy = np.zeros(n, np.float32)
        if self.px is not None:
            m = min(old, n)
            npx[:m], npy[:m] = self.px[:m], self.py[:m]
            nvx[:m], nvy[:m] = self.vx[:m], self.vy[:m]
        self.px, self.py, self.vx, self.vy = npx, npy, nvx, nvy

    def resize_area(self, w, h):
        sx, sy = w / self.w, h / self.h
        self.px *= sx
        self.py *= sy
        self.w, self.h = w, h
        self.build_homes()

    def set_shape(self, shape):
        self.shape = shape
        self.build_homes()

    def scatter(self, lo=4.0, hi=18.0):
        a = self.rng.random(self.n) * math.tau
        s = lo + self.rng.random(self.n) * (hi - lo)
        self.vx += (np.cos(a) * s).astype(np.float32)
        self.vy += (np.sin(a) * s).astype(np.float32)

    # ---- physics -----------------------------------------------------
    def step(self, dt, t, mx, my, mvx, mvy, radius, force, flow, turb, spring):
        px, py, vx, vy = self.px, self.py, self.vx, self.vy

        dx = px - mx
        dy = py - my
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2) + 1e-3
        k = np.clip(1.0 - d / radius, 0.0, 1.0)
        k *= k  # strongest at the pointer, fades smoothly to 0 at the radius
        push = force * k * 3.0 * dt
        vx += dx / d * push + mvx * flow * k * 0.35
        vy += dy / d * push + mvy * flow * k * 0.35

        # curl-like noise field made from a few sines
        tt = t * 0.35
        fq = 0.004
        ang = (np.sin(px * fq + tt) + np.cos(py * fq * 1.3 - tt * 0.7)
               + np.sin((px + py) * fq * 0.6 + tt * 0.4)) * 1.6
        vx += np.cos(ang) * (turb * 0.08 * dt)
        vy += np.sin(ang) * (turb * 0.08 * dt)

        # spring back to the home shape
        vx += (self.hx - px) * (spring * dt)
        vy += (self.hy - py) * (spring * dt)

        damp = 0.93 ** dt
        vx *= damp
        vy *= damp

        sp = np.sqrt(vx * vx + vy * vy)
        over = sp > 30.0
        if over.any():
            sc = np.where(over, 30.0 / np.maximum(sp, 1e-6), 1.0).astype(np.float32)
            vx *= sc
            vy *= sc
            sp = np.minimum(sp, 30.0)

        px += vx * dt
        py += vy * dt
        return sp


def draw_particles(canvas, field, speed, lut):
    w, h = canvas.get_size()
    xi = field.px.astype(np.int32)
    yi = field.py.astype(np.int32)
    ok = (xi >= 0) & (xi < w - 1) & (yi >= 0) & (yi < h - 1)
    xi, yi = xi[ok], yi[ok]
    bins = np.minimum((speed[ok] * 1.1).astype(np.int32), BINS - 1)
    cols = lut[bins]

    arr = pygame.surfarray.pixels3d(canvas)
    cur = np.maximum(arr[xi, yi].astype(np.int16), cols)
    arr[xi, yi] = cur.astype(np.uint8)

    big = bins >= 5  # fast particles get a 2x2 dot so they glow
    if big.any():
        bx, by, bc = xi[big] + 1, yi[big] + 1, cols[big]
        cur = np.maximum(arr[bx, by].astype(np.int16), bc)
        arr[bx, by] = cur.astype(np.uint8)
    del arr  # release the surface lock


def parse_args():
    p = argparse.ArgumentParser(description="Mouse-driven particle field (NumPy + pygame)")
    p.add_argument("--particles", type=int, default=16000)
    p.add_argument("--size", default="1280x800", help="window size, e.g. 1280x800")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--color", default="", help='start with your own color, e.g. "#ff3366"')
    p.add_argument("--bench", type=int, default=0, help="run N frames headless, print fps, exit")
    p.add_argument("--shot", default="", help="with --bench: save a screenshot to this path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.bench:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    w, h = (int(v) for v in args.size.lower().split("x"))

    pygame.init()
    flags = pygame.RESIZABLE | (pygame.FULLSCREEN if args.fullscreen else 0)
    screen = pygame.display.set_mode((w, h), flags)
    pygame.display.set_caption("Particle Field")
    w, h = screen.get_size()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,dejavusansmono,couriernew,monospace", 14)

    canvas = pygame.Surface((w, h)).convert()
    canvas.fill(BG)

    field = Field(w, h, args.particles, "grid")
    field.scatter(2.0, 10.0)  # start scattered so it assembles on launch

    palette_i = 0             # 0-3 built-in, 4 = custom color
    custom_h, custom_s = 0.5, 1.0
    if args.color:
        try:
            cr, cg, cb = hex_to_rgb(args.color)
            custom_h, custom_s, _ = colorsys.rgb_to_hsv(cr / 255, cg / 255, cb / 255)
            custom_s = max(custom_s, 0.35)
            palette_i = 4
        except ValueError:
            print("Could not read --color, expected something like #ff3366")
    rainbow = False
    dragging_hue = False

    def current_lut():
        if palette_i == 4:
            return make_custom_lut(custom_h, custom_s)
        return make_lut(PALETTE_KEYS[palette_i])

    def strip_rect():
        return pygame.Rect(16, h - 36, max(60, min(360, w - 32)), 16)

    def hue_from_x(x):
        r = strip_rect()
        return min(max((x - r.x) / r.w, 0.0), 1.0)

    lut = current_lut()
    shape_i = 0
    radius, force, flow, turb, spring = 170.0, 1.2, 0.55, 0.18, 0.004
    trail = 0.22          # 0.04 (long trails) .. 1.0 (none)
    mode = 1              # 1 repel, -1 attract
    show_hud = True

    mx, my = w / 2, h / 2     # smoothed pointer
    tx, ty = mx, my           # target pointer
    last_input = -1e9
    t_start = time.time()
    fps_smooth = 60.0
    frame_no = 0

    running = True
    while running:
        dt = min(clock.tick(0 if args.bench else 60) / 16.667, 2.5)
        if args.bench:
            dt = 1.0
        now = time.time() - (t_start if not args.bench else t_start)
        t = frame_no / 60.0 if args.bench else now

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEMOTION:
                if dragging_hue:
                    custom_h = hue_from_x(ev.pos[0])
                    palette_i = 4
                    lut = current_lut()
                else:
                    tx, ty = ev.pos
                    last_input = t
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button < 4:
                if ev.button == 1 and show_hud and strip_rect().inflate(0, 14).collidepoint(ev.pos):
                    dragging_hue = True
                    custom_h = hue_from_x(ev.pos[0])
                    palette_i = 4
                    lut = current_lut()
                else:
                    tx, ty = ev.pos
                    last_input = t
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    dragging_hue = False
            elif ev.type == pygame.MOUSEWHEEL:
                custom_h = (custom_h + ev.y * 0.02) % 1.0
                palette_i = 4
                lut = current_lut()
            elif ev.type == pygame.VIDEORESIZE or ev.type == getattr(pygame, "WINDOWRESIZED", -1):
                screen = pygame.display.get_surface()
                nw, nh = screen.get_size()
                if (nw, nh) != (w, h) and nw > 0 and nh > 0:
                    w, h = nw, nh
                    canvas = pygame.Surface((w, h)).convert()
                    canvas.fill(BG)
                    field.resize_area(w, h)
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_m:
                    mode = -mode
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    palette_i = k - pygame.K_1
                    lut = current_lut()
                elif k in (pygame.K_z, pygame.K_x):
                    custom_h = (custom_h + (0.02 if k == pygame.K_x else -0.02)) % 1.0
                    palette_i = 4
                    lut = current_lut()
                elif k == pygame.K_r:
                    rainbow = not rainbow
                elif k == pygame.K_s:
                    shape_i = (shape_i + 1) % len(SHAPES)
                    field.set_shape(SHAPES[shape_i])
                elif k == pygame.K_SPACE:
                    field.scatter()
                elif k == pygame.K_LEFTBRACKET:
                    radius = max(40.0, radius - 15)
                elif k == pygame.K_RIGHTBRACKET:
                    radius = min(500.0, radius + 15)
                elif k == pygame.K_COMMA:
                    trail = min(1.0, trail + 0.04)
                elif k == pygame.K_PERIOD:
                    trail = max(0.04, trail - 0.04)
                elif k == pygame.K_UP:
                    field.resize_count(min(60000, field.n + 2000))
                elif k == pygame.K_DOWN:
                    field.resize_count(max(2000, field.n - 2000))
                elif k == pygame.K_LEFT:
                    turb = max(0.0, turb - 0.03)
                elif k == pygame.K_RIGHT:
                    turb = min(1.0, turb + 0.03)
                elif k == pygame.K_h:
                    show_hud = not show_hud
                elif k == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                    screen = pygame.display.get_surface()
                    nw, nh = screen.get_size()
                    if (nw, nh) != (w, h):
                        w, h = nw, nh
                        canvas = pygame.Surface((w, h)).convert()
                        canvas.fill(BG)
                        field.resize_area(w, h)

        # idle pointer so the field moves before you touch the mouse
        if t - last_input > 4.0:
            tx = w / 2 + math.sin(t * 0.7) * w * 0.32
            ty = h / 2 + math.sin(t * 1.1 + 1.3) * h * 0.28

        if rainbow:
            custom_h = (custom_h + 0.0015 * dt) % 1.0
            palette_i = 4
            lut = current_lut()

        pmx, pmy = mx, my
        mx += (tx - mx) * 0.25
        my += (ty - my) * 0.25
        mvx = (mx - pmx) / dt
        mvy = (my - pmy) / dt

        pressed = (pygame.mouse.get_pressed()[0] and not dragging_hue) if not args.bench else False
        f = force * mode * (1.8 if pressed else 1.0)

        speed = field.step(dt, t, mx, my, mvx, mvy, radius, f, flow, turb, spring)

        fade = int(round(2 + trail * trail * 253))
        canvas.fill((fade, fade, fade), special_flags=pygame.BLEND_RGB_SUB)
        draw_particles(canvas, field, speed, lut)

        screen.blit(canvas, (0, 0))
        if show_hud:
            fps_smooth += (clock.get_fps() - fps_smooth) * 0.1 if not args.bench else 0
            lines = [
                f"{field.n:,} particles   {clock.get_fps():.0f} fps",
                f"{'repel' if mode == 1 else 'attract'}  radius {radius:.0f}  turb {turb:.2f}  trail {trail:.2f}",
                f"{'custom' if palette_i == 4 else PALETTE_KEYS[palette_i]}{' (rainbow)' if rainbow else ''} / {SHAPES[shape_i]}   [H] hide  [Esc] quit",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (138, 144, 163))
                screen.blit(surf, (16, 14 + i * 18))

            # hue bar: click or drag to pick a color
            sr = strip_rect()
            for xx in range(sr.w):
                rr, gg, bb = colorsys.hsv_to_rgb(xx / sr.w, 1.0, 1.0)
                pygame.draw.line(screen, (int(rr * 255), int(gg * 255), int(bb * 255)),
                                 (sr.x + xx, sr.y), (sr.x + xx, sr.bottom - 1))
            tick = sr.x + int(custom_h * sr.w)
            pygame.draw.rect(screen, (255, 255, 255), (tick - 2, sr.y - 3, 5, sr.h + 6), 2)
            label = font.render("hue: click/drag, wheel, Z/X   R = rainbow   5 = custom", True, (138, 144, 163))
            screen.blit(label, (sr.x, sr.y - 22))
        pygame.display.flip()
        frame_no += 1

        if args.bench and frame_no >= args.bench:
            if args.shot:
                pygame.image.save(screen, args.shot)
            break

    if args.bench:
        print("bench done:", frame_no, "frames")
    pygame.quit()


if __name__ == "__main__":
    main()
