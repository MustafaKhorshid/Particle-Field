# Particle Field

A mouse-driven particle system in Python. About 16,000 particles get pushed away by your mouse, dragged along when you move fast, and pulled back into shape by a spring when you stop. You can change the color live.

<!-- Add your OBS recording here as a GIF: ![demo](demo.gif) -->

It is a Python take on the idea from Anya Maryina's TouchDesigner tutorial, [Interactive Particles with Slamtec LiDAR (Part I)](https://www.youtube.com/watch?v=Jz-irdEYUZM). This version covers the mouse mode only and is not a copy of her project.

## Run it

You need Python 3 and two packages:

```
pip install numpy pygame-ce
python particle_field.py
```

Use `pygame-ce` and not `pygame`. The original `pygame` package has no ready-made build for the newest Python versions (3.14 on Windows, for example) and the install fails. `pygame-ce` is a drop-in replacement, so the code is the same. Don't install both in the same Python.

Options:

```
python particle_field.py --particles 30000 --size 1600x900
python particle_field.py --fullscreen
python particle_field.py --color "#ff3366"
```

## Controls

| Input | What it does |
| --- | --- |
| Mouse move | Push the particles. Hold the left button to push harder. |
| M | Switch between repel and attract |
| 1 2 3 4 | Color sets: aurora, ember, ultraviolet, mono |
| 5 | Your own color |
| Hue bar (bottom-left) | Click or drag to pick a color |
| Mouse wheel, Z / X | Rotate the hue |
| R | Rainbow mode, the hue drifts on its own |
| S | Change the home shape: grid, rings, spiral |
| Space | Scatter the particles |
| [ and ] | Smaller / larger pointer radius |
| , and . | Shorter / longer trails |
| Up / Down | More / fewer particles |
| Left / Right | Less / more turbulence |
| H | Hide or show the text overlay |
| F | Fullscreen |
| Esc or Q | Quit |

If you leave the mouse alone for 4 seconds, a ghost cursor takes over and keeps the particles moving.

## How it works

Every particle is six numbers: position (`px`, `py`), speed (`vx`, `vy`) and a home spot (`hx`, `hy`). They live in NumPy arrays, so one line of math updates all of them at once. Each frame:

1. Particles near the mouse are pushed away. The push is strongest at the mouse and fades to zero at the edge of the radius.
2. The mouse's own speed is added to nearby particles, so fast movement drags them along.
3. A slow noise field made from sine waves adds gentle drift.
4. A spring pulls every particle back toward its home spot.
5. Friction slows everything down, and a speed limit stops particles flying off.
6. Particles move, and each one is drawn in a color picked from its speed. Old frames fade out slowly, which gives the trails.

To get a feel for it, change one number in this line and run it again:

```python
radius, force, flow, turb, spring = 170.0, 1.2, 0.55, 0.18, 0.004
```

## Credits

- Idea and effect: [Anya Maryina's TouchDesigner tutorial](https://www.youtube.com/watch?v=Jz-irdEYUZM)
- Built with help from Claude (Anthropic)

## License

MIT, see [LICENSE](LICENSE).
