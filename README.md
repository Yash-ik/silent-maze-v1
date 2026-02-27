# 🌫️ Silent Maze

A first-person psychological horror maze game built with Python and Pygame, inspired by the atmosphere of *Silent Hill*. Navigate a procedurally generated maze, avoid monsters, and escape before your sanity runs out.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Pygame](https://img.shields.io/badge/Pygame-2.x-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange)

---

## 🎮 Features

- **Raycasting renderer** — classic pseudo-3D engine at 1280×720
- **Procedurally generated maze** — unique layout every run, with open rooms, long corridors, and scattered pillars
- **Dynamic monster AI** — enemies path-find toward the player using BFS, with difficulty-scaled speed and sight range
- **Sanity system** — prolonged exposure to monsters drains sanity, warping your vision and amplifying horror effects
- **Procedural audio** — all sound effects (heartbeat, growls, breathing, footsteps, whispers) are synthesized at runtime using NumPy — no audio files required
- **Three difficulty modes** with distinct challenge profiles
- **Atmospheric effects** — fog, screen shake, film grain, floating embers, vignette, and glitch effects on death
- **Minimap** — toggleable in-game overlay

---

## 📋 Requirements

- Python 3.8+
- [Pygame](https://www.pygame.org/) 2.x
- [NumPy](https://numpy.org/)

---

## 🚀 Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/your-username/silent-maze.git
cd silent-maze

# 2. Install dependencies
pip install pygame numpy

# 3. Run the game
python silent_hill_game.py
```

---

## 🕹️ Controls

| Key / Input | Action |
|---|---|
| `W A S D` | Move forward / strafe left / move back / strafe right |
| `Mouse` | Look left / right |
| `M` | Toggle minimap |
| `R` | Restart (after death or win) |
| `ESC` | Quit |

---

## ⚙️ Difficulty Modes

| Mode | Monsters | Speed | Fog | Notes |
|---|---|---|---|---|
| **Easy** | 1 | Slow | Light | Forgiving, good for first-timers |
| **Hard** | 3 | Fast | Heavy | Warning sounds when monsters approach |
| **Nightmare** | 4 | Near-sprint | Near-blind | Darkness closes in; vignette intensifies |

Select your difficulty from the main menu before starting.

---

## 🗺️ Objective

Find the **exit door** and escape the maze. The exit is marked on your minimap. Avoid the monsters — if one catches you, it's over.

---

## 🔊 Audio

All sounds are generated procedurally at startup using NumPy — no external audio files are needed. Effects include:

- Heartbeat (pulses faster as danger increases)
- Monster growl, hiss, and stalking sounds
- Ambient whispers and rumble
- Breathing and footstep effects

You can toggle sound on/off from the **Settings** menu.

---

## 📁 Project Structure

```
silent-maze/
└── silent_hill_game.py   # Single-file game — everything lives here
```

---

## 🛠️ Built With

- [`pygame`](https://www.pygame.org/) — window, rendering, input, and audio playback
- [`numpy`](https://numpy.org/) — sound synthesis and fast particle/grain updates

---

## 📄 License

This project is released under the [MIT License](LICENSE).
