# Quick Start Guide — Apollo Lunar Lander AI

## Prerequisites

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, PyTorch, numpy, gymnasium[box2d], pygame

## Play the Game

### With AI Autopilot (Single Agent)

```bash
python apollolandergame_with_ai.py
```

Press **A** to toggle AI autopilot on/off. The default agent (seed 999) loads automatically.

### With Triple Ensemble Voting

```bash
python apollolandergame_with_ai.py luna --models 456 123 1111
```

The first seed (456) is the master/tie-breaker. The HUD shows vote agreement:
- **Green (3/3)** — all 3 agents agree (unanimous)
- **Yellow (2/3)** — 2 agents agree (majority)
- **Red (1/3\*)** — all differ, master decides

### Manual Control Only

```bash
python apollolandergame.py
```

## Auto-Run Evaluation

Run N episodes automatically with logging and a summary report:

```bash
# Ensemble — 100 episodes
python apollolandergame_with_ai.py luna --models 456 123 1111 --episodes=100

# Single agent — 50 episodes
python apollolandergame_with_ai.py luna --episodes=50
```

## Train a New Agent

Train the full 3-stage curriculum for a seed:

```bash
python train_in_game.py --seed 456 --episodes 5000
```

**Stages:**
1. **Stabilize + Hover** (5 actions) — learn upright orientation and altitude control
2. **Approach Pad** (7 actions) — navigate toward target pad, controlled descent
3. **Precision Landing** (7 actions) — touch down on pad

Each stage auto-graduates when success thresholds are met, then loads weights into the next stage.

### Training Options

| Flag | Description |
|------|-------------|
| `--seed N` | Random seed (default: 42) |
| `--stage N` | Start from stage N (1-3, default: 1) |
| `--episodes N` | Max episodes per stage (default: 5000) |
| `--single-stage` | Only train the specified stage |
| `--visualize` | Render training live (slower) |

### Retrain a Single Stage

```bash
python train_in_game.py --seed 456 --stage 3 --episodes 5000 --single-stage
```

## Evaluate a Trained Agent

```bash
python evaluate_stage3.py --seed 456 --episodes 100
python evaluate_stage3.py --seed 456 --episodes 10 --visualize
```

## Game Controls

| Key | Action |
|-----|--------|
| **A** | Toggle AI autopilot on/off |
| **Tab / Shift-Tab** | Cycle target landing pad |
| **Up / Down** | Throttle +/- 5% |
| **Numpad 7 / Home** | RCS rotate left |
| **Numpad 9 / PgUp** | RCS rotate right |
| **Numpad 4 / Left** | Translate left |
| **Numpad 6 / Right** | Translate right |
| **R** | Reset / new game |
| **ESC** | Quit |

## Trained Models

Pre-trained models are in `models/`. Best performers (100-episode headless eval):

| Seed | Landing | On-Target | Avg Speed |
|------|---------|-----------|-----------|
| 456 | 99% | 99% | 1.06 m/s |
| 123 | 99% | 98% | 1.27 m/s |
| 1111 | 95% | 89% | 1.61 m/s |

Recommended ensemble: `--models 456 123 1111`

## Planets

Available: `mercury`, `venus`, `earth`, `luna` (default), `mars`, `jupiter`, `saturn`, `uranus`, `neptune`

```bash
python apollolandergame_with_ai.py mars --models 456 123 1111
```

## Terrain Difficulty

```bash
python apollolandergame_with_ai.py luna --difficulty=1   # 6 pads (easy)
python apollolandergame_with_ai.py luna --difficulty=2   # 4 pads (medium)
python apollolandergame_with_ai.py luna --difficulty=3   # 2 pads (hard)
```
