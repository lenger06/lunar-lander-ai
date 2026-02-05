# Apollo Lunar Lander AI — Project Plan

## Current Status

The core system is complete and operational:

- **Double DQN agents** trained through a 3-stage curriculum (stabilize → approach → land)
- **Triple ensemble voting** implemented for inference (majority vote with master tie-breaker)
- **9 seeds trained** (42, 123, 456, 789, 999, 1111, 2222, 3333, 5555) with full Stage 3 models
- **Recommended ensemble**: seeds 456, 123, 1111 (99%, 99%, 95% landing rates)
- **Game** supports single-agent, ensemble, and auto-run evaluation modes

## Project Structure

```
lunar-lander-ai/
│
├── # ── Core AI ──────────────────────────────────────────────
│
├── double_dqn_agent.py          # Double DQN agent (QNetwork, ReplayBuffer, act/learn/save/load)
│                                #   Network: 9 → 256 → 256 → 128 → N actions
│                                #   Soft target updates (tau=1e-3), gradient clipping (1.0)
│
├── triple_dqn_ensemble.py       # Training-time ensemble (shared replay buffer, coordinated training)
│                                #   Used during training only — game uses inline voting logic
│
├── spring_rewards.py            # Potential-based reward shaping engine
│                                #   Springs: rotation (3-point), horizontal, vertical (quadratic/log)
│                                #   Dampers: angular velocity, descent profile, ascent penalty, horizontal velocity
│
├── apollo_lander_env.py         # Gymnasium environment wrapper (CurriculumGameEnv)
│                                #   Wraps game physics for RL training with identical Box2D settings
│
├── # ── Training & Evaluation ────────────────────────────────
│
├── train_in_game.py             # Main training script — 3-stage curriculum
│                                #   Stage 1: Stabilize + hover (5 actions, rotation + throttle)
│                                #   Stage 2: Approach pad (7 actions, adds lateral translation)
│                                #   Stage 3: Precision landing (7 actions, log vertical spring)
│                                #   Auto-graduation based on success rate thresholds
│
├── evaluate_stage3.py           # Single-seed headless evaluation (N episodes, eps=0)
│                                #   Reports: landing/crash rates, speed, angle, distance, fuel, quality
│
├── evaluate_seeds.py            # Multi-seed batch evaluation and ranking
│
├── # ── Game ─────────────────────────────────────────────────
│
├── apollolandergame_with_ai.py  # Main game with AI/ensemble autopilot + auto-run evaluation
│                                #   CLI: --models SEED1 SEED2 SEED3 (ensemble with master tie-break)
│                                #   CLI: --episodes=N (auto-run N episodes with logging + summary)
│                                #   CLI: --difficulty=1|2|3 (terrain pad count)
│                                #   HUD: vote agreement display (green/yellow/red)
│
├── apollolandergame.py          # Game — manual control only (no AI)
│
├── # ── Game Physics & Rendering ─────────────────────────────
│
├── apollolander.py              # Lunar Module orchestrator (descent + ascent stages, separation)
├── apollo_descent_module.py     # Descent stage physics & rendering (Box2D body, legs, engine bell)
├── apollo_ascent_module.py      # Ascent stage physics & rendering (cabin, windows, docking port)
├── apollo_rcs_pod.py            # RCS thruster pod (quad thrusters with visual effects)
├── apollocsm.py                 # Command/Service Module orchestrator
├── apollo_command_module.py     # Command module (cone geometry, heat shield)
├── apollo_service_module.py     # Service module (SPS engine with gimbal, RCS)
├── apollo_csm_module.py         # Combined CSM module coordinator
│
├── apollo_terrain.py            # Procedural terrain generation (3 difficulty levels, landing pads)
│                                #   Difficulty 1: 6 pads, Difficulty 2: 4 pads, Difficulty 3: 2 pads
├── apollo_hud.py                # HUD display (altitude, velocity, fuel, throttle, attitude)
├── apollo_sky.py                # Sky rendering (stars, gradient)
├── mini_map.py                  # Mini-map overlay
├── world.py                     # World/physics management (Box2D world setup)
│
├── # ── Models ───────────────────────────────────────────────
│
├── models/                      # Trained model weights and summaries
│   ├── lateral_stage{1,2,3}_seed{N}_best.pth   # Best model per stage per seed
│   ├── lateral_stage{1,2,3}_seed{N}_final.pth   # Final model per stage per seed
│   ├── lateral_summary_seed{N}.json             # Training summary with metrics
│   └── (legacy: apollo_ddqn_*, curriculum_*, ingame_* — older training runs)
│
├── # ── Documentation ────────────────────────────────────────
│
├── README.md                    # Full documentation (architecture, training, game, commands)
├── PLAN.md                      # This file — project structure and status
├── QUICKSTART.md                # Quick-start guide
├── WORLD_ARCHITECTURE.md        # World/physics architecture notes
│
├── # ── Config ───────────────────────────────────────────────
│
├── requirements.txt             # Python dependencies (torch, numpy, gymnasium, pygame, Box2D-py)
└── .gitignore                   # Ignores venv/, __pycache__/, save dirs, IDE files
```

## Trained Seeds — Evaluation Results

100 episodes per seed, eps=0.0, headless evaluation via `evaluate_stage3.py`:

| Rank | Seed | Landing | On-Target | Crashed | Avg Speed | Soft (<1 m/s) | Avg Dist |
|------|------|---------|-----------|---------|-----------|---------------|----------|
| 1 | **456** | **99%** | **99%** | 1% | 1.06 m/s | **35%** | **1.22m** |
| 2 | **123** | **99%** | **98%** | 1% | 1.27 m/s | 25% | 4.67m |
| 3 | **1111** | 95% | 89% | 3% | 1.61 m/s | 4% | 3.82m |
| 4 | 3333 | 98% | 16% | 2% | 1.06 m/s | 20% | 14.01m |
| 5 | 2222 | 91% | 78% | 0% | 1.25 m/s | 22% | 5.74m |
| 6 | 999 | 86% | 77% | 12% | 1.63 m/s | 16% | 5.73m |

**Recommended ensemble**: `--models 456 123 1111` (top 3 by on-target landing rate)

## Architecture Overview

### Double DQN Agent
- Network: 9-dim state → 256 → 256 → 128 → N actions (5 for Stage 1, 7 for Stages 2-3)
- Experience replay buffer (100K transitions)
- Soft target network updates (tau=1e-3)
- Gradient clipping at 1.0
- Epsilon-greedy exploration with per-stage decay schedules

### 3-Stage Curriculum
1. **Stage 1 — Stabilize + Hover** (5 actions): Learn upright orientation and altitude control
2. **Stage 2 — Approach Pad** (7 actions): Navigate horizontally toward target, controlled descent
3. **Stage 3 — Precision Landing** (7 actions): Touch down on pad with log vertical spring reward

Each stage loads the previous stage's best weights. Stage 2 expands the action space from 5→7 by adding lateral translation actions.

### Triple Ensemble Voting (Inference)
- 3 independently trained agents loaded from separate seeds
- **Unanimous (3/3)**: All agree → use that action
- **Majority (2/3)**: 2 agree → use majority action
- **Split (1/3)**: All differ → master agent (first seed) decides
- NASA-style triple redundancy — robust against any single agent error

### Reward System
Potential-based reward shaping: `reward = -(E_new - E_old)`

**Springs** (position): rotation (3-point), horizontal (pad distance), vertical (altitude — quadratic or log mode)

**Dampers** (velocity): angular velocity, descent profile tracking, ascent penalty, horizontal velocity

**Terminal**: +4000 on-target landing, +500 off-target, -1500 crash, plus bonuses for uprightness, soft landing, and fuel efficiency

## Completed Milestones

- [x] Box2D physics engine with game-identical solver settings
- [x] Double DQN agent with experience replay
- [x] 3-stage curriculum training system
- [x] Spring-based potential reward shaping
- [x] Gymnasium environment wrapper (CurriculumGameEnv)
- [x] Spacecraft module refactoring (descent, ascent, CSM separated into modules)
- [x] Procedural terrain with 3 difficulty levels
- [x] 9 seeds trained through full curriculum
- [x] Triple ensemble voting for inference
- [x] Auto-run evaluation mode with episode logging and summary
- [x] HUD vote agreement display
- [x] Headless evaluation scripts (single-seed and multi-seed)

## Potential Future Work

- [ ] Train additional seeds to find more 99%+ performers
- [ ] Multi-planet ensemble evaluation (beyond Luna)
- [ ] Fuel-constrained training scenarios
- [ ] Variable wind/disturbance training for robustness
- [ ] Ascent stage training (surface → orbit)
- [ ] Full mission: descent → surface ops → ascent → rendezvous
