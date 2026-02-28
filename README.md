# Apollo Lunar Lander AI

A reinforcement learning system that trains Double DQN agents to land an Apollo-style lunar module on target landing pads. Agents are trained using a 3-stage curriculum with spring-based potential reward shaping, then deployed in a playable pygame game as either a single AI autopilot or a **variable-size ensemble with majority voting**.

https://github.com/lenger06/lunar-lander-ai/raw/main/screenshots/Screen%20Recording%202026-02-27%20203651.mp4

<p align="center">
  <em>AI autopilot controlled flight and landings — 7-model ensemble (456, 3333, 7777, 456, 1337, 9012, 6789)</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot 2026-02-28 085855.png" width="49%" alt="Lander approach to pad">
  <img src="screenshots/Screenshot 2026-02-28 090021.png" width="49%" alt="Final landing adjustments">
</p>
<p align="center">
  <em>Left: Lander approach &nbsp;|&nbsp; Right: Final landing adjustments</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot 2026-02-28 090056.png" width="49%" alt="Near surface contact">
  <img src="screenshots/Screenshot 2026-02-28 090326.png" width="49%" alt="Surface contact and automatic engine cut-off">
</p>
<p align="center">
  <em>Left: Near surface contact &nbsp;|&nbsp; Right: Surface contact and automatic engine cut-off</em>
</p>

## Architecture

### Double DQN Agent

Each agent is a **Double Deep Q-Network (Double DQN)** with experience replay:

- Network: 9-dim input → 256 → 256 → 128 → N actions
- Soft target network updates (tau=1e-3), gradient clipping at 1.0
- Epsilon-greedy exploration with per-stage decay schedules

Multiple agents are trained independently with different random seeds, then the best are combined into an ensemble for inference.

### Ensemble Majority Voting

The ensemble system loads an odd number (3, 5, 7, ...) of independently trained agents and selects actions via majority vote:

- **Unanimous (N/N):** All agents agree → use that action
- **Majority (M/N):** A majority agrees → use the majority action
- **No majority (M/N\*):** No majority exists → the **master agent** (first seed specified) decides

The first model specified on the command line is the master/tie-breaker. Seeds can be repeated to give a model extra voting weight (e.g., listing seed 456 twice gives it 2 out of 5 votes). This provides NASA-style redundancy — the system is robust against any single agent making a bad decision.

### Recommended Ensemble: Seeds 456, 3333, 7777, 456, 1337, 9012, 6789

A 7-agent ensemble with seed 456 doubled as master for approach precision anchoring.

```bash
python apollolandergame_with_ai.py --models 456 3333 7777 456 1337 9012 6789 --episodes=1000
```

#### Performance (1,000 episodes)

| Metric | Result |
|--------|--------|
| **Landing rate** | **99.9%** (999 / 1,000) |
| **On-target landings** | **99.7%** (997 / 1,000) |
| **Crash rate** | 0.1% (1 / 1,000) |

#### Landing Quality

| Metric | Average | Max |
|--------|---------|-----|
| Landing speed | 0.08 m/s | 1.26 m/s |
| Landing angle | 0.6° | 13.1° |
| Distance from target | 1.13 m | 13.95 m |
| Fuel remaining | 65.4% | — |
| Steps to land | 1,946 | 3,538 |

#### Landing Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| Soft (<1 m/s) | 958 | 95.9% |
| Gentle (1–2 m/s) | 41 | 4.1% |
| Hard (>2 m/s) | 0 | 0.0% |

#### Ensemble Design Rationale

| Seed | Role | S2 Final | S3 Final | S3 Best |
|------|------|----------|----------|---------|
| 456 (×2) | Master + approach anchor | 98% | 92% | 100% |
| 3333 | Best overall finisher | 96% | 100% | 100% |
| 7777 | Strong finisher, dynamic-start trained | 89% | 96% | 100% |
| 1337 | Strong all-round | 97% | 94% | 100% |
| 9012 | Near-perfect finisher | 95% | 99% | 100% |
| 6789 | Consistent high finisher | 89% | 98% | 99% |

Doubling seed 456 gives it 2 of 7 votes, anchoring approach decisions with its top-tier S2 accuracy (98%). The remaining five models all achieve ≥94% S3 Final rates, providing strong crash resistance. 9012 (99% final) and 6789 (98% final) are the most consistent finishers; 7777 was trained with dynamic Stage 1 starting conditions matching the game's extreme initial angles (±90°). The sole failure in 1,000 episodes was a crash from a −83.7° initial angle that converted recovered angular momentum into uncontrollable lateral velocity.

---

### Ensemble Testing History

All evaluations run on `luna` with random initial angles up to ±90°, random lateral velocity ±3 m/s, and random descent rate 2–5 m/s.

| Ensemble (seeds) | Size | Episodes | On-target | Crashes | Notes |
|------------------|------|----------|-----------|---------|-------|
| 456, 3333, 7654, 456, 1337 | 5 | 500 | 99.4% | — | First strong result |
| 456, 3333, 6789, 456, 1337 | 5 | 400 | ~99% | 1 | 1 crash + few off-target |
| 456, 3333, 7654, 456, 1337, 9012, 6789 | 7 | 1,000 | 99.6% | 2 | Both crashes on-pad |
| 3333, 9012, 6789, 7777, 3141, 1111, 1337 | 7 | ~400 | <99% | low | Significant off-target — no double-456 |
| **456, 3333, 7777, 456, 1337, 9012, 6789** | **7** | **1,000** | **99.7%** | **1** | **Current best** |
| 456, 3333, 7777, 456, 1337, 9012, 6789, 3141, 1111 | 9 | ~3 | — | 1 | Crashed ep 3, abandoned |

**Key lessons:**
- Double-456 is essential for on-target accuracy — removing it causes significant off-target rate
- Replacing 7654 (85% S3 final) with 7777 (96% S3 final) reduced crashes from 2 → 1
- 9-model ensemble with 3141/1111 added performed worse than 7-model in early testing
- The single remaining crash is from extreme initial angle (−83.7°) — a near-horizontal start that the AI recovers from rotationally but loses lateral control

### State Space (9 dimensions)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | X position relative to target pad | / (world_width / 2) |
| 1 | Y position relative to target pad | / 50.0 |
| 2 | X velocity | / 10.0 |
| 3 | Y velocity | / 10.0 |
| 4 | Angle (continuous, normalized) | atan2(sin, cos) |
| 5 | Angular velocity (negated) | clipped to [-5, 5] |
| 6 | Left leg contact | binary (0 or 1) |
| 7 | Right leg contact | binary (0 or 1) |
| 8 | Throttle level | 0.1 to 1.0 |

### Action Space

| Action | ID | Available |
|--------|----|-----------|
| NOOP | 0 | All stages |
| RCS Rotate Left | 1 | All stages |
| Throttle Up (+5%) | 2 | All stages |
| Throttle Down (-5%) | 3 | All stages |
| RCS Rotate Right | 4 | All stages |
| Translate Left | 5 | Stages 2-3 (7-action) |
| Translate Right | 6 | Stages 2-3 (7-action) |

The main engine fires continuously at the current throttle level every step. Actions adjust throttle or fire RCS thrusters. Both AI and manual control use identical physics (15x boosted RCS thrust for meaningful control authority).

### Physics

Uses Box2D with game-identical solver settings (10 velocity iterations, 10 position iterations, 1/60s timestep). The training environment (`CurriculumGameEnv`) runs the same physics as the actual game, ensuring trained behavior transfers directly.

## Physics Accuracy: Game vs Real Apollo

The game physics are modeled on the real Apollo Lunar Module and Moon. Most parameters are taken directly from NASA specifications for the Apollo 11 mission.

### Moon

| Parameter | Game | Real Moon | Accuracy |
|-----------|------|-----------|----------|
| Surface gravity | 1.62 m/s² | 1.62 m/s² | Exact |
| Atmosphere/damping | 0.0 (vacuum) | 0.0 (vacuum) | Exact |
| Diameter | 3,475 km | 3,474 km | Exact |
| Low orbit altitude | 100 km | 110 km (60 nmi) | Close |

### Lunar Module Mass

| Parameter | Game | Real (Apollo 11) | Accuracy |
|-----------|------|-------------------|----------|
| Descent stage dry mass | 2,034 kg | 2,034 kg | Exact |
| Descent fuel capacity | 8,200 kg | 8,248 kg | ~99% |
| Descent stage total | 10,234 kg | 10,282 kg | ~99% |
| Ascent stage dry mass | 2,445 kg | 2,445 kg | Exact |
| Ascent fuel capacity | 2,353 kg | 2,376 kg | ~99% |
| Ascent stage total | 4,798 kg | 4,821 kg | ~99% |
| **Full LM total** | **15,032 kg** | **15,103 kg** | **~99%** |

### Engines

| Parameter | Game | Real LM | Accuracy |
|-----------|------|---------|----------|
| DPS max thrust | 45,040 N | 45,040 N (10,125 lbf) | Exact |
| DPS throttle range | 10% – 100% | ~10% – 100% | Exact |
| DPS gimbal range | ±6° | ±6° | Exact |
| DPS specific impulse | ~311 s | 299–305 s | ~2% high |
| DPS fuel consumption | 14.77 kg/s | ~15.06 kg/s | ~98% |
| APS thrust | 15,600 N | 15,600 N (3,500 lbf) | Exact |
| APS type | Fixed thrust | Fixed, non-gimbaled | Match |
| APS fuel consumption | 5.12 kg/s | ~5.2 kg/s | ~98% |

The real DPS had a restricted throttle band (65%–92.5% avoided due to nozzle erosion). The game does not model this restriction.

### Reaction Control System (RCS)

| Parameter | Game | Real LM | Notes |
|-----------|------|---------|-------|
| Thrust per thruster | 445 N | 445 N (100 lbf) | Exact base value |
| Configuration | 4 pods × 4 thrusters | 4 pods × 4 thrusters | Exact |
| **Game boost** | **15×** | 1× | Gameplay scaling |
| Effective per-thruster | 6,675 N | 445 N | 15× stronger |
| **Thrust model** | **Fixed (66.75 force units)** | Mass-proportional | Consistent across all flight phases |

The 15× RCS boost is the largest deliberate deviation from reality. At real thrust levels, RCS provides very gentle rotational authority — appropriate for the real LM's slow orbital descent over many minutes, but too sluggish for a game where responsive control is needed in a short time window.

Both the manual game (`apollolandergame.py`) and AI game now use identical fixed RCS thrust values (15× × 445 N / 100 = 66.75 force units per thruster), ensuring consistent control authority regardless of vehicle mass. This matches the AI training physics and provides responsive control throughout all mission phases (descent, ascent, CSM operations).

### Dimensions

| Parameter | Game | Real LM | Accuracy |
|-----------|------|---------|----------|
| Descent stage width | 4.2 m | 4.2 m | Exact |
| Descent stage height | 1.6 m | 1.7 m | ~94% |
| Landing leg span | ~7.6 m | 9.4 m (diagonal) | ~81% |
| Foot pad width | 0.9 m | ~0.9 m | Match |
| Foot pad friction | 1.4 | N/A (lunar regolith) | Tuned for gameplay |

### Lunar Surface Sensing Probes (Contact Probes)

| Parameter | Game | Real LM | Notes |
|-----------|------|---------|-------|
| Probe length | 1.7 m | 1.7 m (67 inches) | Exact |
| Number of probes | 2 (visible legs) | 3 (not on ladder leg) | Simplified for 2D |
| Probe orientation | Always vertical | Always vertical | Gravity-fed design |
| Contact detection | Auto engine cutoff | "Contact light" + manual cutoff | Automated in game |

The contact probes extend below each foot pad and always hang straight down regardless of lander orientation (simulating gravity-fed probes). When either probe touches the terrain, a cyan "CONTACT" indicator appears on the HUD and the descent engine automatically shuts off — matching the real Apollo procedure where crew would immediately cut the engine upon Buzz Aldrin's famous "Contact light!" callout.

### Thrust-to-Weight Ratio

| Condition | Game | Real LM |
|-----------|------|---------|
| Full fuel (start of descent) | 2.71 | 1.84 |
| Near empty (landing) | ~2.71 | ~3.66 |

The game uses a constant TWR of 2.71 (`thrust_factor = gravity × 2.71`). The real LM's TWR changed dramatically as fuel burned off — starting around 1.84 at the beginning of powered descent and climbing to ~3.66 near touchdown. The game's value sits in the middle. The real LM started powered descent from 15 km altitude at orbital velocity, burning most of its fuel for deceleration. The game starts at ~50–100 m altitude, so the higher constant TWR provides appropriate control authority for the shorter hover-and-land scenario.

### Fuel System

The project has two game versions with different fuel models:

#### Manual Game (`apollolandergame.py`) — NASA-Accurate

| Parameter | Game | Real Apollo | Accuracy |
|-----------|------|-------------|----------|
| Descent fuel capacity | 8,248 kg | 8,248 kg | Exact |
| Descent dry mass | 1,998 kg | 2,034 kg | ~98% |
| Burn rate formula | `thrust / (Isp × g₀)` | `thrust / (Isp × g₀)` | Same method |
| Burn rate at full throttle | 14.77 kg/s | ~15.06 kg/s (Isp 305 s) | ~98% |
| Burn time at 100% throttle | 558 sec (9.3 min) | ~547 sec (9.1 min) | ~98% |
| Dynamic mass update | Yes — mass decreases as fuel burns | Yes | Match |
| TWR changes with fuel | Yes | Yes | Match |

Uses Isp = 311 s vs the real 299–305 s, giving ~2% lower consumption. Lander mass is updated every frame as fuel depletes, so TWR increases naturally toward touchdown — matching real Apollo behavior.

#### AI Game (`apollolandergame_with_ai.py`) — Terminal Descent Budget

The AI game represents only the **terminal descent phase** (~150 m altitude onward), not the full powered descent from orbit. The fuel budget is calibrated to match what would realistically remain at the start of terminal descent.

| Parameter | AI Game | Realistic terminal descent | Notes |
|-----------|---------|---------------------------|-------|
| Fuel capacity | ~250 units (3.03× real scaled) | ~10% of full tank remains | Apollo 11 landed with ~25 sec of fuel |
| Burn rate | 0.12 units/frame at 100% | — | Compressed for ~30–60 sec scenario |
| Avg fuel remaining at touchdown | ~55% of terminal budget | 14–51% (mission dependent) | Apollo 14 ~49%, Apollo 11 ~14% |
| Dynamic mass update | No — mass stays constant | Yes | TWR fixed at 2.71 throughout |

The 3.03× multiplier gives ~250 total units, matching the training environment exactly. The AI typically lands with ~55% remaining — analogous to a well-flown Apollo 14-style descent. The single remaining crash (extreme −83.7° initial angle) still had 82.6% fuel at impact, confirming fuel was not a factor. A QUANTITY warning light triggers (and is logged) when either propellant drops below 5% of the unscaled real-Apollo maximum.

### Propellant System & Center of Gravity

Both game versions model separate fuel and oxidizer tanks with realistic Apollo LM characteristics:

#### Propellant Tanks

| Component | Mass | Percentage | Notes |
|-----------|------|------------|-------|
| Fuel (Aerozine 50) | 3,164 kg | 38.4% | Hypergolic fuel |
| Oxidizer (N2O4) | 5,084 kg | 61.6% | Nitrogen tetroxide |
| **Total Propellant** | **8,248 kg** | 100% | |
| Mixture ratio | 1.6:1 | (oxidizer:fuel) | By mass consumption |

The fuel and oxidizer are consumed at the 1.6:1 mixture ratio, so both tanks deplete at the same rate (percentage-wise). The HUD displays separate FUEL and OXID gauges.

#### Center of Gravity Dynamics

| Parameter | Game | Real Apollo LM | Notes |
|-----------|------|----------------|-------|
| Dry structure CG offset | 0.15 m | Variable | Equipment asymmetry |
| Propellant CG | Centered on thrust axis | ~Centered | Symmetric tank arrangement |
| CG shift direction | Toward dry structure as fuel depletes | Same behavior | Weighted average |

As propellant is consumed, the center of gravity shifts toward the dry structure's offset position. With full tanks (~8,248 kg propellant vs ~2,034 kg dry mass), the propellant mass dominates and CG is near the thrust axis. As propellant depletes, the weighted average CG moves toward the dry structure's offset (0.15 m right of thrust axis in the game).

#### Automatic Gimbal Compensation

The descent engine automatically gimbals to keep the thrust vector passing through the shifting CG — matching the real Apollo LM's DECA (Descent Engine Control Assembly):

| Parameter | Game | Real Apollo LM |
|-----------|------|----------------|
| Auto-gimbal | Always on | Always on (DECA) |
| Gimbal range | ±6° | ±6° |
| Control gain | 2.0 | Continuous servo |
| Deadband | 0.01 m | Small deadband |

The HUD displays gimbal breakdown: total gimbal angle, auto-gimbal component (CG compensation), and manual trim (player input via Comma/Period keys). The CG indicator shows current lateral CG offset.

### Summary

| Category | Accuracy |
|----------|----------|
| Moon gravity | Exact |
| Lander masses | >99% (Apollo 11 values) |
| Engine thrust values | Exact (DPS + APS) |
| Throttle range & gimbal | Exact (±6°, 10–100%) |
| Fuel consumption rates | ~98% |
| Descent stage dimensions | Exact (4.2 m width) |
| RCS base thrust | Exact (445 N), then 15× boosted |
| TWR profile | Constant 2.71 vs variable 1.84–3.66 |
| Ascent stage dimensions | ~50% scale (visual choice) |
| Gameplay fuel budget | ~1% of real (appropriate for scenario) |

### Controls Comparison

The real Apollo LM was controlled by two hand controllers, a guidance computer (LGC), and associated mode switches. The game simplifies this into discrete key presses (manual) or single-action-per-frame decisions (AI).

#### Attitude Control (Rotation)

| | Game — Manual | Game — AI | Real Apollo LM |
|-|---------------|-----------|----------------|
| **Input device** | Numpad 7/Home, Numpad 9/PgUp | Actions 1 and 4 | ACA (Attitude Controller Assembly) — 3-axis proportional joystick |
| **Input type** | Discrete key press | One action per frame (60 Hz) | Proportional — ±10° deflection with 1° deadband |
| **Thruster firing** | Coupled pair: left pod + right pod fire in opposing vertical directions | Same coupled pair logic | 16 RCS jets (4 pods × 4 jets), jet-select logic computed by LGC or ATCA |
| **Control modes** | Direct thruster firing only | Direct thruster firing only | Rate Command/Attitude Hold (ACA displacement = rate; release = hold), Pulse Mode (minimum impulse per deflection), or full Auto (LGC controls) |
| **Attitude hold** | No — rotation continues until countered | No — agent must learn to counter | Yes — LGC/ATCA automatically holds attitude when ACA is in detent |

#### Translation (Lateral Movement)

| | Game — Manual | Game — AI | Real Apollo LM |
|-|---------------|-----------|----------------|
| **Input device** | Numpad 4/Left, Numpad 6/Right | Actions 5 and 6 (7-action models) | TTCA (Thrust/Translation Controller Assembly) — 3-axis T-handle |
| **Axes** | Lateral only (left/right) | Lateral only (left/right) | All 3 axes (X, Y, Z translation) |
| **Thruster firing** | Side thrusters on one pod | Both pods fire in same direction | RCS jets selected by LGC/ATCA for requested translation axis |
| **Available** | Always | Only in 7-action models (Stages 2–3) | Always (TTCA jets position) |

#### Throttle (Descent Engine)

| | Game — Manual | Game — AI | Real Apollo LM |
|-|---------------|-----------|----------------|
| **Input device** | Up/Down arrow keys | Actions 2 and 3 | TTCA X-axis (throttle position) |
| **Input type** | Discrete 5% steps | Discrete 5% steps | Continuous — proportional to handle deflection |
| **Range** | 10% – 100% | 10% – 100% | 10% – 92.5% (with restricted band at 65%–92.5%) |
| **Soft stop** | No | No | Yes — at ~53% deflection for fine control during landing |
| **Auto mode** | No | No | Yes — LGC commands throttle in AUTO mode; manual input can be summed with or override LGC commands |
| **Engine state** | Always on | Always on | Start/stop via ENGINE START/STOP buttons |

#### Gimbal (Engine Thrust Vectoring)

| | Game — Manual | Game — AI | Real Apollo LM |
|-|---------------|-----------|----------------|
| **Control** | Auto-gimbal + manual trim (Comma/Period/Slash) | Not available (fixed at 0°) | Automatic — DECA trims gimbal to keep thrust through CG |
| **Range** | ±6° | N/A | ±6° in pitch and roll |
| **Auto-gimbal** | Yes — compensates for CG shift | N/A | Yes — DECA (Descent Engine Control Assembly) |
| **Manual trim** | ±1° per key press, additive with auto | N/A | Available via hand controller |
| **Purpose** | CG compensation (auto) + pilot fine-tuning (manual) | N/A | CG offset compensation (automatic trim) |

#### Guidance Computer

| | Game — Manual | Game — AI | Real Apollo LM |
|-|---------------|-----------|----------------|
| **Computer** | None | Double DQN neural network | Apollo Guidance Computer (AGC/LGC) with DSKY interface |
| **Backup** | N/A | Triple ensemble voting | AGS (Abort Guidance System) — independent backup computer |
| **Programs** | N/A | Single trained policy | P63 (braking), P64 (approach), P66 (landing phase) |
| **Landing point redesignation** | Tab/Shift-Tab to cycle pads | Target pad fixed per episode | LPD mode — astronaut nudges landing site via ACA inputs to LGC |

#### Key Differences Summary

| Feature | Game — Manual | Game — AI | Real Apollo LM |
|---------|---------------|-----------|----------------|
| Attitude hold | No | No | Yes (automatic) |
| Proportional inputs | No (discrete) | No (discrete) | Yes (continuous joystick) |
| Engine start/stop | Always on | Always on | Pilot-controlled |
| Gimbal control | Manual | None | Automatic (LGC) |
| Throttle auto mode | No | No | Yes (LGC P66) |
| 3-axis translation | Lateral only | Lateral only | Full 3-axis |
| RCS thrust | 15× boosted | 15× boosted | Real (445 N/jet) |
| Control update rate | 60 Hz (every frame) | 60 Hz (every frame) | 25.6 Hz (AGC cycle) |

## Training

### 3-Stage Curriculum

Training proceeds through 3 stages, each building on the previous stage's learned weights:

#### Stage 1: Stabilize + Hover (5 actions)

**Goal:** Learn to stay upright and maintain altitude using rotation RCS and throttle only.

| Parameter | Value |
|-----------|-------|
| Actions | 5 (no lateral translation) |
| k_rotation | 25.0 |
| c_angular | 40.0 |
| c_descent | 2.0 |
| c_ascend | 4.0 |
| Max steps | 1500 |
| Graduation | 95% success, min 200 episodes |
| Learning rate | 5e-4 |
| Epsilon | 1.0 → 0.05 (decay 0.997) |

**Success criteria:** Angle < 0.35 rad, angular velocity < 0.5 rad/s, vertical velocity < 3.0 m/s.

#### Stage 2: Approach Pad (7 actions)

**Goal:** Navigate horizontally toward the target pad while descending. Introduces lateral RCS translation.

| Parameter | Value |
|-----------|-------|
| Actions | 7 (adds Translate L/R) |
| k_rotation | 20.0 |
| k_horizontal | 3.0 |
| k_vertical | 1.2 (quadratic mode) |
| c_angular | 30.0 |
| c_descent | 3.0 |
| c_ascend | 5.0 |
| proximity_gain | 5.0 |
| Max steps | 2000 |
| Graduation | 70% success, min 1000 episodes |
| Learning rate | 3e-4 |
| Epsilon | 0.3 → 0.02 (decay 0.998) |

**Success criteria:** Angle < 0.26 rad, vertical velocity < 2.0 m/s, horizontal distance to pad < 15m, angular velocity < 0.3 rad/s, altitude < 20m. Loads Stage 1 best weights with action space expanded from 5 to 7.

#### Stage 3: Precision Landing (7 actions)

**Goal:** Actually touch down on the pad. Uses logarithmic vertical spring so reward increases near the surface, encouraging commitment to landing.

| Parameter | Value |
|-----------|-------|
| Actions | 7 |
| k_rotation | 20.0 |
| k_horizontal | 3.0 |
| k_vertical | 2.0 (log mode) |
| vertical_log_scale | 50.0 |
| c_descent | 5.0 |
| c_ascend | 5.0 |
| time_penalty | 0.3 |
| Max steps | 3000 |
| Graduation | 50% success, min 1500 episodes |
| Learning rate | 1e-4 |
| Epsilon | 0.15 → 0.01 (decay 0.999) |

**Success criteria:** Physical touchdown detected (legs contact terrain with speed < 3 m/s and angle < 30 degrees). Loads Stage 2 best weights.

### Training Commands

Train the full 3-stage curriculum for a seed:

```bash
python train_in_game.py --seed 456 --episodes 5000
```

Train only a specific stage (e.g., retrain Stage 3 with existing Stage 2 weights):

```bash
python train_in_game.py --seed 456 --stage 3 --episodes 5000 --single-stage
```

Watch training live with visualization:

```bash
python train_in_game.py --seed 456 --episodes 5000 --visualize
```

**Options:**

| Flag | Description |
|------|-------------|
| `--seed N` | Random seed (default: 42) |
| `--stage N` | Start from stage N (1-3, default: 1) |
| `--episodes N` | Max episodes per stage (default: 5000) |
| `--single-stage` | Only train the specified stage |
| `--visualize` | Render training live (slower) |
| `--save-dir DIR` | Model save directory (default: models/) |

**Output files** (saved to `models/`):
- `lateral_stage{1,2,3}_seed{N}_best.pth` — Best model per stage
- `lateral_stage{1,2,3}_seed{N}_final.pth` — Final model per stage
- `lateral_summary_seed{N}.json` — Training summary with metrics

### Reward System

Rewards use **potential-based reward shaping**: `reward = -(E_new - E_old)`. Reducing total spring energy is rewarded; increasing it is penalized. This guarantees the optimal policy is preserved while providing dense learning signals.

#### Springs (Position-Based)

**A - Rotation Spring:** 3 virtual springs on an imaginary circle penalize deviation from upright. A rotation deadzone (10 degrees) allows small tilts without penalty.

**B - Horizontal Spring:** `E = 0.5 * k_horizontal * dx^2` where `dx` is distance from pad center. Pulls the lander toward the target pad.

**C - Vertical Spring** (two modes):

- **Quadratic (Stages 1-2):** `E = 0.5 * k_eff * h^2`. Gradient `dE/dh = k*h` is proportional to altitude, so reward per meter *decreases* near the surface. The `proximity_gain` parameter increases the effective spring constant at low altitude to partially compensate.

- **Logarithmic (Stage 3):** `E = k * scale * log(1 + h)`. Gradient `dE/dh = k*scale/(1+h)` is inversely proportional to altitude, so reward per meter *increases* near the surface. This creates a "commitment reward" that incentivizes the agent to push through the final meters to touchdown rather than hovering.

#### Dampers (Velocity-Based)

**D - Angular Velocity:** `E = c_angular * omega^2`. Penalizes rotational speed.

**E - Descent Profile:** Tracks a target descent rate that scales linearly with altitude: `target = descent_max_rate * (altitude / spawn_altitude)`. At 50m altitude the target is -2.5 m/s; at 5m it's -0.25 m/s. Penalizes deviating from this profile.

**F - Ascent Penalty:** `E = c_ascend * max(0, vel_y)^2`. Penalizes upward velocity (wastes fuel).

**G - Horizontal Velocity:** `E = c_horizontal_vel * vel_x^2`. Penalizes lateral speed for smooth translation.

#### Terminal Rewards

| Event | Reward |
|-------|--------|
| Landed on target pad | +4000 + 750 * pad_multiplier |
| Landed off target | +500 |
| Crashed | -1500 |
| Out of fuel | -800 |
| Upright bonus (< 0.1 rad) | +400 |
| Upright bonus (< 0.2 rad) | +200 |
| Soft landing (< 1.0 m/s) | +400 |
| Soft landing (< 2.0 m/s) | +200 |
| Fuel efficiency bonus | +200 * (fuel / max_fuel) |

A `time_penalty` is subtracted every step to discourage hovering. Stage 3 uses 0.3 (3x the default 0.1) to push the agent toward faster landings.

## Evaluation

### Single-Seed Evaluation

Evaluate a trained Stage 3 model over N episodes with no exploration (eps=0):

```bash
python evaluate_stage3.py --seed 456 --episodes 100
```

Watch the agent land:

```bash
python evaluate_stage3.py --seed 456 --episodes 10 --visualize
```

**Options:**

| Flag | Description |
|------|-------------|
| `--seed N` | Seed of the model to evaluate (default: 999) |
| `--episodes N` | Number of evaluation episodes (default: 100) |
| `--visualize` | Render episodes live |
| `--save-dir DIR` | Model directory (default: models/) |

**Reports:** Landing rate, on-target rate, crash rate, average speed/angle/steps/fuel, landing quality breakdown (soft/gentle/hard).

### Trained Model Results

All models trained with game-identical Box2D physics (solver iterations 10,10) using the 3-stage lateral RCS curriculum. Sorted by S3 Final rate.

Models marked † were trained with **dynamic Stage 1 starting conditions** (random initial angles ±90°, lateral velocity ±3 m/s) matching the game's actual spawn parameters. Earlier models used static Stage 1 starts.

| Seed | S1 Final | S2 Final | S3 Final | S3 Best | S3 Eps | Time (min) | Notes |
|------|----------|----------|----------|---------|--------|------------|-------|
| 3333 | 100% | 96% | **100%** | 100% | 1500 | 365 | Best individual — only seed with 100%/100% S3 |
| 9012 † | 95% | 95% | **99%** | 100% | 1500 | 291 | Recommended ensemble member |
| 6789 † | 95% | 89% | **98%** | 99% | 1500 | 309 | Recommended ensemble member |
| 7777 † | 95% | 89% | **96%** | 100% | 1500 | 267 | Recommended ensemble member; dynamic-start trained |
| 3141 | 100% | 86% | **96%** | 100% | 1334 | 291 | |
| 1111 | 100% | 92% | **95%** | 100% | 1500 | 336 | |
| 1337 | 100% | 97% | **94%** | 100% | 1334 | 295 | Recommended ensemble member |
| 2000 † | 96% | 95% | **94%** | 99% | 1500 | 237 | New batch, dynamic-start trained |
| 2222 | 100% | 84% | **94%** | 99% | 1500 | 306 | |
| 2718 † | 95% | 91% | **93%** | 100% | 1500 | 298 | |
| 8888 | 99% | 78% | **93%** | 100% | 1334 | 341 | |
| 2468 | 100% | 92% | **93%** | 99% | 1334 | 275 | |
| 456  | 100% | 98% | **92%** | 100% | 1500 | 335 | Recommended ensemble member (×2); top S2 approach |
| 5000 † | 95% | 96% | **91%** | 100% | 1500 | 263 | New batch, dynamic-start trained |
| 1618 † | 95% | 89% | **91%** | 98% | 1500 | 234 | |
| 1000 † | 95% | 92% | **91%** | 97% | 1500 | 210 | New batch, dynamic-start trained |
| 42   | 100% | 94% | **89%** | 100% | 1500 | 306 | |
| 789  | 100% | 96% | **87%** | 98% | 1500 | 334 | |
| 7654 | 100% | 97% | **85%** | 100% | 1334 | 256 | Fastest training |
| 5555 | 100% | 88% | **83%** | 99% | 1500 | 333 | |
| 4242 | 99% | 91% | **72%** | 91% | 1334 | 305 | |
| 123  | 100% | 99% | **68%** | 100% | 1500 | 318 | High S3 instability (100% best → 68% final) |
| 999  | 100% | 78% | — | — | — | 227 | 2-stage only (no S3) |

**Key observations:**
- **S3 Best vs S3 Final gap** indicates late-training instability — seeds with small gaps (3333, 9012, 6789) are most stable ensemble candidates
- **S2 Final** correlates with on-target accuracy — 456's 98% S2 is why it anchors the ensemble
- **Dynamic Stage 1 (†)** models were trained with the full game spawn range; these handle extreme initial angles better
- Seed 3333 is the only model to achieve 100% on both S3 Final and S3 Best
- The 4 newest seeds (1000, 2000, 5000, 7777) were trained after all training/game environment parity fixes were applied

## Playing the Game

### Single Agent Mode

Run the game with the default AI agent (seed 999):

```bash
python apollolandergame_with_ai.py [planet]
```

### Ensemble Mode

Run with ensemble voting (first seed = master/tie-breaker, odd number of seeds >= 3):

```bash
# Recommended 7-agent ensemble (99.7% on-target over 1,000 episodes)
python apollolandergame_with_ai.py [planet] --models 456 3333 7777 456 1337 9012 6789

# 3-agent ensemble
python apollolandergame_with_ai.py [planet] --models 456 3333 1337
```

Seeds can be repeated to give a model extra voting weight. The HUD displays real-time vote agreement: green (unanimous), yellow (majority), or red (no majority — master decides).

### Auto-Run Evaluation

Run N episodes automatically with logging and a summary report:

```bash
# Recommended ensemble evaluation — 100 episodes
python apollolandergame_with_ai.py luna --models 456 3333 7777 456 1337 9012 6789 --episodes=100

# Single agent evaluation — 50 episodes
python apollolandergame_with_ai.py luna --episodes=50
```

Each episode logs status (ON-TARGET / OFF-TARGET / CRASHED), speed, angle, distance to pad, fuel remaining, and step count. After all episodes, a full summary report is printed.

When a crash occurs during auto-run evaluation, detailed diagnostic data is automatically saved to `crash_logs/` including full step-by-step flight recorder data, initial conditions, terrain profile around the pad, and crash reason analysis. When the QUANTITY warning light triggers (either propellant tank below 5%), the flight state at that moment is saved to `qty_logs/` for fuel management analysis. Both counts are reported in the end-of-run summary.

### Game Options

| Flag | Description |
|------|-------------|
| `[planet]` | Planet name (default: `luna`) |
| `--models S1 S2 S3 [S4 S5 ...]` | Enable ensemble with odd N >= 3 seeds (first = master) |
| `--episodes=N` | Auto-run N episodes with AI, log results |
| `--difficulty=1\|2\|3` | Terrain difficulty (1=easy/6 pads, 2=medium/4 pads, 3=hard/2 pads) |

Available planets: `mercury`, `venus`, `earth`, `luna` (default), `mars`, `jupiter`, `saturn`, `uranus`, `neptune`.

### In-Game Controls

| Key | Action |
|-----|--------|
| **A** | Toggle AI autopilot / ensemble on/off |
| **Tab / Shift-Tab** | Cycle target landing pad |
| **Up / Down** | Throttle +/- 5% |
| **Numpad 7 / Home** | RCS rotate left |
| **Numpad 9 / PgUp** | RCS rotate right |
| **Numpad 4 / Left** | Translate left |
| **Numpad 6 / Right** | Translate right |
| **,** / **.** / **/** | Gimbal left / right / center |
| **R** | Reset / new game |
| **ESC** | Quit |

The engine is always on at the current throttle level (matches training). Both AI and manual mode use identical RCS thrust physics.

### Model Loading

The game automatically searches for the best available model per seed. Search priority:
1. `lateral_stage3_seed{N}_best.pth` (7 actions) — preferred
2. `lateral_stage3_seed{N}_final.pth` (7 actions)
3. `lateral_stage2_seed{N}_best.pth` (7 actions) — fallback
4. Legacy 5-action models — last resort

Run the game without AI (manual only):

```bash
python apollolandergame.py [planet]
```

## Visual Features

### Parallax Starfield

The game features a starfield rendered on an inner cylinder with parallax scrolling:

| Parameter | Value | Effect |
|-----------|-------|--------|
| Cylinder ratio | 10% of world width | Strong depth perception |
| Star count | 800 | Good density without performance impact |
| Parallax effect | Stars move at 10% of terrain speed | Creates sense of vast distance |

Stars are distributed across the visible altitude range and rendered as single pixels for performance. The terrain is drawn on top of the starfield, naturally occluding stars below the horizon.

### Orbiting Satellite

A Sputnik-style satellite orbits across the sky as a background element:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Altitude | 65 m | Above lander spawn (50 m) |
| Velocity | 8 m/s horizontal | Constant orbital motion |
| Body diameter | ~0.58 m | Scaled 3x for visibility |
| Antennas | 4 swept-back | 2.4 m and 2.9 m lengths (like real Sputnik) |

The satellite features a copper/gold spherical body with swept-back antennas that slowly rotate. It wraps around the cylindrical world and appears on the minimap as an orange dot.

### Apollo LM Instrument Panel

The HUD features realistic instrument gauges modeled after actual Apollo Lunar Module cockpit indicators, rendered with 3D metallic bezels, dark instrument faces, and needle/bar indicators.

| Instrument | Style | Description |
|------------|-------|-------------|
| **Propellant Quantity** | Dual vertical columns | FUEL and OXID gauges with shared housing, 0–100% scale, green needle indicators (orange < 25%, red < 10%) |
| **Range Alt / Alt Rate** | Dual vertical columns | Left column: altitude (0–500 m), Right column: vertical velocity (-30 to +30 m/s, center zero), numeric readouts at bottom |
| **Horizontal Velocity** | Horizontal bar | Center-zero bar (-10 to +10 m/s), vertical needle slides left/right, color-coded by speed |
| **Throttle** | Single vertical column | 0–100% scale with needle indicator, color shifts at high throttle |
| **CG / Gimbal** | Dual horizontal bars | CG lateral offset (top) and gimbal angle (bottom) with auto (cyan) and trim (yellow) triangle markers |
| **FDAI Attitude** | 8-ball style | Rotating horizon ball (brown ground / blue sky), fixed spacecraft symbol, roll rate bar |
| **Lunar Contact** | Indicator light panel | Cyan light illuminates on probe contact, gray housing with corner screws |
| **Quantity Warning** | Indicator light panel | Red light flashes at ~2 Hz when either fuel or oxidizer drops below 5% |

## Project Structure

```
lunar-lander-ai/
  # Core AI
  double_dqn_agent.py        # Double DQN agent (QNetwork, act, learn, save/load)
  triple_dqn_ensemble.py     # 3-agent ensemble with majority voting (training)
  spring_rewards.py          # Spring-based reward computation engine
  apollo_lander_env.py       # Gymnasium environment wrapper

  # Training
  train_in_game.py           # Main training script (3-stage curriculum)
  evaluate_stage3.py         # Stage 3 evaluation script (headless)
  evaluate_seeds.py          # Multi-seed evaluation and ranking

  # Game
  apollolandergame_with_ai.py  # Game with AI/ensemble autopilot + auto-run eval
  apollolandergame.py          # Game (manual only)
  world.py                     # World/physics management (ApolloWorld, WorldObject base)
  apollo_terrain.py            # Procedural terrain generation (3 difficulty levels)
  apollo_hud.py                # HUD display (Apollo LM-style instrument gauges)
  apollo_sky.py                # Sky rendering (parallax starfield)
  satellite.py                 # Orbiting Sputnik-style satellite
  mini_map.py                  # Mini-map overlay

  # Spacecraft Modules
  apollolander.py              # Lunar Module orchestrator (descent + ascent stages)
  apollo_descent_module.py     # Descent stage (Box2D body, legs, engine bell)
  apollo_ascent_module.py      # Ascent stage (cabin, docking port, RCS)
  apollo_rcs_pod.py            # RCS thruster pod (quad thrusters)
  apollocsm.py                 # Command/Service Module orchestrator
  apollo_command_module.py     # Command module (cone geometry, heat shield)
  apollo_service_module.py     # Service module (SPS engine, gimbal, RCS)
  apollo_csm_module.py         # CSM WorldObject wrapper

  # Models, Config & Logs
  models/                      # Saved .pth model weights and summary JSON files
  crash_logs/                  # Per-crash diagnostic JSON (auto-generated during eval)
  qty_logs/                    # Per-episode QUANTITY light trigger logs (auto-generated)
  requirements.txt             # Python dependencies
```

## Dependencies

```bash
pip install -r requirements.txt
```

This installs all dependencies including Box2D via `gymnasium[box2d]`, which handles the correct package resolution and prebuilt wheels automatically.

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| torch | >= 2.0.0 |
| numpy | >= 1.24.0 |
| gymnasium[box2d] | >= 0.29.0 |
| pygame | >= 2.5.0 |
