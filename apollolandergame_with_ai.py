"""
Apollo Lunar Lander Game with AI Autopilot Toggle
A 3-screen scrolling lunar lander game with:
- AI autopilot mode using trained Double DQN agent
- Toggle between AI and manual control with 'A' key
- Procedurally generated terrain with multiple landing pads
- Fuel management (descent and ascent stages)
- Scoring system based on pad difficulty and fuel efficiency
- Realistic Apollo LM physics and controls
- Camera scrolling to follow the lander
"""

import json
import math
import os
import pygame
import random
import sys
import numpy as np
import torch
from collections import Counter
from Box2D import b2World, b2PolygonShape, b2Vec2
from apollo_terrain import (
    ApolloTerrain,
    is_point_on_pad,
    get_terrain_height_at,
    draw_terrain_pygame,
)
from apollolander import (
    ApolloLander,
    draw_body,
    draw_rcs_pods,
    draw_thrusters,
    world_to_screen,
    check_contact_probes,
    PPM,
    RCS_THRUST_FACTOR,
    RCS_OFFSET_X,
    RCS_OFFSET_Y,
    RCS_OFFSET_UP,
    MAX_GIMBAL_DEG,
    MASS_SCALE,
)
from apollo_hud import ApolloHUD
from apollocsm import ApolloCSM
from apollo_sky import ApolloSky, draw_sky_pygame
from satellite import Satellite

# Try to import AI agent
AI_AVAILABLE = False
try:
    from double_dqn_agent import DoubleDQNAgent
    AI_AVAILABLE = True
    print("[OK] AI agent loaded successfully")
except ImportError:
    print("[!] AI agent not found - manual mode only")


def find_model(seed, save_dir='models'):
    """Find best available model for a seed, preferring lateral RCS 7-action models."""
    candidates = [
        # Stage 3 precision landing models (7 actions) — preferred
        (os.path.join(save_dir, f'lateral_stage3_seed{seed}_best.pth'), 7),
        (os.path.join(save_dir, f'lateral_stage3_seed{seed}_final.pth'), 7),
        # Stage 2 approach models (7 actions) — fallback
        (os.path.join(save_dir, f'lateral_stage2_seed{seed}_best.pth'), 7),
        (os.path.join(save_dir, f'lateral_stage2_seed{seed}_final.pth'), 7),
        # Legacy 5-action models — fallback
        (os.path.join(save_dir, f'ingame_stage4_seed{seed}_best.pth'), 5),
        (os.path.join(save_dir, f'ingame_stage4_seed{seed}_final.pth'), 5),
        (os.path.join(save_dir, f'ingame_stage3_seed{seed}_best.pth'), 5),
        (os.path.join(save_dir, f'curriculum_stage4_seed{seed}_best.pth'), 5),
        (os.path.join(save_dir, f'curriculum_stage4_seed{seed}_final.pth'), 5),
        (os.path.join(save_dir, f'curriculum_stage3_seed{seed}_best.pth'), 5),
        (os.path.join(save_dir, f'apollo_ddqn_seed{seed}_best.pth'), 5),
        (os.path.join(save_dir, f'apollo_ddqn_seed{seed}_final.pth'), 5),
    ]
    for path, action_size in candidates:
        if os.path.exists(path):
            return path, action_size
    return None, 5

# -------------------------------------------------
# Config
# -------------------------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 1000
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS

# Terrain and spawn settings
TERRAIN_ROUGHNESS = 1.0  # Controls terrain bumpiness (higher = rougher)
SPAWN_ALTITUDE = 50.0    # meters in game world (= 50 km real altitude)

# Planet properties
PLANET_PROPERTIES = {
    "mercury":  {"gravity": 3.7,   "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 4879,  "orbit_alt": 130},
    "venus":    {"gravity": 8.87,  "linear_damping": 0.35, "angular_damping": 0.4,  "diameter": 12104, "orbit_alt": 300},
    "earth":    {"gravity": 9.81,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 12742, "orbit_alt": 350},
    "luna":     {"gravity": 1.62,  "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 3475,  "orbit_alt": 100},
    "mars":     {"gravity": 3.71,  "linear_damping": 0.05, "angular_damping": 0.15, "diameter": 6779,  "orbit_alt": 180},
    "jupiter":  {"gravity": 24.79, "linear_damping": 0.4,  "angular_damping": 0.5,  "diameter": 139820, "orbit_alt": 4200},
    "saturn":   {"gravity": 10.44, "linear_damping": 0.3,  "angular_damping": 0.4,  "diameter": 116460, "orbit_alt": 3500},
    "uranus":   {"gravity": 8.69,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 50724, "orbit_alt": 1500},
    "neptune":  {"gravity": 11.15, "linear_damping": 0.3,  "angular_damping": 0.35, "diameter": 49244, "orbit_alt": 1500},
}

# Parse command-line arguments
SELECTED_PLANET = "luna"
TERRAIN_DIFFICULTY = 1
ENSEMBLE_SEEDS = None  # None = single agent mode, list of odd N >= 3 = ensemble mode
AUTO_EPISODES = 0  # 0 = normal interactive mode, >0 = auto-restart for N episodes
CRASH_LOG_DIR = "crash_logs"
QTY_LOG_DIR = "qty_logs"
argv = sys.argv[1:]
i = 0
while i < len(argv):
    arg = argv[i]
    lower_arg = arg.lower()
    if lower_arg in PLANET_PROPERTIES:
        SELECTED_PLANET = lower_arg
    elif lower_arg.startswith("--difficulty="):
        try:
            TERRAIN_DIFFICULTY = int(lower_arg.split("=")[1])
            TERRAIN_DIFFICULTY = max(1, min(3, TERRAIN_DIFFICULTY))
        except ValueError:
            print(f"Invalid difficulty '{arg}'. Using default: 1")
    elif lower_arg == "--models":
        # Consume subsequent integer args as seed numbers (first = master/tie-breaker)
        seeds = []
        j = 1
        while i + j < len(argv):
            try:
                seeds.append(int(argv[i + j]))
                j += 1
            except ValueError:
                break
        if len(seeds) >= 3 and len(seeds) % 2 == 1:
            ENSEMBLE_SEEDS = seeds
            print(f"Ensemble mode: {len(seeds)} agents, seeds {seeds} (master: {seeds[0]})")
        else:
            print(f"Usage: --models SEED1 SEED2 SEED3 [SEED4 SEED5 ...] (odd number >= 3, first seed = master/tie-breaker)")
            if len(seeds) >= 2 and len(seeds) % 2 == 0:
                print(f"  Got {len(seeds)} seeds — ensemble requires an odd number for majority voting")
        i += len(seeds)  # skip the seed args
    elif lower_arg.startswith("--episodes="):
        try:
            AUTO_EPISODES = int(lower_arg.split("=")[1])
            print(f"Auto-run mode: {AUTO_EPISODES} episodes")
        except ValueError:
            print(f"Invalid episodes '{arg}'. Using interactive mode.")
    elif lower_arg in ("1", "2", "3") and arg == argv[-1]:
        TERRAIN_DIFFICULTY = int(lower_arg)
    else:
        print(f"Unknown argument '{arg}'. Available planets:")
        for planet in PLANET_PROPERTIES.keys():
            print(f"  - {planet}")
        print(f"Difficulty: --difficulty=1|2|3")
        print(f"Ensemble: --models SEED1 SEED2 SEED3 [SEED4 SEED5 ...]")
        print(f"Auto-run: --episodes=N")
        print(f"Using defaults: {SELECTED_PLANET}, difficulty {TERRAIN_DIFFICULTY}")
    i += 1

# Get planet properties
PLANET = PLANET_PROPERTIES[SELECTED_PLANET]
LUNAR_GRAVITY = -PLANET["gravity"]
LINEAR_DAMPING = PLANET["linear_damping"]
ANGULAR_DAMPING = PLANET["angular_damping"]

# Calculate thrust factors
DESCENT_THRUST_FACTOR = PLANET["gravity"] * 2.71
ASCENT_THRUST_FACTOR = PLANET["gravity"] * 2.0

print(f"Playing on {SELECTED_PLANET.upper()}")
print(f"   Gravity: {PLANET['gravity']} m/s²")
print(f"   Descent thrust: {DESCENT_THRUST_FACTOR:.2f}x mass")
print(f"   Ascent thrust: {ASCENT_THRUST_FACTOR:.2f}x mass")
print(f"   Atmosphere: {'None' if LINEAR_DAMPING == 0.0 else f'Linear damping {LINEAR_DAMPING}, Angular damping {ANGULAR_DAMPING}'}")

# Separate propellant tanks (scaled for gameplay)
# Real Apollo ratio: 3,164 kg fuel / 5,084 kg oxidizer = 38.4% / 61.6%
# Mixture ratio: 1.6:1 (oxidizer to fuel by mass)
PROPELLANT_SCALE = 100.0  # Scale factor to convert real kg to game units
MAX_DESCENT_FUEL_UNITS = 3164.0 / PROPELLANT_SCALE      # ~31.64 units Aerozine 50
MAX_DESCENT_OXIDIZER_UNITS = 5084.0 / PROPELLANT_SCALE  # ~50.84 units N2O4
MIXTURE_RATIO = 1.6  # Oxidizer:Fuel consumption ratio

# CG shift model (realistic Apollo behavior)
# The dry structure CG is offset from the thrust axis due to:
# - Asymmetric equipment placement
# - Manufacturing tolerances
# - The real LM started with ~1.5° gimbal offset to compensate
# As propellant depletes, CG shifts toward the dry structure's offset position
DRY_STRUCTURE_CG_OFFSET_X = 0.15  # meters (dry structure CG is right of thrust axis)
PROPELLANT_CG_OFFSET_X = 0.0      # Propellant tanks are symmetric around thrust axis

# Auto-gimbal configuration for CG compensation
AUTO_GIMBAL_GAIN = 2.0            # Proportional gain (reduced - realistic CG shift is subtler)
AUTO_GIMBAL_DEADBAND = 0.01       # Meters (ignore tiny offsets)

# Dry mass for CG calculation (scaled)
DESCENT_DRY_MASS_SCALED = 2034.0 / PROPELLANT_SCALE  # ~20.34 units

# Game world size
SCALE_FACTOR = 1000.0
PLANET_DIAMETER_METERS = PLANET["diameter"] * 1000.0 / SCALE_FACTOR
ORBITAL_ALTITUDE_METERS = PLANET["orbit_alt"] * 1000.0 / SCALE_FACTOR

SCREEN_WIDTH_METERS = SCREEN_WIDTH / PPM
WORLD_SCREENS = PLANET_DIAMETER_METERS / SCREEN_WIDTH_METERS
WORLD_WIDTH = SCREEN_WIDTH_METERS * max(3, int(WORLD_SCREENS))

# -------------------------------------------------
# Game State
# -------------------------------------------------
class GameState:
    def __init__(self):
        self.score = 0
        self.landed = False
        self.crashed = False
        self.docked = False
        self.game_over = False
        self.stage = "descent"  # "descent", "surface", "ascent", "orbit"
        self.contact_light = False  # Contact probe detected surface

        # AI autopilot state
        self.ai_enabled = False
        self.ai_agent = None
        self.ai_action_size = 5
        self.ai_ensemble = None  # List of (agent, action_size) tuples, or None
        self.ai_vote_agreement = ""  # "N/N" unanimous, "M/N" majority, "M/N*" master decides

        # AI angle tracking (continuous, matches training env)
        self.ai_prev_raw_angle = 0.0
        self.ai_cumulative_angle = 0.0

    def reset(self):
        self.score = 0
        self.landed = False
        self.crashed = False
        self.docked = False
        self.game_over = False
        self.stage = "descent"
        self.contact_light = False
        # Keep AI state when resetting
        self.ai_prev_raw_angle = 0.0
        self.ai_cumulative_angle = 0.0

def load_ai_agent(model_path=None, seed=999, save_dir='models'):
    """Load the trained AI agent (9-dim state, 5 or 7 action always-on engine model)."""
    if not AI_AVAILABLE:
        return None, 5

    action_size = 5
    if model_path is None:
        model_path, action_size = find_model(seed, save_dir)

    if model_path is None or not os.path.exists(model_path):
        print(f"[!] No AI model found (seed={seed}, dir={save_dir})")
        return None, 5

    try:
        agent = DoubleDQNAgent(state_size=9, action_size=action_size, seed=seed)
        agent.load(model_path)
        print(f"[OK] AI agent loaded from {model_path} ({action_size} actions)")
        return agent, action_size
    except Exception as e:
        print(f"[!] Failed to load AI agent: {e}")
        return None, 5

def get_ai_action(agent, state):
    """Get action from AI agent given current state."""
    if agent is None:
        return 0  # No operation

    try:
        # Convert state to numpy array if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # Get action from agent (epsilon=0 for pure exploitation)
        action = agent.act(state, eps=0.0)
        return action
    except Exception as e:
        print(f"[!] AI action error: {e}")
        return 0

def load_ensemble(seeds, save_dir='models'):
    """Load AI agents for ensemble voting. First seed is master/tie-breaker."""
    if not AI_AVAILABLE:
        return None
    agents = []
    for seed in seeds:
        model_path, action_size = find_model(seed, save_dir)
        if model_path is None or not os.path.exists(model_path):
            print(f"[!] No model found for seed {seed} — ensemble requires all {len(seeds)} models")
            return None
        try:
            agent = DoubleDQNAgent(state_size=9, action_size=action_size, seed=seed)
            agent.load(model_path)
            agents.append((agent, action_size))
            role = "MASTER" if len(agents) == 1 else f"agent {len(agents)}"
            print(f"[OK] Ensemble {role}: seed {seed} from {model_path} ({action_size} actions)")
        except Exception as e:
            print(f"[!] Failed to load seed {seed}: {e}")
            return None
    return agents


def get_ensemble_action(agents, state):
    """Get action via majority vote from N agents. Agent[0] is tie-breaker."""
    n = len(agents)
    actions = []
    for agent, _ in agents:
        try:
            s = state if isinstance(state, np.ndarray) else np.array(state, dtype=np.float32)
            actions.append(agent.act(s, eps=0.0))
        except Exception as e:
            print(f"[!] Ensemble agent error: {e}")
            actions.append(0)

    counts = Counter(actions)
    winner, top_count = counts.most_common(1)[0]
    majority = n // 2 + 1

    if top_count >= majority:
        agreement = f"{top_count}/{n}"
        return winner, agreement
    else:
        # No majority — master (agent[0]) decides
        return actions[0], f"{counts[actions[0]]}/{n}*"


def lander_state_to_gym_state(lander, target_x, target_y, descent_throttle, game_state):
    """
    Convert lander physics state to AI model's 9-dim state format.
    Matches apollo_lander_env._get_observation() exactly.

    State format (9 dimensions):
    0: x position relative to target (normalized by world_width/2)
    1: y position relative to target pad (normalized by 50m)
    2: x velocity (normalized by 10)
    3: y velocity (normalized by 10)
    4: angle (continuous tracking, normalized to [-pi, pi])
    5: angular velocity (negated to match convention: + = rotating right)
    6: left leg contact
    7: right leg contact
    8: throttle level (0.1 to 1.0)
    """
    pos = lander.descent_stage.position
    vel = lander.descent_stage.linearVelocity
    raw_angle = lander.descent_stage.angle
    angular_vel = lander.descent_stage.angularVelocity

    # Continuous angle tracking (prevents jumps at +/-pi)
    angle_delta = raw_angle - game_state.ai_prev_raw_angle
    if angle_delta > math.pi:
        angle_delta -= 2 * math.pi
    elif angle_delta < -math.pi:
        angle_delta += 2 * math.pi
    game_state.ai_cumulative_angle -= angle_delta  # Negate: + = tilted right
    game_state.ai_prev_raw_angle = raw_angle

    # Normalize angle to [-pi, pi] for neural network
    angle = game_state.ai_cumulative_angle
    normalized_angle = math.atan2(math.sin(angle), math.cos(angle))

    # Normalize positions (match training env)
    rel_x = np.clip((pos.x - target_x) / (WORLD_WIDTH / 2.0), -1.5, 1.5)
    rel_y = np.clip((pos.y - target_y) / 50.0, -1.5, 1.5)

    # Normalize velocities
    vel_x = np.clip(vel.x / 10.0, -1.5, 1.5)
    vel_y = np.clip(vel.y / 10.0, -1.5, 1.5)
    angular_vel = np.clip(angular_vel, -5.0, 5.0)

    state = np.array([
        rel_x,
        rel_y,
        vel_x,
        vel_y,
        normalized_angle,
        -angular_vel,   # Negate to match angle convention (+ = rotating right)
        0.0,            # left leg contact (no contact listener in game)
        0.0,            # right leg contact
        descent_throttle,
    ], dtype=np.float32)

    return state

def _log_episode(ep, total, game_state, lander, descent_fuel, max_descent_fuel,
                  target_pad, steps, is_point_on_pad_fn, qty_triggered=False):
    """Log a single episode result. Returns a dict for summary aggregation."""
    pos = lander.descent_stage.position
    vel = lander.descent_stage.linearVelocity
    angle_deg = math.degrees(lander.descent_stage.angle)
    speed = math.sqrt(vel.x**2 + vel.y**2)
    fuel_pct = (descent_fuel / max_descent_fuel) * 100 if max_descent_fuel > 0 else 0
    on_target = is_point_on_pad_fn(pos.x, pos.y, target_pad, tolerance_x=2.5, tolerance_y=3.0)
    pad_cx = (target_pad["x1"] + target_pad["x2"]) / 2.0
    dist_x = abs(pos.x - pad_cx)

    if game_state.crashed:
        status = "CRASHED"
    elif on_target:
        status = "ON-TARGET"
    else:
        status = "OFF-TARGET"

    print(f"  Ep {ep:4d}/{total} | {status:10s} | Speed: {speed:5.2f} m/s | "
          f"Angle: {angle_deg:+6.1f}° | Dist: {dist_x:5.1f}m | "
          f"Fuel: {fuel_pct:4.1f}% | Steps: {steps}")

    return {
        "status": status,
        "speed": speed,
        "angle": abs(angle_deg),
        "dist_x": dist_x,
        "fuel_pct": fuel_pct,
        "steps": steps,
        "qty_triggered": qty_triggered,
    }


def _print_summary(results, ensemble_seeds):
    """Print evaluation summary after all episodes complete."""
    total = len(results)
    landed = [r for r in results if r["status"] != "CRASHED"]
    on_target = [r for r in results if r["status"] == "ON-TARGET"]
    crashed = [r for r in results if r["status"] == "CRASHED"]
    qty_episodes = sum(1 for r in results if r.get("qty_triggered", False))

    print("\n" + "=" * 70)
    if ensemble_seeds:
        print(f"GAME EVALUATION — ENSEMBLE [{', '.join(str(s) for s in ensemble_seeds)}]")
    else:
        print("GAME EVALUATION — SINGLE AGENT")
    print("=" * 70)
    print(f"Episodes:      {total}")
    print(f"Landed:        {len(landed)} ({100.0 * len(landed) / total:.1f}%)")
    print(f"  On target:   {len(on_target)} ({100.0 * len(on_target) / total:.1f}%)")
    print(f"  Off target:  {len(landed) - len(on_target)}")
    print(f"Crashed:       {len(crashed)} ({100.0 * len(crashed) / total:.1f}%)")
    print(f"Qty light:     {qty_episodes} ({100.0 * qty_episodes / total:.1f}%)")

    if landed:
        speeds = [r["speed"] for r in landed]
        angles = [r["angle"] for r in landed]
        dists = [r["dist_x"] for r in landed]
        fuels = [r["fuel_pct"] for r in landed]
        steps = [r["steps"] for r in landed]
        soft = sum(1 for s in speeds if s < 1.0)
        gentle = sum(1 for s in speeds if 1.0 <= s < 2.0)
        hard = sum(1 for s in speeds if s >= 2.0)

        print(f"\n--- Landing Quality ---")
        print(f"Avg speed:     {np.mean(speeds):.2f} m/s  (max {np.max(speeds):.2f})")
        print(f"Avg angle:     {np.mean(angles):.1f}°  (max {np.max(angles):.1f}°)")
        print(f"Avg dist:      {np.mean(dists):.2f} m  (max {np.max(dists):.2f})")
        print(f"Avg fuel left: {np.mean(fuels):.1f}%")
        print(f"Avg steps:     {np.mean(steps):.0f}  (min {np.min(steps)}, max {np.max(steps)})")
        print(f"\nLanding breakdown:")
        print(f"  Soft (<1 m/s):     {soft} ({100.0 * soft / len(landed):.1f}%)")
        print(f"  Gentle (1-2 m/s):  {gentle} ({100.0 * gentle / len(landed):.1f}%)")
        print(f"  Hard (>2 m/s):     {hard} ({100.0 * hard / len(landed):.1f}%)")
    print("=" * 70)


def _dump_crash_log(episode_num, ensemble_seeds, initial_conditions, flight_data,
                    lander, terrain_pts, target_pad, descent_fuel, max_descent_fuel):
    """Save detailed crash diagnostics to a JSON file."""
    os.makedirs(CRASH_LOG_DIR, exist_ok=True)

    pos = lander.descent_stage.position
    vel = lander.descent_stage.linearVelocity
    angle_deg = math.degrees(lander.descent_stage.angle)
    speed = math.sqrt(vel.x**2 + vel.y**2)
    terrain_h = get_terrain_height_at(pos.x, terrain_pts)
    altitude = pos.y - terrain_h

    # Determine crash reason
    crash_reasons = []
    if abs(angle_deg) > 30:
        crash_reasons.append(f"angle={angle_deg:+.1f}° (threshold: ±30°)")
    if abs(vel.y) > 3.0 or abs(vel.x) > 3.0:
        crash_reasons.append(f"high_speed: vel_x={vel.x:.2f}, vel_y={vel.y:.2f} (threshold: ±3.0 m/s)")
    if 20 <= abs(angle_deg) <= 30 and not is_point_on_pad(pos.x, pos.y, target_pad, tolerance_x=2.5, tolerance_y=3.0):
        crash_reasons.append(f"off_target_bad_angle={angle_deg:+.1f}° (20-30° range, not on pad)")
    if not crash_reasons:
        crash_reasons.append("unknown (review thresholds)")

    # Terrain profile around pad (±50m)
    pad_cx = (target_pad["x1"] + target_pad["x2"]) / 2.0
    terrain_profile = []
    for pt_x, pt_y in terrain_pts:
        if pad_cx - 50 <= pt_x <= pad_cx + 50:
            terrain_profile.append({"x": round(pt_x, 2), "y": round(pt_y, 2)})

    final_state = {
        "pos_x": round(pos.x, 3),
        "pos_y": round(pos.y, 3),
        "vel_x": round(vel.x, 3),
        "vel_y": round(vel.y, 3),
        "speed": round(speed, 3),
        "angle_deg": round(angle_deg, 2),
        "angular_vel": round(lander.descent_stage.angularVelocity, 4),
        "altitude": round(altitude, 3),
        "terrain_height": round(terrain_h, 3),
        "fuel_pct": round((descent_fuel / max_descent_fuel) * 100, 1) if max_descent_fuel > 0 else 0,
        "crash_reasons": crash_reasons,
    }

    log_data = {
        "episode": episode_num,
        "ensemble_seeds": ensemble_seeds,
        "initial_conditions": initial_conditions,
        "terrain_profile_around_pad": terrain_profile,
        "target_pad": {
            "x1": target_pad["x1"],
            "x2": target_pad["x2"],
            "y": target_pad["y"],
            "width": abs(target_pad["x2"] - target_pad["x1"]),
            "center_x": pad_cx,
        },
        "total_steps": len(flight_data),
        "final_state": final_state,
        "flight_data": flight_data,
    }

    if ensemble_seeds:
        seeds_str = "_".join(str(s) for s in ensemble_seeds)
        filename = f"crash_ep{episode_num:04d}_ensemble_{seeds_str}.json"
    else:
        filename = f"crash_ep{episode_num:04d}_single.json"

    filepath = os.path.join(CRASH_LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  >> Crash log saved: {filepath}")


def _dump_qty_log(episode_num, ensemble_seeds, initial_conditions, flight_data,
                  lander, terrain_pts, target_pad,
                  fuel_units, oxidizer_units, max_fuel_units, max_oxid_units, step):
    """Save quantity-warning diagnostics to a JSON file."""
    os.makedirs(QTY_LOG_DIR, exist_ok=True)

    pos = lander.descent_stage.position
    vel = lander.descent_stage.linearVelocity
    angle_deg = math.degrees(lander.descent_stage.angle)
    terrain_h = get_terrain_height_at(pos.x, terrain_pts)
    altitude = pos.y - terrain_h
    pad_cx = (target_pad["x1"] + target_pad["x2"]) / 2.0

    fuel_pct  = round(fuel_units     / max_fuel_units  * 100, 1) if max_fuel_units  > 0 else 0
    oxid_pct  = round(oxidizer_units / max_oxid_units  * 100, 1) if max_oxid_units  > 0 else 0

    log_data = {
        "episode": episode_num,
        "ensemble_seeds": ensemble_seeds,
        "initial_conditions": initial_conditions,
        "target_pad": {
            "x1": target_pad["x1"],
            "x2": target_pad["x2"],
            "y": target_pad["y"],
            "center_x": round(pad_cx, 3),
        },
        "state_at_qty_trigger": {
            "step": step,
            "pos_x": round(pos.x, 3),
            "pos_y": round(pos.y, 3),
            "vel_x": round(vel.x, 3),
            "vel_y": round(vel.y, 3),
            "angle_deg": round(angle_deg, 2),
            "altitude": round(altitude, 3),
            "dist_to_pad": round(pos.x - pad_cx, 3),
            "fuel_units": round(fuel_units, 3),
            "fuel_pct": fuel_pct,
            "fuel_lbs": round(fuel_units * 220.462, 1),
            "oxidizer_units": round(oxidizer_units, 3),
            "oxidizer_pct": oxid_pct,
            "oxidizer_lbs": round(oxidizer_units * 220.462, 1),
        },
        "flight_data_to_trigger": flight_data,
    }

    if ensemble_seeds:
        seeds_str = "_".join(str(s) for s in ensemble_seeds)
        filename = f"qty_ep{episode_num:04d}_ensemble_{seeds_str}.json"
    else:
        filename = f"qty_ep{episode_num:04d}_single.json"

    filepath = os.path.join(QTY_LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  >> Qty warning log saved: {filepath}")


# -------------------------------------------------
# CG and Auto-Gimbal Functions
# -------------------------------------------------
def calculate_cg_offset(fuel_units, oxidizer_units, dry_mass_units):
    """
    Calculate lateral CG offset based on propellant depletion.

    Models realistic Apollo LM behavior where:
    - Dry structure CG is offset from thrust axis (manufacturing tolerance, equipment)
    - Propellant CG is centered (symmetric tank arrangement)
    - As propellant depletes, overall CG shifts toward dry structure's offset

    The real LM started with ~1.5° gimbal offset and adjusted throughout descent.

    Args:
        fuel_units: Current fuel amount (scaled units)
        oxidizer_units: Current oxidizer amount (scaled units)
        dry_mass_units: Dry mass (scaled units)

    Returns:
        cg_offset_x: Lateral offset of CG from thrust axis (meters)
    """
    propellant_mass = fuel_units + oxidizer_units
    total_mass = dry_mass_units + propellant_mass
    if total_mass <= 0:
        return 0.0

    # Weighted average of dry structure CG and propellant CG
    # CG = (dry_mass * dry_cg + propellant_mass * propellant_cg) / total_mass
    moment = (dry_mass_units * DRY_STRUCTURE_CG_OFFSET_X +
              propellant_mass * PROPELLANT_CG_OFFSET_X)
    return moment / total_mass


def calculate_auto_gimbal(cg_offset_x):
    """
    Calculate gimbal angle to keep thrust through CG.

    Args:
        cg_offset_x: Lateral CG offset (meters)

    Returns:
        auto_gimbal_deg: Recommended gimbal angle
    """
    if abs(cg_offset_x) < AUTO_GIMBAL_DEADBAND:
        return 0.0

    engine_to_cg_vertical = 2.0  # meters
    angle_rad = math.atan2(cg_offset_x, engine_to_cg_vertical)
    angle_deg = math.degrees(angle_rad) * AUTO_GIMBAL_GAIN

    return max(-MAX_GIMBAL_DEG, min(MAX_GIMBAL_DEG, angle_deg))


def compute_final_gimbal(auto_gimbal_deg, manual_gimbal_deg):
    """
    Combine auto-gimbal with manual input.

    Args:
        auto_gimbal_deg: CG compensation
        manual_gimbal_deg: Player input

    Returns:
        Combined gimbal angle (clamped)
    """
    combined = auto_gimbal_deg + manual_gimbal_deg
    return max(-MAX_GIMBAL_DEG, min(MAX_GIMBAL_DEG, combined))


# -------------------------------------------------
# Main Game
# -------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Apollo Lunar Lander - Atari Style [AI Toggle: A key]")
    clock = pygame.time.Clock()

    # Initialize game
    game_state = GameState()
    hud = ApolloHUD(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Try to load AI agent(s)
    if AI_AVAILABLE:
        if ENSEMBLE_SEEDS is not None:
            ensemble = load_ensemble(ENSEMBLE_SEEDS)
            if ensemble is not None:
                game_state.ai_ensemble = ensemble
                game_state.ai_agent = ensemble[0][0]  # master agent as fallback reference
                game_state.ai_action_size = max(a for _, a in ensemble)
                print("\n" + "="*60)
                print(f"ENSEMBLE AUTOPILOT READY (seeds: {ENSEMBLE_SEEDS})")
                print(f"Master/tie-breaker: seed {ENSEMBLE_SEEDS[0]}")
                print("="*60)
                print("Press 'A' to toggle between AI and Manual control")
                print("="*60 + "\n")
            else:
                print("[!] Ensemble loading failed, falling back to single agent")
                game_state.ai_agent, game_state.ai_action_size = load_ai_agent()
        else:
            game_state.ai_agent, game_state.ai_action_size = load_ai_agent()
        if game_state.ai_agent is not None and game_state.ai_ensemble is None:
            print("\n" + "="*60)
            print("AI AUTOPILOT READY")
            print("="*60)
            print("Press 'A' to toggle between AI and Manual control")
            print("="*60 + "\n")

    # Initialize variables
    world = None
    terrain_gen = None
    terrain_body = None
    terrain_pts = []
    pads_info = []
    lander = None
    target_pad = None
    target_pad_index = 0
    csm = None
    csm_altitude = 60.0
    csm_x = 0.0
    max_world_height = 136.4
    spawn_x = 0.0
    spawn_y = 0.0
    cam_x = 0.0
    cam_y = 0.0
    descent_fuel = 0.0  # Legacy total for compatibility
    descent_fuel_units = 0.0      # Aerozine 50 (scaled units)
    descent_oxidizer_units = 0.0  # N2O4 (scaled units)
    ascent_fuel = 0.0
    csm_fuel = 0.0
    descent_throttle = 0.0
    descent_gimbal_deg = 0.0      # Combined gimbal (auto + manual)
    auto_gimbal_deg = 0.0         # Auto CG compensation
    manual_gimbal_deg = 0.0       # Player manual input
    cg_offset_x = 0.0             # CG offset from center (meters)
    max_descent_fuel = 82.0       # Real Apollo scaled (legacy)
    sky = None  # Star field with parallax
    satellite = None  # Orbiting Sputnik-like satellite

    def new_game():
        """Initialize a new game session."""
        nonlocal world, terrain_gen, terrain_body, terrain_pts, pads_info
        nonlocal lander, target_pad, target_pad_index, cam_x, cam_y, csm, csm_altitude
        nonlocal descent_fuel, descent_fuel_units, descent_oxidizer_units
        nonlocal ascent_fuel, csm_fuel, descent_throttle
        nonlocal descent_gimbal_deg, auto_gimbal_deg, manual_gimbal_deg, cg_offset_x
        nonlocal csm_x, spawn_x, spawn_y, max_world_height, max_descent_fuel, sky, satellite

        game_state.reset()
        world = b2World(gravity=(0, LUNAR_GRAVITY))

        # Generate terrain
        terrain_gen = ApolloTerrain(world_width_meters=WORLD_WIDTH, difficulty=TERRAIN_DIFFICULTY, roughness=TERRAIN_ROUGHNESS)
        terrain_body, terrain_pts, pads_info = terrain_gen.generate_terrain(world)

        # Generate star field with parallax on inner cylinder (10% ratio = strong depth)
        sky = ApolloSky(WORLD_WIDTH, SPAWN_ALTITUDE, star_cylinder_ratio=0.1, num_stars=800)

        # Create orbiting satellite above lander start altitude, moving right
        sat_x = random.uniform(0, WORLD_WIDTH)
        sat_y = SPAWN_ALTITUDE * 1.3  # 65m - above lander's 50m start
        satellite = Satellite(world, position=(sat_x, sat_y), velocity=8.0, scale=3.0, world_width=WORLD_WIDTH)

        # Select target pad
        target_pad_index = random.randint(0, len(pads_info) - 1)
        target_pad = pads_info[target_pad_index]
        target_x = (target_pad["x1"] + target_pad["x2"]) / 2.0

        # Spawn lander at configured altitude
        # Game scale: 1 game meter = 1 real km
        # Offset is +/- 5 to 20 meters from center (never directly over pad)
        offset_magnitude = random.uniform(5.0, 20.0)
        offset_sign = random.choice([-1, 1])
        spawn_x = target_x + (offset_magnitude * offset_sign)
        spawn_y = SPAWN_ALTITUDE
        max_world_height = SPAWN_ALTITUDE * 1.5

        lander = ApolloLander(world, position=b2Vec2(spawn_x, spawn_y), scale=0.75, gravity=PLANET["gravity"])

        # Apply atmospheric damping to both stages (Luna has no atmosphere, so 0.0)
        lander.descent_stage.linearDamping = LINEAR_DAMPING
        lander.descent_stage.angularDamping = ANGULAR_DAMPING
        lander.ascent_stage.linearDamping = LINEAR_DAMPING
        lander.ascent_stage.angularDamping = ANGULAR_DAMPING

        # Give lander some initial velocity (descending with horizontal component)
        # Real Apollo had ~1.5 m/s horizontal, ~20-30 m/s vertical descent at powered descent start
        initial_velocity_x = random.uniform(-3.0, 3.0)  # Horizontal drift
        initial_velocity_y = random.uniform(-5.0, -2.0)  # Slight downward velocity
        lander.descent_stage.linearVelocity = b2Vec2(initial_velocity_x, initial_velocity_y)
        lander.ascent_stage.linearVelocity = b2Vec2(initial_velocity_x, initial_velocity_y)

        # Apply random initial rotation (+/- 90 degrees) to match training environment
        # This makes the game more challenging and consistent with AI training
        initial_angle = random.uniform(-math.pi/2, math.pi/2)
        lander.descent_stage.angle = initial_angle
        lander.ascent_stage.angle = initial_angle

        # Also add some initial angular velocity
        initial_angular_vel = random.uniform(-0.5, 0.5)
        lander.descent_stage.angularVelocity = initial_angular_vel
        lander.ascent_stage.angularVelocity = initial_angular_vel

        # Initialize AI angle tracking for continuous angle observation
        game_state.ai_prev_raw_angle = initial_angle
        game_state.ai_cumulative_angle = -initial_angle  # Negate: + = tilted right

        csm = None  # CSM disabled for now
        csm_fuel = 0.0

        # Terminal descent fuel budget (~3x real Apollo = represents fuel remaining at ~150m altitude)
        # Separate tanks for fuel (Aerozine 50) and oxidizer (N2O4)
        fuel_multiplier = 3.03  # ~250 units total, matching training env terminal descent budget
        descent_fuel_units = MAX_DESCENT_FUEL_UNITS * fuel_multiplier     # ~95.97 units
        descent_oxidizer_units = MAX_DESCENT_OXIDIZER_UNITS * fuel_multiplier  # ~154.05 units
        max_descent_fuel = (MAX_DESCENT_FUEL_UNITS + MAX_DESCENT_OXIDIZER_UNITS) * fuel_multiplier  # ~250.02 total
        descent_fuel = descent_fuel_units + descent_oxidizer_units  # Legacy total

        ascent_fuel = 800.0
        descent_throttle = 0.6  # Start with 60% throttle (should hover at ~1.63x gravity force)
        descent_gimbal_deg = 0.0
        auto_gimbal_deg = 0.0
        manual_gimbal_deg = 0.0
        cg_offset_x = 0.0

        cam_x = lander.ascent_stage.position.x
        cam_y = lander.ascent_stage.position.y

    # Start new game
    new_game()

    # Auto-run episode tracking
    episode_num = 0
    episode_step = 0
    episode_logged = False  # prevents double-logging same episode
    episode_results = []  # list of dicts for summary
    auto_restart_delay = 0  # frames to wait before auto-restart
    episode_flight_data = []  # per-step flight recorder (dumped on crash only)
    episode_initial_conditions = {}  # captured at episode start
    crash_log_count = 0       # number of crash logs saved
    qty_log_count = 0         # number of qty warning logs saved
    episode_qty_triggered = False  # whether qty_warning triggered this episode

    def _capture_initial_conditions():
        """Snapshot initial conditions for crash diagnostics."""
        return {
            "spawn_x": round(spawn_x, 3),
            "spawn_y": round(spawn_y, 3),
            "initial_vel_x": round(lander.descent_stage.linearVelocity.x, 3),
            "initial_vel_y": round(lander.descent_stage.linearVelocity.y, 3),
            "initial_angle_deg": round(math.degrees(lander.descent_stage.angle), 2),
            "initial_angular_vel": round(lander.descent_stage.angularVelocity, 4),
            "target_pad_x1": target_pad["x1"],
            "target_pad_x2": target_pad["x2"],
            "target_pad_y": target_pad["y"],
            "target_pad_index": target_pad_index,
            "terrain_difficulty": TERRAIN_DIFFICULTY,
        }

    # Auto-enable AI in auto-run mode
    if AUTO_EPISODES > 0 and game_state.ai_agent is not None:
        game_state.ai_enabled = True
        episode_num = 1
        episode_initial_conditions = _capture_initial_conditions()
        print(f"\n--- Auto-run: Episode {episode_num}/{AUTO_EPISODES} ---")

    # Main game loop
    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        episode_step += 1

        # Auto-run: handle episode end + auto-restart
        if AUTO_EPISODES > 0 and (game_state.crashed or game_state.landed):
            if not episode_logged:
                episode_logged = True
                # Dump crash diagnostics before logging
                if game_state.crashed:
                    _dump_crash_log(episode_num, ENSEMBLE_SEEDS, episode_initial_conditions,
                                    episode_flight_data, lander, terrain_pts, target_pad,
                                    descent_fuel, max_descent_fuel)
                    crash_log_count += 1
                result = _log_episode(episode_num, AUTO_EPISODES, game_state, lander,
                                      descent_fuel, max_descent_fuel, target_pad,
                                      episode_step, is_point_on_pad,
                                      qty_triggered=episode_qty_triggered)
                episode_results.append(result)
                auto_restart_delay = 30  # half-second pause to see result

            if auto_restart_delay > 0:
                auto_restart_delay -= 1
            else:
                episode_num += 1
                if episode_num > AUTO_EPISODES:
                    _print_summary(episode_results, ENSEMBLE_SEEDS)
                    if crash_log_count > 0:
                        print(f"\nCrash logs saved: {crash_log_count} files in {CRASH_LOG_DIR}/")
                    if qty_log_count > 0:
                        print(f"Qty warning logs: {qty_log_count} files in {QTY_LOG_DIR}/")
                    running = False
                    continue
                new_game()
                episode_step = 0
                episode_logged = False
                episode_flight_data = []
                episode_qty_triggered = False
                episode_initial_conditions = _capture_initial_conditions()
                game_state.ai_enabled = True
                print(f"\n--- Auto-run: Episode {episode_num}/{AUTO_EPISODES} ---")

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # ESC: Quit
                if event.key == pygame.K_ESCAPE:
                    running = False

                # R: Reset game
                elif event.key == pygame.K_r:
                    new_game()

                # A: Toggle AI autopilot
                elif event.key == pygame.K_a:
                    if game_state.ai_agent is not None:
                        game_state.ai_enabled = not game_state.ai_enabled
                        mode = "ENABLED" if game_state.ai_enabled else "DISABLED"
                        print(f"\n{'='*60}")
                        print(f"AI AUTOPILOT {mode}")
                        print(f"{'='*60}\n")
                        # Initialize angle tracking when AI is turned on mid-flight
                        if game_state.ai_enabled and lander:
                            raw_angle = lander.descent_stage.angle
                            game_state.ai_prev_raw_angle = raw_angle
                            game_state.ai_cumulative_angle = -raw_angle
                            # Enforce minimum throttle (AI trained with 10% min)
                            descent_throttle = max(0.1, descent_throttle)
                    else:
                        print("[!] AI agent not available")

                # Throttle control (5% per key press, min 10% to match AI)
                elif event.key == pygame.K_KP8 or event.key == pygame.K_UP:
                    descent_throttle = min(1.0, descent_throttle + 0.05)
                elif event.key == pygame.K_KP2 or event.key == pygame.K_DOWN:
                    descent_throttle = max(0.1, descent_throttle - 0.05)

                # Tab/Shift-Tab: Cycle target landing pad
                elif event.key == pygame.K_TAB:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        target_pad_index = (target_pad_index - 1) % len(pads_info)
                    else:
                        target_pad_index = (target_pad_index + 1) % len(pads_info)
                    target_pad = pads_info[target_pad_index]
                    pad_x = (target_pad["x1"] + target_pad["x2"]) / 2.0
                    pad_mult = target_pad.get("mult", 1)
                    print(f"Target pad: {target_pad_index + 1}/{len(pads_info)} | x={pad_x:.0f}m | {pad_mult}x multiplier")

                # Manual gimbal trim (1 degree per key press)
                # This adjusts manual_gimbal_deg; auto_gimbal_deg is computed automatically
                elif event.key == pygame.K_COMMA:
                    manual_gimbal_deg = max(-MAX_GIMBAL_DEG, manual_gimbal_deg - 1.0)
                elif event.key == pygame.K_PERIOD:
                    manual_gimbal_deg = min(MAX_GIMBAL_DEG, manual_gimbal_deg + 1.0)
                elif event.key == pygame.K_SLASH:
                    manual_gimbal_deg = 0.0  # Reset manual trim only

        # Get controls (manual or AI)
        keys = pygame.key.get_pressed()

        # Initialize control inputs
        main_thrust = False
        # Individual RCS thruster controls
        rcs_left_up = False
        rcs_left_down = False
        rcs_left_side = False
        rcs_right_up = False
        rcs_right_down = False
        rcs_right_side = False
        game_state._ai_translate = 0  # -1=LEFT, 0=none, 1=RIGHT

        if game_state.ai_enabled and game_state.ai_agent is not None:
            # AI CONTROL MODE (5 or 7 action always-on engine model)
            target_x = (target_pad["x1"] + target_pad["x2"]) / 2.0
            target_y = target_pad["y"]
            state = lander_state_to_gym_state(lander, target_x, target_y, descent_throttle, game_state)
            if game_state.ai_ensemble is not None:
                action, game_state.ai_vote_agreement = get_ensemble_action(game_state.ai_ensemble, state)
            else:
                action = get_ai_action(game_state.ai_agent, state)
                game_state.ai_vote_agreement = ""

            # Flight recorder: capture step data (only in auto-run mode)
            if AUTO_EPISODES > 0:
                _pos = lander.descent_stage.position
                _vel = lander.descent_stage.linearVelocity
                _th = get_terrain_height_at(_pos.x, terrain_pts)
                _pad_cx = (target_pad["x1"] + target_pad["x2"]) / 2.0
                episode_flight_data.append({
                    "step": episode_step,
                    "pos_x": round(_pos.x, 3),
                    "pos_y": round(_pos.y, 3),
                    "vel_x": round(_vel.x, 3),
                    "vel_y": round(_vel.y, 3),
                    "angle_deg": round(math.degrees(lander.descent_stage.angle), 2),
                    "angular_vel": round(lander.descent_stage.angularVelocity, 4),
                    "throttle": round(descent_throttle, 3),
                    "altitude": round(_pos.y - _th, 3),
                    "terrain_height": round(_th, 3),
                    "fuel_pct": round((descent_fuel / max_descent_fuel) * 100, 1) if max_descent_fuel > 0 else 0,
                    "action": int(action),
                    "vote": game_state.ai_vote_agreement,
                    "dist_to_pad": round(_pos.x - _pad_cx, 3),
                })

            # Quantity warning detection — log first trigger per episode
            if AUTO_EPISODES > 0 and not episode_qty_triggered:
                _fuel_r = descent_fuel_units / MAX_DESCENT_FUEL_UNITS if MAX_DESCENT_FUEL_UNITS > 0 else 0
                _oxid_r = descent_oxidizer_units / MAX_DESCENT_OXIDIZER_UNITS if MAX_DESCENT_OXIDIZER_UNITS > 0 else 0
                if _fuel_r <= 0.05 or _oxid_r <= 0.05:
                    episode_qty_triggered = True
                    qty_log_count += 1
                    _dump_qty_log(episode_num, ENSEMBLE_SEEDS, episode_initial_conditions,
                                  list(episode_flight_data), lander, terrain_pts, target_pad,
                                  descent_fuel_units, descent_oxidizer_units,
                                  MAX_DESCENT_FUEL_UNITS, MAX_DESCENT_OXIDIZER_UNITS, episode_step)

            # Engine is ALWAYS ON at current throttle (matches training env)
            main_thrust = True

            # Map action space to game controls:
            # 0: NOOP (engine continues at current throttle, no RCS)
            # 1: RCS_L - Rotate LEFT (coupled: left pod down + right pod up)
            # 2: THR_UP - Increase throttle by 5%
            # 3: THR_DOWN - Decrease throttle by 5%
            # 4: RCS_R - Rotate RIGHT (coupled: right pod down + left pod up)
            # 5: TRANSLATE_L - Both pods fire to push craft LEFT (7-action only)
            # 6: TRANSLATE_R - Both pods fire to push craft RIGHT (7-action only)
            if action == 1:
                # RCS_L: rotate left
                rcs_left_up = True     # flame UP at left pod → force DOWN at left
                rcs_right_down = True  # flame DOWN at right pod → force UP at right
            elif action == 2:
                # THR_UP: increase throttle
                descent_throttle = min(1.0, descent_throttle + 0.05)
            elif action == 3:
                # THR_DOWN: decrease throttle (min 10% like training env)
                descent_throttle = max(0.1, descent_throttle - 0.05)
            elif action == 4:
                # RCS_R: rotate right
                rcs_right_up = True    # flame UP at right pod → force DOWN at right
                rcs_left_down = True   # flame DOWN at left pod → force UP at left
            elif action == 5:
                # TRANSLATE_L: both pods fire LEFT force (same as training env)
                # Can't use rcs_left_side/rcs_right_side — game fires them inward,
                # which would cancel out. Instead apply forces directly below.
                game_state._ai_translate = -1  # LEFT
            elif action == 6:
                # TRANSLATE_R: both pods fire RIGHT force (same as training env)
                game_state._ai_translate = 1   # RIGHT
        else:
            # MANUAL CONTROL MODE — matches AI action space
            # Engine is ALWAYS ON at current throttle (same as AI)
            main_thrust = True

            # 7/Home: RCS_L — Rotate LEFT (coupled: left pod up + right pod down)
            if keys[pygame.K_KP7] or keys[pygame.K_HOME]:
                rcs_left_up = True
                rcs_right_down = True
            # 9/PgUp: RCS_R — Rotate RIGHT (coupled: right pod up + left pod down)
            if keys[pygame.K_KP9] or keys[pygame.K_PAGEUP]:
                rcs_right_up = True
                rcs_left_down = True
            # 4/Left: TRANSLATE_L — Fire right pod side thruster (pushes craft left)
            if keys[pygame.K_KP4] or keys[pygame.K_LEFT]:
                rcs_right_side = True
            # 6/Right: TRANSLATE_R — Fire left pod side thruster (pushes craft right)
            if keys[pygame.K_KP6] or keys[pygame.K_RIGHT]:
                rcs_left_side = True


        # Apply forces to lander
        if not game_state.crashed and not game_state.landed:
            lander_pos = lander.descent_stage.position
            lander_angle = lander.descent_stage.angle

            # Disable descent engine after contact light (like real Apollo procedure)
            if game_state.contact_light:
                main_thrust = False
                descent_throttle = 0.0

            # Main engine thrust - requires both fuel AND oxidizer
            if main_thrust and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                # Calculate CG offset from asymmetric propellant depletion
                cg_offset_x = calculate_cg_offset(descent_fuel_units, descent_oxidizer_units,
                                                   DESCENT_DRY_MASS_SCALED * 4.0)  # 4x scale

                # Auto-gimbal compensation for CG offset
                auto_gimbal_deg = calculate_auto_gimbal(cg_offset_x)

                # Combine auto-gimbal with manual input
                descent_gimbal_deg = compute_final_gimbal(auto_gimbal_deg, manual_gimbal_deg)

                # Calculate total lander mass (both stages welded together)
                total_mass = lander.descent_stage.mass + lander.ascent_stage.mass
                thrust_magnitude = DESCENT_THRUST_FACTOR * total_mass * descent_throttle
                gimbal_rad = math.radians(descent_gimbal_deg)
                total_angle = lander_angle + gimbal_rad

                # Thrust direction: engine fires DOWN from lander, pushing lander UP
                thrust_x = -thrust_magnitude * math.sin(total_angle)
                thrust_y = thrust_magnitude * math.cos(total_angle)

                # Apply thrust at the combined center of mass (adjusted for CG offset)
                descent_mass = lander.descent_stage.mass
                ascent_mass = lander.ascent_stage.mass
                descent_pos = lander.descent_stage.worldCenter
                ascent_pos = lander.ascent_stage.worldCenter

                combined_com_x = (descent_mass * descent_pos.x + ascent_mass * ascent_pos.x) / total_mass
                combined_com_y = (descent_mass * descent_pos.y + ascent_mass * ascent_pos.y) / total_mass
                combined_com = b2Vec2(combined_com_x, combined_com_y)

                # Apply force at the combined center of mass (no torque)
                lander.descent_stage.ApplyForce((thrust_x, thrust_y), combined_com, True)

                # Consume propellants at mixture ratio (1.6:1 oxidizer:fuel)
                # Total burn rate ~0.12 per frame at full throttle, split by ratio
                total_burn_rate = 0.12 * descent_throttle
                fuel_burn = total_burn_rate / (1.0 + MIXTURE_RATIO)     # ~0.046 per frame
                oxidizer_burn = fuel_burn * MIXTURE_RATIO               # ~0.074 per frame
                descent_fuel_units = max(0.0, descent_fuel_units - fuel_burn)
                descent_oxidizer_units = max(0.0, descent_oxidizer_units - oxidizer_burn)

                # Update legacy total for compatibility
                descent_fuel = descent_fuel_units + descent_oxidizer_units

            # RCS thrusters - Individual thruster control
            # Each pod has 3 thrusters: up, down, and sideways
            # Real Apollo RCS: 445 N per thruster

            # Get RCS pod positions from lander user data
            user_data = getattr(lander.ascent_stage, "userData", {}) or {}
            scale = user_data.get("scale", 1.0)
            rcs_center_offset_x = user_data.get("rcs_center_offset_x", 2.1 * scale)
            rcs_center_y = user_data.get("rcs_center_y", 0.0)

            # RCS thrust: 15x boosted from realistic Apollo 445 N per thruster
            # Both AI and manual use same force for consistent physics
            rcs_thrust_per_thruster = 15.0 * 445.0 / 100.0  # 66.75 force units (matches training env)

            # Pod positions
            right_pod_pos = b2Vec2(rcs_center_offset_x, rcs_center_y)
            left_pod_pos = b2Vec2(-rcs_center_offset_x, rcs_center_y)

            # Transform to world coordinates
            right_pod_world = lander.ascent_stage.GetWorldPoint(right_pod_pos)
            left_pod_world = lander.ascent_stage.GetWorldPoint(left_pod_pos)

            # Track how many thrusters are firing for fuel consumption
            thrusters_firing = 0

            # LEFT POD - UP THRUSTER (Numpad 7/Home)
            # Flame goes UP, reaction force pushes craft DOWN on left side -> rotates CLOCKWISE
            if rcs_left_up and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(0.0, -1.0)  # Reaction force DOWN (flame goes UP)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    left_pod_world,
                    True
                )
                thrusters_firing += 1

            # LEFT POD - DOWN THRUSTER (Numpad 1/End)
            # Flame goes DOWN, reaction force pushes craft UP on left side -> rotates COUNTER-CLOCKWISE
            if rcs_left_down and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(0.0, 1.0)  # Reaction force UP (flame goes DOWN)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    left_pod_world,
                    True
                )
                thrusters_firing += 1

            # LEFT POD - SIDE THRUSTER (Numpad 4/Left)
            # Flame goes LEFT (inward), reaction force pushes craft RIGHT -> translates RIGHT
            if rcs_left_side and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(1.0, 0.0)  # Reaction force RIGHT (flame goes LEFT/inward)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    left_pod_world,
                    True
                )
                thrusters_firing += 1

            # RIGHT POD - UP THRUSTER (Numpad 9/PgUp)
            # Flame goes UP, reaction force pushes craft DOWN on right side -> rotates COUNTER-CLOCKWISE
            if rcs_right_up and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(0.0, -1.0)  # Reaction force DOWN (flame goes UP)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    right_pod_world,
                    True
                )
                thrusters_firing += 1

            # RIGHT POD - DOWN THRUSTER (Numpad 3/PgDn)
            # Flame goes DOWN, reaction force pushes craft UP on right side -> rotates CLOCKWISE
            if rcs_right_down and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(0.0, 1.0)  # Reaction force UP (flame goes DOWN)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    right_pod_world,
                    True
                )
                thrusters_firing += 1

            # RIGHT POD - SIDE THRUSTER (Numpad 6/Right)
            # Flame goes RIGHT (inward), reaction force pushes craft LEFT -> translates LEFT
            if rcs_right_side and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                thrust_dir_local = b2Vec2(-1.0, 0.0)  # Reaction force LEFT (flame goes RIGHT/inward)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    right_pod_world,
                    True
                )
                thrusters_firing += 1

            # Fuel consumption based on number of thrusters firing
            # RCS uses small amount of propellant at mixture ratio
            if thrusters_firing > 0:
                rcs_total_burn = 0.01 * thrusters_firing
                rcs_fuel_burn = rcs_total_burn / (1.0 + MIXTURE_RATIO)
                rcs_oxidizer_burn = rcs_fuel_burn * MIXTURE_RATIO
                descent_fuel_units = max(0.0, descent_fuel_units - rcs_fuel_burn)
                descent_oxidizer_units = max(0.0, descent_oxidizer_units - rcs_oxidizer_burn)
                descent_fuel = descent_fuel_units + descent_oxidizer_units

            # AI lateral translation (actions 5/6) — apply both pod forces in same direction
            if game_state._ai_translate != 0 and descent_fuel_units > 0 and descent_oxidizer_units > 0:
                translate_dir = float(game_state._ai_translate)  # -1.0 or 1.0
                thrust_dir_local = b2Vec2(translate_dir, 0.0)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    left_pod_world, True
                )
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    right_pod_world, True
                )
                # Set side thruster flags for visualization
                rcs_left_side = True
                rcs_right_side = True
                # Consume propellants for translation
                translate_burn = 0.02
                translate_fuel_burn = translate_burn / (1.0 + MIXTURE_RATIO)
                translate_oxidizer_burn = translate_fuel_burn * MIXTURE_RATIO
                descent_fuel_units = max(0.0, descent_fuel_units - translate_fuel_burn)
                descent_oxidizer_units = max(0.0, descent_oxidizer_units - translate_oxidizer_burn)
                descent_fuel = descent_fuel_units + descent_oxidizer_units


        # Update physics
        world.Step(TIME_STEP, 10, 10)

        # Update satellite position and wrapping
        if satellite:
            satellite.update(TIME_STEP)

        # Check contact probes (lunar surface sensing probes)
        if lander and not game_state.crashed and not game_state.landed and not game_state.contact_light:
            if check_contact_probes(lander.descent_stage, lambda x: get_terrain_height_at(x, terrain_pts), WORLD_WIDTH):
                game_state.contact_light = True
                # Auto-cutoff descent engine on contact (like real Apollo)
                descent_throttle = 0.0

        # Check landing/crash conditions
        if lander and not game_state.crashed and not game_state.landed:
            pos_x = lander.descent_stage.position.x
            pos_y = lander.descent_stage.position.y
            terrain_height = get_terrain_height_at(pos_x, terrain_pts)
            altitude = pos_y - terrain_height

            vel = lander.descent_stage.linearVelocity
            vel_y = vel.y
            vel_x = vel.x
            angle_deg = math.degrees(lander.descent_stage.angle)

            # Check if on target pad (body center is on or near pad)
            on_target_pad = is_point_on_pad(pos_x, pos_y, target_pad, tolerance_x=2.5, tolerance_y=3.0)

            # Landing detection - check if lander is nearly stationary on ground
            # Body center within 3.5m of terrain (feet extended ~2.4m below body)
            is_on_ground = altitude < 3.5
            is_nearly_stopped = abs(vel_y) < 1.0 and abs(vel_x) < 1.0

            if is_on_ground and is_nearly_stopped:
                # Successful landing: on target pad AND reasonable angle (< 20°)
                if on_target_pad and abs(angle_deg) < 20:
                    game_state.landed = True
                # Crash: bad angle (> 30°)
                elif abs(angle_deg) > 30:
                    game_state.crashed = True
                # Wrong pad but gentle landing
                elif not on_target_pad and abs(angle_deg) < 20:
                    game_state.landed = True  # Landed but on wrong pad
                else:
                    game_state.crashed = True

            # High-speed impact detection (crash even if not stopped)
            elif is_on_ground and (abs(vel_y) > 3.0 or abs(vel_x) > 3.0):
                game_state.crashed = True

        # Update camera to follow lander
        if lander:
            cam_x = lander.ascent_stage.position.x
            cam_y = lander.ascent_stage.position.y

        # Rendering
        screen.fill((0, 0, 0))

        # Draw stars with parallax effect (inner cylinder creates depth)
        draw_sky_pygame(
            screen, sky.get_stars(), sky.star_cylinder_ratio, sky.star_cylinder_width,
            WORLD_WIDTH, terrain_pts, cam_x, cam_y, SCREEN_WIDTH, SCREEN_HEIGHT, PPM
        )

        # Draw terrain
        draw_terrain_pygame(screen, terrain_pts, pads_info, cam_x, PPM, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y, WORLD_WIDTH)

        # Draw orbiting satellite
        if satellite:
            satellite.draw(screen, cam_x, cam_y, SCREEN_WIDTH, SCREEN_HEIGHT, PPM, WORLD_WIDTH)

        # Draw lander (draw ascent first, then descent on top)
        if lander:
            # Define colors
            ascent_color = (200, 200, 200)  # Light gray for ascent
            descent_color = (180, 150, 100)  # Gold for descent

            # Draw ascent stage first
            draw_body(screen, lander.ascent_stage, ascent_color, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
            draw_rcs_pods(screen, lander.ascent_stage, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)

            # Draw descent stage on top
            draw_body(screen, lander.descent_stage, descent_color, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)

            # Draw thruster flames
            # Descent stage thrusters (only main engine, only if fuel remains)
            draw_thrusters(
                screen,
                lander.descent_stage,
                is_ascent=False,
                main_on=main_thrust and descent_fuel > 0,
                tl=False,
                bl=False,
                tr=False,
                br=False,
                sl=False,
                sr=False,
                gimbal_angle_deg=descent_gimbal_deg,
                cam_x=cam_x,
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                cam_y=cam_y,
                throttle=descent_throttle,
            )

            # Ascent stage thrusters (RCS) - Individual thruster visualization
            draw_thrusters(
                screen,
                lander.ascent_stage,
                is_ascent=True,
                main_on=False,  # Ascent engine not used during descent
                tl=rcs_left_up,      # Left pod up thruster (Numpad 7/Home)
                bl=rcs_left_down,    # Left pod down thruster (Numpad 1/End)
                tr=rcs_right_up,     # Right pod up thruster (Numpad 9/PgUp)
                br=rcs_right_down,   # Right pod down thruster (Numpad 3/PgDn)
                sl=rcs_left_side,    # Left pod sideways thruster (Numpad 4/Left)
                sr=rcs_right_side,   # Right pod sideways thruster (Numpad 6/Right)
                gimbal_angle_deg=0.0,
                cam_x=cam_x,
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                cam_y=cam_y,
            )

        # Draw HUD elements
        # Draw mini-map (top-right corner) showing terrain and landing pads
        # Cylindrical wrapping: lander is always centered, world wraps around
        minimap_width = 600
        minimap_height = 120
        minimap_x = SCREEN_WIDTH - minimap_width - 10
        minimap_y = 10
        minimap_margin = 5

        # Draw background
        minimap_rect = pygame.Rect(minimap_x, minimap_y, minimap_width, minimap_height)
        pygame.draw.rect(screen, (0, 0, 0, 180), minimap_rect)
        pygame.draw.rect(screen, (100, 100, 100), minimap_rect, 2)

        # Calculate scale factors for mini-map
        world_height = max_world_height
        map_inner_width = minimap_width - 2 * minimap_margin
        map_inner_height = minimap_height - 2 * minimap_margin
        scale_x = map_inner_width / WORLD_WIDTH
        scale_y = map_inner_height / world_height

        # Get lander position for centering the cylindrical view
        lander_world_x = lander.ascent_stage.position.x if lander else WORLD_WIDTH / 2.0

        def wrap_x_relative(world_x):
            """Get x position relative to lander, wrapped for cylindrical world."""
            # Calculate offset from lander (center of minimap)
            dx = world_x - lander_world_x
            # Wrap to nearest representation (-WORLD_WIDTH/2 to +WORLD_WIDTH/2)
            while dx > WORLD_WIDTH / 2.0:
                dx -= WORLD_WIDTH
            while dx < -WORLD_WIDTH / 2.0:
                dx += WORLD_WIDTH
            return dx

        def world_to_minimap(world_x, world_y):
            """Convert world coordinates to minimap screen coordinates (cylindrical)."""
            # Get x relative to lander (centered)
            rel_x = wrap_x_relative(world_x)
            # Map to minimap: lander at center
            mx = minimap_x + minimap_margin + map_inner_width / 2.0 + rel_x * scale_x
            my = minimap_y + minimap_height - minimap_margin - world_y * scale_y
            return int(mx), int(my)

        # Draw terrain on mini-map with cylindrical wrapping
        if len(terrain_pts) > 0:
            # Build terrain segments, breaking when they wrap around
            minimap_segments = []
            current_segment = []
            prev_rel_x = None

            for pt in terrain_pts:
                rel_x = wrap_x_relative(pt[0])
                mx, my = world_to_minimap(pt[0], pt[1])

                # Check if we crossed the wrap boundary (large jump in relative x)
                if prev_rel_x is not None and abs(rel_x - prev_rel_x) > WORLD_WIDTH / 4.0:
                    # Save current segment and start new one
                    if len(current_segment) > 1:
                        minimap_segments.append(current_segment)
                    current_segment = []

                current_segment.append((mx, my))
                prev_rel_x = rel_x

            # Don't forget the last segment
            if len(current_segment) > 1:
                minimap_segments.append(current_segment)

            # Draw all segments
            for segment in minimap_segments:
                pygame.draw.lines(screen, (150, 150, 150), False, segment, 1)

        # Draw landing pads
        for i, pad in enumerate(pads_info):
            pad_x1, pad_y = pad["x1"], pad["y"]
            pad_x2 = pad["x2"]

            # Check if pad is visible (within half world width of lander)
            pad_center_x = (pad_x1 + pad_x2) / 2.0
            rel_center = wrap_x_relative(pad_center_x)
            if abs(rel_center) > WORLD_WIDTH / 2.0:
                continue  # Pad is beyond visible range

            mp1 = world_to_minimap(pad_x1, pad_y)
            mp2 = world_to_minimap(pad_x2, pad_y)

            # Get pad difficulty color based on multiplier
            mult = pad["mult"]
            if mult == 1:
                pad_color = (100, 255, 100)  # Easy - Green
            elif mult == 2:
                pad_color = (255, 255, 100)  # Medium - Yellow
            else:
                pad_color = (255, 100, 100)  # Hard - Red

            # Highlight target pad
            if i == target_pad_index:
                thickness = 3
                # Draw pulsing indicator
                if not game_state.crashed and not game_state.landed:
                    pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5
                    center_minimap = world_to_minimap(pad_center_x, pad_y)
                    pulse_radius1 = int(3 + pulse * 3)
                    pulse_radius2 = int(5 + pulse * 3)
                    pygame.draw.circle(screen, pad_color, center_minimap, pulse_radius1, 1)
                    pygame.draw.circle(screen, pad_color, center_minimap, pulse_radius2, 1)
            else:
                thickness = 2

            # Draw pad line
            pygame.draw.line(screen, pad_color, mp1, mp2, thickness)
            pygame.draw.circle(screen, pad_color, mp1, 2)
            pygame.draw.circle(screen, pad_color, mp2, 2)

        # Draw lander on mini-map (always at center)
        if lander:
            lander_minimap_x = minimap_x + minimap_margin + map_inner_width // 2
            lander_minimap_y = minimap_y + minimap_height - minimap_margin - int(lander.ascent_stage.position.y * scale_y)
            if game_state.crashed:
                lander_color = (255, 0, 0)  # Red when crashed
            elif game_state.landed:
                lander_color = (100, 255, 100)  # Light green when landed
            else:
                lander_color = (0, 255, 0)  # Green when flying
            pygame.draw.circle(screen, lander_color, (lander_minimap_x, lander_minimap_y), 3)

        # Draw satellite on minimap (orange dot)
        if satellite:
            sat_mx, sat_my = world_to_minimap(satellite.body.position.x, satellite.body.position.y)
            sat_mx, sat_my = int(sat_mx), int(sat_my)
            # Only draw if within minimap bounds
            if minimap_x < sat_mx < minimap_x + minimap_width and minimap_y < sat_my < minimap_y + minimap_height:
                pygame.draw.circle(screen, (255, 150, 0), (sat_mx, sat_my), 2)

        # Label
        font_mini = pygame.font.SysFont("Arial", 10, bold=True)
        minimap_label = font_mini.render("MAP", True, (150, 150, 150))
        screen.blit(minimap_label, (minimap_x + 5, minimap_y + 3))

        # ============================================================
        # APOLLO LM INSTRUMENT PANEL — Gray metal plates with screws
        # ============================================================
        # Panel layout constants (tightly packed vertical stack, left side)
        PX = 5           # Panel left edge
        PW = 172         # Panel width (main column)
        INST_X = 10      # Instrument left margin within panels

        # Panel Y positions (bottom-up, 2px seam gaps between panels)
        P_CTRL_Y = 893                # Control panel (CG/Gimbal) — bottom
        P_CTRL_H = 102
        P_ENG_Y = P_CTRL_Y - 2 - 178 # Engine panel (Throttle + Propellant)
        P_ENG_H = 178
        P_FLT_Y = P_ENG_Y - 2 - 228  # Flight panel (Horz Vel + Alt/Rate)
        P_FLT_H = 228
        P_WARN_Y = P_FLT_Y - 2 - 90  # Warning lights panel (Contact + Quantity)
        P_WARN_H = 90

        # FDAI panel (to the right, overlaps with engine panel vertically)
        P_FDAI_X = PX + PW + 2       # Right of main column with 2px seam
        P_FDAI_W = 140
        P_FDAI_Y = P_ENG_Y
        P_FDAI_H = P_ENG_H  # Spans engine panel height (centered on propellant gauges)

        # --- Draw panel plates (backgrounds first, then instruments on top) ---

        # Warning lights panel
        hud.draw_panel_plate(screen, PX, P_WARN_Y, PW, P_WARN_H, [
            (PX + 8, P_WARN_Y + 8), (PX + PW - 8, P_WARN_Y + 8),
            (PX + 8, P_WARN_Y + P_WARN_H - 8), (PX + PW - 8, P_WARN_Y + P_WARN_H - 8),
        ])

        # Flight instruments panel
        hud.draw_panel_plate(screen, PX, P_FLT_Y, PW, P_FLT_H, [
            (PX + 8, P_FLT_Y + 8), (PX + PW - 8, P_FLT_Y + 8),
            (PX + 8, P_FLT_Y + P_FLT_H - 8), (PX + PW - 8, P_FLT_Y + P_FLT_H - 8),
            (PX + PW // 2, P_FLT_Y + 8), (PX + PW // 2, P_FLT_Y + P_FLT_H - 8),
        ])

        # Engine panel
        hud.draw_panel_plate(screen, PX, P_ENG_Y, PW, P_ENG_H, [
            (PX + 8, P_ENG_Y + 8), (PX + PW - 8, P_ENG_Y + 8),
            (PX + 8, P_ENG_Y + P_ENG_H - 8), (PX + PW - 8, P_ENG_Y + P_ENG_H - 8),
            (PX + PW // 2, P_ENG_Y + 8), (PX + PW // 2, P_ENG_Y + P_ENG_H - 8),
        ])

        # Control panel
        hud.draw_panel_plate(screen, PX, P_CTRL_Y, PW, P_CTRL_H, [
            (PX + 8, P_CTRL_Y + 8), (PX + PW - 8, P_CTRL_Y + 8),
            (PX + 8, P_CTRL_Y + P_CTRL_H - 8), (PX + PW - 8, P_CTRL_Y + P_CTRL_H - 8),
        ])

        # FDAI mounting plate (right of engine panel)
        hud.draw_panel_plate(screen, P_FDAI_X, P_FDAI_Y, P_FDAI_W, P_FDAI_H, [
            (P_FDAI_X + 8, P_FDAI_Y + 8), (P_FDAI_X + P_FDAI_W - 8, P_FDAI_Y + 8),
            (P_FDAI_X + 8, P_FDAI_Y + P_FDAI_H - 8),
            (P_FDAI_X + P_FDAI_W - 8, P_FDAI_Y + P_FDAI_H - 8),
            (P_FDAI_X + P_FDAI_W // 2, P_FDAI_Y + 8),
            (P_FDAI_X + P_FDAI_W // 2, P_FDAI_Y + P_FDAI_H - 8),
        ])

        # Blank panel below FDAI (fills space below attitude indicator)
        hud.draw_panel_plate(screen, P_FDAI_X, P_CTRL_Y, P_FDAI_W, P_CTRL_H, [
            (P_FDAI_X + 8, P_CTRL_Y + 8), (P_FDAI_X + P_FDAI_W - 8, P_CTRL_Y + 8),
            (P_FDAI_X + 8, P_CTRL_Y + P_CTRL_H - 8),
            (P_FDAI_X + P_FDAI_W - 8, P_CTRL_Y + P_CTRL_H - 8),
        ])

        # --- Panel seams (thin dark lines between adjacent panels) ---
        hud.draw_panel_seam(screen, PX, P_FLT_Y - 1, PX + PW, P_FLT_Y - 1)
        hud.draw_panel_seam(screen, PX, P_ENG_Y - 1, PX + PW + 2 + P_FDAI_W, P_ENG_Y - 1)
        hud.draw_panel_seam(screen, PX, P_CTRL_Y - 1, PX + PW + 2 + P_FDAI_W, P_CTRL_Y - 1)
        # Vertical seam between main column and right panels
        hud.draw_panel_seam(screen, PX + PW + 1, P_FDAI_Y, PX + PW + 1, P_CTRL_Y + P_CTRL_H)

        # ============================================================
        # INSTRUMENTS — drawn on top of panel plates
        # ============================================================

        # === WARNING LIGHTS (on warning panel) ===
        # Instrument positions within warning panel
        contact_x = INST_X
        contact_y = P_WARN_Y + 5
        qty_x = INST_X + 85
        qty_y = P_WARN_Y + 5
        light_size = 80
        light_radius = 22

        # --- LUNAR CONTACT light ---
        pygame.draw.rect(screen, (80, 85, 90),
                         (contact_x, contact_y, light_size, light_size), 0, 6)
        pygame.draw.rect(screen, (60, 65, 70),
                         (contact_x, contact_y, light_size, light_size), 2, 6)
        font_label = pygame.font.SysFont("arial", 9, bold=True)
        lbl = font_label.render("LUNAR CONTACT", True, (180, 185, 190))
        screen.blit(lbl, lbl.get_rect(center=(contact_x + light_size // 2, contact_y + 14)))
        lc_cx = contact_x + light_size // 2
        lc_cy = contact_y + light_size // 2 + 5
        pygame.draw.circle(screen, (30, 32, 35), (lc_cx, lc_cy), light_radius + 6)
        pygame.draw.circle(screen, (50, 52, 55), (lc_cx, lc_cy), light_radius + 4)
        pygame.draw.circle(screen, (70, 72, 75), (lc_cx, lc_cy), light_radius + 2)
        if game_state.contact_light:
            pygame.draw.circle(screen, (0, 180, 220), (lc_cx, lc_cy), light_radius)
            pygame.draw.circle(screen, (0, 220, 255), (lc_cx, lc_cy), light_radius - 4)
            pygame.draw.circle(screen, (100, 240, 255), (lc_cx, lc_cy), light_radius - 9)
            pygame.draw.circle(screen, (200, 250, 255), (lc_cx, lc_cy), light_radius - 14)
            pygame.draw.circle(screen, (255, 255, 255), (lc_cx - 6, lc_cy - 6), 4)
        else:
            pygame.draw.circle(screen, (20, 40, 50), (lc_cx, lc_cy), light_radius)
            pygame.draw.circle(screen, (25, 50, 60), (lc_cx, lc_cy), light_radius - 6)
            pygame.draw.circle(screen, (40, 60, 70), (lc_cx - 5, lc_cy - 5), 3)
        pygame.draw.circle(screen, (140, 145, 150), (lc_cx, lc_cy), light_radius + 1, 2)

        # --- QUANTITY warning light ---
        pygame.draw.rect(screen, (80, 85, 90),
                         (qty_x, qty_y, light_size, light_size), 0, 6)
        pygame.draw.rect(screen, (60, 65, 70),
                         (qty_x, qty_y, light_size, light_size), 2, 6)
        qlbl = font_label.render("QUANTITY", True, (180, 185, 190))
        screen.blit(qlbl, qlbl.get_rect(center=(qty_x + light_size // 2, qty_y + 14)))
        qc_cx = qty_x + light_size // 2
        qc_cy = qty_y + light_size // 2 + 5
        pygame.draw.circle(screen, (30, 32, 35), (qc_cx, qc_cy), light_radius + 6)
        pygame.draw.circle(screen, (50, 52, 55), (qc_cx, qc_cy), light_radius + 4)
        pygame.draw.circle(screen, (70, 72, 75), (qc_cx, qc_cy), light_radius + 2)
        fuel_ratio = descent_fuel_units / MAX_DESCENT_FUEL_UNITS if MAX_DESCENT_FUEL_UNITS > 0 else 0
        oxid_ratio = descent_oxidizer_units / MAX_DESCENT_OXIDIZER_UNITS if MAX_DESCENT_OXIDIZER_UNITS > 0 else 0
        qty_warning = fuel_ratio <= 0.05 or oxid_ratio <= 0.05
        qty_flash_on = qty_warning and (pygame.time.get_ticks() % 500 < 250)
        if qty_flash_on:
            pygame.draw.circle(screen, (180, 0, 0), (qc_cx, qc_cy), light_radius)
            pygame.draw.circle(screen, (220, 0, 0), (qc_cx, qc_cy), light_radius - 4)
            pygame.draw.circle(screen, (255, 60, 60), (qc_cx, qc_cy), light_radius - 9)
            pygame.draw.circle(screen, (255, 150, 150), (qc_cx, qc_cy), light_radius - 14)
            pygame.draw.circle(screen, (255, 255, 255), (qc_cx - 6, qc_cy - 6), 4)
        else:
            pygame.draw.circle(screen, (40, 15, 15), (qc_cx, qc_cy), light_radius)
            pygame.draw.circle(screen, (50, 20, 20), (qc_cx, qc_cy), light_radius - 6)
            pygame.draw.circle(screen, (60, 25, 25), (qc_cx - 5, qc_cy - 5), 3)
        pygame.draw.circle(screen, (140, 145, 150), (qc_cx, qc_cy), light_radius + 1, 2)

        # === FLIGHT INSTRUMENTS (on flight panel) ===
        if lander:
            # Alt/Rate gauge (top of flight panel)
            alt = lander.ascent_stage.position.y
            vel_y = lander.ascent_stage.linearVelocity.y
            hud.draw_range_rate_gauge(screen, INST_X, P_FLT_Y + 5, alt, vel_y)

            # Horizontal velocity gauge (below alt/rate)
            vel_x = lander.ascent_stage.linearVelocity.x
            hud.draw_horizontal_velocity_gauge(screen, INST_X, P_FLT_Y + 178, vel_x)

        # === ENGINE INSTRUMENTS (on engine panel) ===
        if lander:
            # Throttle gauge (left side)
            hud.draw_throttle_gauge(screen, INST_X, P_ENG_Y + 4, descent_throttle)

            # Propellant gauges (right of throttle)
            fuel_mult = 3.03
            max_fuel_display = MAX_DESCENT_FUEL_UNITS * fuel_mult
            max_oxidizer_display = MAX_DESCENT_OXIDIZER_UNITS * fuel_mult
            hud.draw_propellant_gauges(
                screen, INST_X + 55, P_ENG_Y + 4,
                descent_fuel_units, max_fuel_display,
                descent_oxidizer_units, max_oxidizer_display,
            )

        # === CONTROL INSTRUMENTS (on control panel) ===
        if lander:
            hud.draw_cg_gimbal_indicator(screen, cg_offset_x, descent_gimbal_deg,
                                          auto_gimbal_deg, manual_gimbal_deg,
                                          x=INST_X, y=P_CTRL_Y + 5,
                                          width=160, height=92)

        # Draw FDAI-style attitude indicator (on FDAI mounting plate)
        font_small = pygame.font.SysFont("consolas", 12)
        if lander:
            fdai_center_x = P_FDAI_X + P_FDAI_W // 2
            fdai_center_y = P_FDAI_Y + P_FDAI_H // 2
            fdai_radius = 55

            # Get current angle and angular velocity
            current_angle_deg = math.degrees(lander.descent_stage.angle)
            current_angle_rad = lander.descent_stage.angle
            angular_vel = lander.descent_stage.angularVelocity  # rad/s
            angular_vel_deg = math.degrees(angular_vel)  # deg/s

            # === OCTAGONAL HOUSING ===
            # Draw octagonal frame (like real FDAI)
            oct_radius = fdai_radius + 8
            octagon_points = []
            for i in range(8):
                angle = math.pi / 8 + i * math.pi / 4  # Start rotated 22.5 degrees
                ox = fdai_center_x + oct_radius * math.cos(angle)
                oy = fdai_center_y + oct_radius * math.sin(angle)
                octagon_points.append((ox, oy))
            pygame.draw.polygon(screen, (60, 60, 70), octagon_points, 0)  # Fill
            pygame.draw.polygon(screen, (120, 120, 130), octagon_points, 2)  # Border

            # === ATTITUDE BALL (clipped to circle) ===
            # Create a surface for the ball that we'll clip
            ball_surface = pygame.Surface((fdai_radius * 2, fdai_radius * 2), pygame.SRCALPHA)
            ball_center = (fdai_radius, fdai_radius)

            # Sky color (top half when level) and ground color (bottom half when level)
            SKY_COLOR = (50, 80, 120)  # Dark blue sky
            GROUND_COLOR = (100, 70, 50)  # Brown ground

            # Draw rotating ball background
            # Ball rotates with lander angle so horizon tilts opposite from pilot's perspective
            ball_angle = current_angle_rad

            # Draw sky/ground split (rotating horizon)
            # Create a larger rect and rotate it
            horizon_offset = 0  # Could add pitch offset here for vertical movement
            cos_a = math.cos(ball_angle)
            sin_a = math.sin(ball_angle)

            # Draw ground half (below horizon)
            ground_points = []
            for i in range(180):
                a = math.radians(i) + ball_angle
                px = ball_center[0] + fdai_radius * math.cos(a)
                py = ball_center[1] + fdai_radius * math.sin(a)
                ground_points.append((px, py))
            if len(ground_points) >= 3:
                pygame.draw.polygon(ball_surface, GROUND_COLOR, ground_points)

            # Draw sky half (above horizon)
            sky_points = []
            for i in range(180):
                a = math.radians(i) + ball_angle + math.pi
                px = ball_center[0] + fdai_radius * math.cos(a)
                py = ball_center[1] + fdai_radius * math.sin(a)
                sky_points.append((px, py))
            if len(sky_points) >= 3:
                pygame.draw.polygon(ball_surface, SKY_COLOR, sky_points)

            # Draw horizon line on ball
            horizon_len = fdai_radius - 5
            h_x1 = ball_center[0] - horizon_len * cos_a
            h_y1 = ball_center[1] - horizon_len * sin_a
            h_x2 = ball_center[0] + horizon_len * cos_a
            h_y2 = ball_center[1] + horizon_len * sin_a
            pygame.draw.line(ball_surface, (255, 255, 255), (h_x1, h_y1), (h_x2, h_y2), 2)

            # Draw pitch ladder marks on the ball (rotate with horizon)
            for pitch_offset in [-30, -20, -10, 10, 20, 30]:
                # Calculate perpendicular offset from horizon
                perp_x = -sin_a * pitch_offset * 0.8  # Scale factor for visual
                perp_y = cos_a * pitch_offset * 0.8
                # Short horizontal marks
                mark_half_len = 8 if abs(pitch_offset) == 30 else 12
                m_cx = ball_center[0] + perp_x
                m_cy = ball_center[1] + perp_y
                m_x1 = m_cx - mark_half_len * cos_a
                m_y1 = m_cy - mark_half_len * sin_a
                m_x2 = m_cx + mark_half_len * cos_a
                m_y2 = m_cy + mark_half_len * sin_a
                # Check if within ball radius
                dist = math.sqrt(perp_x**2 + perp_y**2)
                if dist < fdai_radius - 10:
                    color = (200, 200, 200) if pitch_offset > 0 else (150, 150, 150)
                    pygame.draw.line(ball_surface, color, (m_x1, m_y1), (m_x2, m_y2), 1)

            # Draw circular border over ball to create clean edge (much faster than pixel masking)
            # First blit the ball surface
            screen.blit(ball_surface, (fdai_center_x - fdai_radius, fdai_center_y - fdai_radius))

            # Then draw the octagon fill again around the ball edge to mask overflow
            # Draw thick circle border to clean up edges
            pygame.draw.circle(screen, (60, 60, 70), (fdai_center_x, fdai_center_y), fdai_radius + 3, 6)

            # === FIXED SPACECRAFT SYMBOL (orange W-shape like real FDAI) ===
            symbol_color = (255, 165, 0)  # Orange
            # Center dot
            pygame.draw.circle(screen, symbol_color, (fdai_center_x, fdai_center_y), 3, 0)
            # Wings (horizontal lines)
            wing_len = 20
            pygame.draw.line(screen, symbol_color,
                           (fdai_center_x - wing_len, fdai_center_y),
                           (fdai_center_x - 8, fdai_center_y), 2)
            pygame.draw.line(screen, symbol_color,
                           (fdai_center_x + 8, fdai_center_y),
                           (fdai_center_x + wing_len, fdai_center_y), 2)
            # Down ticks at wing ends
            pygame.draw.line(screen, symbol_color,
                           (fdai_center_x - wing_len, fdai_center_y),
                           (fdai_center_x - wing_len, fdai_center_y + 6), 2)
            pygame.draw.line(screen, symbol_color,
                           (fdai_center_x + wing_len, fdai_center_y),
                           (fdai_center_x + wing_len, fdai_center_y + 6), 2)

            # === RATE INDICATOR (bottom - like YAW RATE on Apollo FDAI) ===
            rate_bar_y = fdai_center_y + fdai_radius + 12
            rate_bar_width = 80
            rate_bar_height = 8

            # Background
            pygame.draw.rect(screen, (40, 40, 50),
                           (fdai_center_x - rate_bar_width // 2, rate_bar_y,
                            rate_bar_width, rate_bar_height))

            # Center mark
            pygame.draw.line(screen, (150, 150, 150),
                           (fdai_center_x, rate_bar_y - 2),
                           (fdai_center_x, rate_bar_y + rate_bar_height + 2), 1)

            # Rate needle (angular velocity indicator)
            # Scale: ±20 deg/s maps to full width
            max_rate = 20.0  # deg/s for full deflection
            rate_normalized = max(-1, min(1, angular_vel_deg / max_rate))
            needle_x = fdai_center_x + int(rate_normalized * (rate_bar_width // 2 - 4))

            # Color based on rate
            if abs(angular_vel_deg) < 5:
                rate_color = (100, 255, 100)  # Green - stable
            elif abs(angular_vel_deg) < 15:
                rate_color = (255, 255, 0)    # Yellow - moderate
            else:
                rate_color = (255, 100, 100)  # Red - high rate

            pygame.draw.rect(screen, rate_color,
                           (needle_x - 2, rate_bar_y, 4, rate_bar_height))

            # Rate label
            rate_label = font_small.render("RATE", True, (180, 180, 180))
            screen.blit(rate_label, (fdai_center_x - 14, rate_bar_y + rate_bar_height + 2))

            # === ANGLE READOUT ===
            font_att = pygame.font.SysFont("consolas", 12, bold=True)

            # Color based on angle (for situational awareness)
            if abs(current_angle_deg) < 15:
                angle_color = (100, 255, 100)  # Green
            elif abs(current_angle_deg) < 30:
                angle_color = (255, 255, 0)    # Yellow
            elif abs(current_angle_deg) < 45:
                angle_color = (255, 165, 0)    # Orange
            else:
                angle_color = (255, 100, 100)  # Red

            # Angle display (top)
            angle_text = font_att.render(f"{current_angle_deg:+.1f}\u00b0", True, angle_color)
            angle_rect = angle_text.get_rect(center=(fdai_center_x, fdai_center_y - fdai_radius - 12))
            screen.blit(angle_text, angle_rect)

            # Rate value (bottom right of rate bar)
            rate_text = font_small.render(f"{angular_vel_deg:+.1f}\u00b0/s", True, rate_color)
            screen.blit(rate_text, (fdai_center_x + rate_bar_width // 2 + 5, rate_bar_y - 2))

        # === ZIGZAG PERIMETER TRIM (castellated edges around panel assembly) ===
        PANEL_RIGHT = PX + PW + 2 + P_FDAI_W  # Right edge of full assembly
        # Top of panel assembly
        hud.draw_panel_zigzag(screen, 'top', PX, P_WARN_Y, PW)
        # Right edge of narrow section (warning + flight panels)
        hud.draw_panel_zigzag(screen, 'right', PX + PW, P_WARN_Y,
                              P_ENG_Y - P_WARN_Y)
        # Step top (where panels widen for FDAI column)
        hud.draw_panel_zigzag(screen, 'top', PX + PW, P_ENG_Y, 2 + P_FDAI_W)
        # Right edge of wide section (FDAI + blank panels)
        hud.draw_panel_zigzag(screen, 'right', PANEL_RIGHT, P_ENG_Y,
                              P_CTRL_Y + P_CTRL_H - P_ENG_Y)

        # Draw AI status indicator
        if game_state.ai_enabled:
            font = pygame.font.Font(None, 36)
            if game_state.ai_ensemble is not None:
                seeds_str = ", ".join(str(s) for s in ENSEMBLE_SEEDS)
                ai_text = font.render(f"ENSEMBLE [{seeds_str}] [A to disable]", True, (0, 255, 0))
                screen.blit(ai_text, (SCREEN_WIDTH // 2 - 200, 10))
                # Vote agreement indicator
                font_ai_sub = pygame.font.Font(None, 24)
                vote = game_state.ai_vote_agreement
                n_agents = len(ENSEMBLE_SEEDS)
                if vote == f"{n_agents}/{n_agents}":
                    vote_color = (0, 255, 0)  # green = unanimous
                elif vote.endswith("*"):
                    vote_color = (255, 100, 100)  # red = no majority (master decides)
                else:
                    vote_color = (255, 255, 0)  # yellow = majority
                vote_text = font_ai_sub.render(f"Vote: {vote} | Master: seed {ENSEMBLE_SEEDS[0]}", True, vote_color)
                screen.blit(vote_text, (SCREEN_WIDTH // 2 - 150, 38))
            else:
                ai_text = font.render("AI AUTOPILOT [A to disable]", True, (0, 255, 0))
                screen.blit(ai_text, (SCREEN_WIDTH // 2 - 150, 10))
                # Show engine is always-on
                font_ai_sub = pygame.font.Font(None, 24)
                n_act = game_state.ai_action_size
                rcs_label = "RCS: rotation + lateral" if n_act >= 7 else "RCS: rotation only"
                engine_text = font_ai_sub.render(f"Engine: ALWAYS ON | {rcs_label} ({n_act} actions)", True, (0, 200, 200))
                screen.blit(engine_text, (SCREEN_WIDTH // 2 - 180, 38))
        elif AI_AVAILABLE and game_state.ai_agent is not None:
            font = pygame.font.Font(None, 24)
            label = "Press A for ENSEMBLE autopilot" if game_state.ai_ensemble else "Press A for AI autopilot"
            ai_text = font.render(label, True, (150, 150, 150))
            screen.blit(ai_text, (SCREEN_WIDTH // 2 - 130, 10))

        # Draw game over / success indicator
        if game_state.crashed or game_state.landed:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))

            font_big = pygame.font.SysFont("Arial", 72, bold=True)
            font_sub = pygame.font.SysFont("Arial", 36)
            font_info = pygame.font.SysFont("consolas", 24)

            if game_state.crashed:
                # Crash message
                title_text = font_big.render("CRASH!", True, (255, 50, 50))
                title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
                screen.blit(title_text, title_rect)

                sub_text = font_sub.render("Mission Failed", True, (255, 150, 150))
                sub_rect = sub_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
                screen.blit(sub_text, sub_rect)
            else:
                # Success message
                title_text = font_big.render("LANDED!", True, (50, 255, 50))
                title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
                screen.blit(title_text, title_rect)

                # Check if on target pad
                if lander:
                    pos_x = lander.descent_stage.position.x
                    pos_y = lander.descent_stage.position.y
                    on_target = is_point_on_pad(pos_x, pos_y, target_pad, tolerance_x=2.5, tolerance_y=3.0)
                    if on_target:
                        sub_text = font_sub.render("Perfect Landing on Target!", True, (150, 255, 150))
                    else:
                        sub_text = font_sub.render("Landed (Wrong Pad)", True, (255, 255, 150))
                    sub_rect = sub_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
                    screen.blit(sub_text, sub_rect)

            # Show final stats
            if lander:
                final_vel = lander.descent_stage.linearVelocity
                final_angle = math.degrees(lander.descent_stage.angle)
                stats_y = SCREEN_HEIGHT // 2 + 40

                vel_text = font_info.render(f"Final Velocity: {abs(final_vel.x):.1f} m/s H, {abs(final_vel.y):.1f} m/s V", True, (200, 200, 200))
                vel_rect = vel_text.get_rect(center=(SCREEN_WIDTH // 2, stats_y))
                screen.blit(vel_text, vel_rect)

                angle_text = font_info.render(f"Final Angle: {final_angle:+.1f}", True, (200, 200, 200))
                angle_rect = angle_text.get_rect(center=(SCREEN_WIDTH // 2, stats_y + 30))
                screen.blit(angle_text, angle_rect)

                fuel_pct = (descent_fuel / max_descent_fuel) * 100 if max_descent_fuel > 0 else 0
                fuel_text = font_info.render(f"Fuel Remaining: {fuel_pct:.0f}%", True, (200, 200, 200))
                fuel_rect = fuel_text.get_rect(center=(SCREEN_WIDTH // 2, stats_y + 60))
                screen.blit(fuel_text, fuel_rect)

            # Restart prompt
            restart_text = font_sub.render("Press R to Restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 140))
            screen.blit(restart_text, restart_rect)

        # Draw controls legend (lower right)
        font_legend = pygame.font.SysFont("consolas", 12)
        if game_state.ai_enabled:
            n_act = game_state.ai_action_size
            if n_act >= 7:
                legend_lines = [
                    "AI AUTOPILOT ACTIVE",
                    "-------------------",
                    "Engine: Always on",
                    "AI controls: RCS +",
                    "  throttle + lateral",
                    f"  ({n_act} actions)",
                    "",
                    "A        Manual mode",
                    "R        Reset",
                    "ESC      Quit",
                ]
            else:
                legend_lines = [
                    "AI AUTOPILOT ACTIVE",
                    "-------------------",
                    "Engine: Always on",
                    "AI controls: RCS +",
                    f"  throttle ({n_act} actions)",
                    "",
                    "A        Manual mode",
                    "R        Reset",
                    "ESC      Quit",
                ]
        else:
            legend_lines = [
                "MANUAL CONTROLS",
                "---------------",
                "SPACE    Main Engine",
                "UP/DOWN  Throttle +/-5%",
                ", . /    Gimbal L/R/Center",
                "",
                "Q/E      Translate L/R",
                "LEFT/RIGHT  Side RCS",
                "Num 7/9  RCS Up L/R",
                "Num 1/3  RCS Down L/R",
                "",
                "A        Toggle AI",
                "R        Reset",
                "ESC      Quit",
            ]
        legend_x = SCREEN_WIDTH - 180
        legend_y = SCREEN_HEIGHT - len(legend_lines) * 14 - 10
        for i, line in enumerate(legend_lines):
            color = (150, 150, 150) if i > 0 else (200, 200, 200)
            text = font_legend.render(line, True, color)
            screen.blit(text, (legend_x, legend_y + i * 14))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
