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
ENSEMBLE_SEEDS = None  # None = single agent mode, list of 3 = ensemble mode
AUTO_EPISODES = 0  # 0 = normal interactive mode, >0 = auto-restart for N episodes
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
        # Consume next 3 args as seed numbers (first = master/tie-breaker)
        seeds = []
        for j in range(1, 4):
            if i + j < len(argv):
                try:
                    seeds.append(int(argv[i + j]))
                except ValueError:
                    print(f"Invalid seed '{argv[i + j]}'. --models requires 3 integer seeds.")
                    seeds = []
                    break
        if len(seeds) == 3:
            ENSEMBLE_SEEDS = seeds
            print(f"Ensemble mode: seeds {seeds} (master: {seeds[0]})")
        else:
            print("Usage: --models SEED1 SEED2 SEED3 (first seed = master/tie-breaker)")
        i += 3  # skip the 3 seed args
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
        print(f"Ensemble: --models SEED1 SEED2 SEED3")
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

        # AI autopilot state
        self.ai_enabled = False
        self.ai_agent = None
        self.ai_action_size = 5
        self.ai_ensemble = None  # List of (agent, action_size) tuples, or None
        self.ai_vote_agreement = ""  # "3/3", "2/3", "1/3*"

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
    """Load 3 AI agents for ensemble voting. First seed is master/tie-breaker."""
    if not AI_AVAILABLE:
        return None
    agents = []
    for seed in seeds:
        model_path, action_size = find_model(seed, save_dir)
        if model_path is None or not os.path.exists(model_path):
            print(f"[!] No model found for seed {seed} — ensemble requires all 3 models")
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
    """Get action via majority vote from 3 agents. Agent[0] is tie-breaker."""
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

    if top_count >= 2:
        # Majority (2 or 3 agree)
        agreement = "3/3" if top_count == 3 else "2/3"
        return winner, agreement
    else:
        # All 3 different — master (agent[0]) decides
        return actions[0], "1/3*"


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
                  target_pad, steps, is_point_on_pad_fn):
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
    }


def _print_summary(results, ensemble_seeds):
    """Print evaluation summary after all episodes complete."""
    total = len(results)
    landed = [r for r in results if r["status"] != "CRASHED"]
    on_target = [r for r in results if r["status"] == "ON-TARGET"]
    crashed = [r for r in results if r["status"] == "CRASHED"]

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
    descent_fuel = 0.0
    ascent_fuel = 0.0
    csm_fuel = 0.0
    descent_throttle = 0.0
    descent_gimbal_deg = 0.0
    max_descent_fuel = 82.0  # Real Apollo scaled

    def new_game():
        """Initialize a new game session."""
        nonlocal world, terrain_gen, terrain_body, terrain_pts, pads_info
        nonlocal lander, target_pad, target_pad_index, cam_x, cam_y, csm, csm_altitude
        nonlocal descent_fuel, ascent_fuel, csm_fuel, descent_throttle, descent_gimbal_deg
        nonlocal csm_x, spawn_x, spawn_y, max_world_height, max_descent_fuel

        game_state.reset()
        world = b2World(gravity=(0, LUNAR_GRAVITY))

        # Generate terrain
        terrain_gen = ApolloTerrain(world_width_meters=WORLD_WIDTH, difficulty=TERRAIN_DIFFICULTY, roughness=TERRAIN_ROUGHNESS)
        terrain_body, terrain_pts, pads_info = terrain_gen.generate_terrain(world)

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

        lander = ApolloLander(world, position=b2Vec2(spawn_x, spawn_y), scale=0.75)

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

        # Quadruple the fuel for better gameplay (4x real Apollo)
        max_descent_fuel = 4 * 8200.0 / 100.0  # 328.0 units (4x real Apollo scaled)
        descent_fuel = max_descent_fuel
        ascent_fuel = 800.0
        descent_throttle = 0.6  # Start with 60% throttle (should hover at ~1.63x gravity force)
        descent_gimbal_deg = 0.0

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

    # Auto-enable AI in auto-run mode
    if AUTO_EPISODES > 0 and game_state.ai_agent is not None:
        game_state.ai_enabled = True
        episode_num = 1
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
                result = _log_episode(episode_num, AUTO_EPISODES, game_state, lander,
                                      descent_fuel, max_descent_fuel, target_pad,
                                      episode_step, is_point_on_pad)
                episode_results.append(result)
                auto_restart_delay = 30  # half-second pause to see result

            if auto_restart_delay > 0:
                auto_restart_delay -= 1
            else:
                episode_num += 1
                if episode_num > AUTO_EPISODES:
                    _print_summary(episode_results, ENSEMBLE_SEEDS)
                    running = False
                    continue
                new_game()
                episode_step = 0
                episode_logged = False
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

                # Gimbal control (1 degree per key press, max 6 degrees)
                elif event.key == pygame.K_COMMA:
                    descent_gimbal_deg = max(-MAX_GIMBAL_DEG, descent_gimbal_deg - 1.0)
                elif event.key == pygame.K_PERIOD:
                    descent_gimbal_deg = min(MAX_GIMBAL_DEG, descent_gimbal_deg + 1.0)
                elif event.key == pygame.K_SLASH:
                    descent_gimbal_deg = 0.0

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

            # Main engine thrust
            if main_thrust and descent_fuel > 0:
                # Calculate total lander mass (both stages welded together)
                total_mass = lander.descent_stage.mass + lander.ascent_stage.mass
                thrust_magnitude = DESCENT_THRUST_FACTOR * total_mass * descent_throttle
                gimbal_rad = math.radians(descent_gimbal_deg)
                total_angle = lander_angle + gimbal_rad

                # Thrust direction: engine fires DOWN from lander, pushing lander UP
                # In screen coords (Y-flipped), positive angle = nose tilts RIGHT visually
                # Thrust should push opposite to where nose points (out the bottom of lander)
                # Negate sin to correct for Y-flip in rendering
                thrust_x = -thrust_magnitude * math.sin(total_angle)
                thrust_y = thrust_magnitude * math.cos(total_angle)

                # Apply thrust at the combined center of mass to avoid unwanted torque
                # Calculate the combined center of mass of both stages
                descent_mass = lander.descent_stage.mass
                ascent_mass = lander.ascent_stage.mass
                descent_pos = lander.descent_stage.worldCenter
                ascent_pos = lander.ascent_stage.worldCenter

                combined_com_x = (descent_mass * descent_pos.x + ascent_mass * ascent_pos.x) / total_mass
                combined_com_y = (descent_mass * descent_pos.y + ascent_mass * ascent_pos.y) / total_mass
                combined_com = b2Vec2(combined_com_x, combined_com_y)

                # Apply force at the combined center of mass (no torque)
                lander.descent_stage.ApplyForce((thrust_x, thrust_y), combined_com, True)

                # Realistic fuel consumption: Apollo descent engine burned ~8,200kg over ~12 minutes = ~11.4 kg/s
                # At 60 FPS, that's ~0.19 kg per frame at 100% throttle
                # With mass scale 100, that's 0.0019 units per frame
                fuel_burn_rate = 0.12 * descent_throttle  # Adjusted for gameplay (was 2.0, too fast)
                descent_fuel = max(0.0, descent_fuel - fuel_burn_rate)

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
            if rcs_left_up and descent_fuel > 0:
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
            if rcs_left_down and descent_fuel > 0:
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
            if rcs_left_side and descent_fuel > 0:
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
            if rcs_right_up and descent_fuel > 0:
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
            if rcs_right_down and descent_fuel > 0:
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
            if rcs_right_side and descent_fuel > 0:
                thrust_dir_local = b2Vec2(-1.0, 0.0)  # Reaction force LEFT (flame goes RIGHT/inward)
                thrust_dir_world = lander.ascent_stage.GetWorldVector(thrust_dir_local)
                lander.ascent_stage.ApplyForce(
                    thrust_dir_world * rcs_thrust_per_thruster,
                    right_pod_world,
                    True
                )
                thrusters_firing += 1

            # Fuel consumption based on number of thrusters firing
            # Each thruster uses 0.01 fuel units per frame
            if thrusters_firing > 0:
                descent_fuel = max(0.0, descent_fuel - 0.01 * thrusters_firing)

            # AI lateral translation (actions 5/6) — apply both pod forces in same direction
            if game_state._ai_translate != 0 and descent_fuel > 0:
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
                descent_fuel = max(0.0, descent_fuel - 0.02)


        # Update physics
        world.Step(TIME_STEP, 10, 10)

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
        screen.fill((0, 0, 20))

        # Draw terrain
        draw_terrain_pygame(screen, terrain_pts, pads_info, cam_x, PPM, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y, WORLD_WIDTH)

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

        # Label
        font_mini = pygame.font.SysFont("Arial", 10, bold=True)
        minimap_label = font_mini.render("MAP", True, (150, 150, 150))
        screen.blit(minimap_label, (minimap_x + 5, minimap_y + 3))

        # Draw basic telemetry (top-left)
        font_hud = pygame.font.SysFont("consolas", 18)
        if lander:
            alt = lander.ascent_stage.position.y
            vel = lander.ascent_stage.linearVelocity
            angle_deg = math.degrees(lander.ascent_stage.angle)

            hud_x, hud_y = 10, 10
            line_height = 25

            # Altitude
            alt_color = (100, 255, 100) if alt > 15.0 else (255, 165, 0) if alt > 5.0 else (255, 100, 100)
            text = font_hud.render(f"ALT:  {alt:6.1f} m", True, alt_color)
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Vertical speed
            vv_color = (255, 255, 255) if vel.y > -2.0 else (255, 165, 0) if vel.y > -5.0 else (255, 100, 100)
            text = font_hud.render(f"Vv:   {vel.y:+6.2f} m/s", True, vv_color)
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Horizontal speed
            text = font_hud.render(f"Vh:   {vel.x:+6.2f} m/s", True, (255, 255, 255))
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Angle
            angle_color = (100, 255, 100) if abs(angle_deg) < 20 else (255, 165, 0) if abs(angle_deg) < 45 else (255, 100, 100)
            text = font_hud.render(f"ANG:  {angle_deg:+6.1f}°", True, angle_color)
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Fuel percentage
            fuel_pct = (descent_fuel / max_descent_fuel) * 100 if max_descent_fuel > 0 else 0
            fuel_color = (100, 255, 100) if fuel_pct > 50 else (255, 165, 0) if fuel_pct > 25 else (255, 100, 100)
            text = font_hud.render(f"FUEL: {fuel_pct:5.1f}%", True, fuel_color)
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Throttle
            text = font_hud.render(f"THR:  {descent_throttle*100:5.1f}%", True, (100, 200, 255))
            screen.blit(text, (hud_x, hud_y))
            hud_y += line_height

            # Gimbal angle
            gimbal_color = (255, 255, 100) if descent_gimbal_deg != 0 else (150, 150, 150)
            text = font_hud.render(f"GMB:  {descent_gimbal_deg:+5.1f}°", True, gimbal_color)
            screen.blit(text, (hud_x, hud_y))

        # Draw fuel gauge (vertical bar on left side)
        gauge_x = 10
        gauge_y = SCREEN_HEIGHT - 210
        gauge_width = 30
        gauge_height = 150

        # Background
        pygame.draw.rect(screen, (40, 40, 40), (gauge_x, gauge_y, gauge_width, gauge_height), 0)
        pygame.draw.rect(screen, (255, 255, 255), (gauge_x, gauge_y, gauge_width, gauge_height), 2)

        # Fuel level
        if lander:
            fuel_ratio = max(0.0, min(1.0, descent_fuel / max_descent_fuel))
            fuel_bar_height = int(gauge_height * fuel_ratio)
            fuel_bar_y = gauge_y + gauge_height - fuel_bar_height

            # Color based on fuel level
            if fuel_ratio > 0.5:
                gauge_color = (100, 255, 100)
            elif fuel_ratio > 0.25:
                gauge_color = (255, 165, 0)
            else:
                gauge_color = (255, 100, 100)

            if fuel_bar_height > 0:
                pygame.draw.rect(screen, gauge_color, (gauge_x + 2, fuel_bar_y, gauge_width - 4, fuel_bar_height), 0)

        # Label
        font_small = pygame.font.SysFont("consolas", 12)
        fuel_label = font_small.render("FUEL", True, (255, 255, 255))
        screen.blit(fuel_label, (gauge_x + 2, gauge_y + gauge_height + 5))

        # Draw attitude indicator (circular dial showing angle)
        if lander:
            att_center_x = 80
            att_center_y = SCREEN_HEIGHT - 80
            att_radius = 50

            # Get current angle
            current_angle_deg = math.degrees(lander.ascent_stage.angle)
            current_angle_rad = lander.ascent_stage.angle

            # Background circle
            pygame.draw.circle(screen, (30, 30, 30), (att_center_x, att_center_y), att_radius, 0)
            pygame.draw.circle(screen, (100, 100, 100), (att_center_x, att_center_y), att_radius, 2)

            # Draw tick marks every 15 degrees
            for tick_deg in range(-90, 91, 15):
                tick_rad = math.radians(tick_deg)
                inner_r = att_radius - 8 if tick_deg % 45 == 0 else att_radius - 5
                outer_r = att_radius - 2

                x1 = att_center_x + inner_r * math.sin(tick_rad)
                y1 = att_center_y - inner_r * math.cos(tick_rad)
                x2 = att_center_x + outer_r * math.sin(tick_rad)
                y2 = att_center_y - outer_r * math.cos(tick_rad)

                tick_color = (255, 255, 255) if tick_deg == 0 else (150, 150, 150)
                pygame.draw.line(screen, tick_color, (x1, y1), (x2, y2), 2 if tick_deg % 45 == 0 else 1)

            # Draw horizon line (fixed)
            horizon_len = att_radius - 15
            pygame.draw.line(screen, (0, 150, 255),
                           (att_center_x - horizon_len, att_center_y),
                           (att_center_x + horizon_len, att_center_y), 2)

            # Draw lander indicator (rotates with lander)
            indicator_len = att_radius - 10
            indicator_tip_x = att_center_x + indicator_len * math.sin(current_angle_rad)
            indicator_tip_y = att_center_y - indicator_len * math.cos(current_angle_rad)

            # Color based on angle
            if abs(current_angle_deg) < 15:
                indicator_color = (100, 255, 100)  # Green - safe
            elif abs(current_angle_deg) < 30:
                indicator_color = (255, 255, 0)    # Yellow - caution
            elif abs(current_angle_deg) < 45:
                indicator_color = (255, 165, 0)    # Orange - warning
            else:
                indicator_color = (255, 100, 100)  # Red - danger

            # Draw lander body indicator (triangle)
            pygame.draw.line(screen, indicator_color,
                           (att_center_x, att_center_y),
                           (indicator_tip_x, indicator_tip_y), 3)

            # Draw small wings
            wing_len = 15
            wing_angle_offset = math.pi / 2  # 90 degrees
            left_wing_x = att_center_x + wing_len * math.sin(current_angle_rad - wing_angle_offset)
            left_wing_y = att_center_y - wing_len * math.cos(current_angle_rad - wing_angle_offset)
            right_wing_x = att_center_x + wing_len * math.sin(current_angle_rad + wing_angle_offset)
            right_wing_y = att_center_y - wing_len * math.cos(current_angle_rad + wing_angle_offset)

            pygame.draw.line(screen, indicator_color, (att_center_x, att_center_y), (left_wing_x, left_wing_y), 2)
            pygame.draw.line(screen, indicator_color, (att_center_x, att_center_y), (right_wing_x, right_wing_y), 2)

            # Center dot
            pygame.draw.circle(screen, indicator_color, (att_center_x, att_center_y), 4, 0)

            # Angle readout below
            font_att = pygame.font.SysFont("consolas", 14, bold=True)
            angle_text = font_att.render(f"{current_angle_deg:+.1f}", True, indicator_color)
            text_rect = angle_text.get_rect(center=(att_center_x, att_center_y + att_radius + 15))
            screen.blit(angle_text, text_rect)

            # Label
            att_label = font_small.render("ATT", True, (255, 255, 255))
            screen.blit(att_label, (att_center_x - 10, att_center_y - att_radius - 18))

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
                if vote == "3/3":
                    vote_color = (0, 255, 0)  # green = unanimous
                elif vote == "2/3":
                    vote_color = (255, 255, 0)  # yellow = majority
                else:
                    vote_color = (255, 100, 100)  # red = split (master decides)
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
