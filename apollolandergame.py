"""
Apollo Lunar Lander Game - Atari Style

A 3-screen scrolling lunar lander game with:
- Procedurally generated terrain with multiple landing pads
- Fuel management (descent and ascent stages)
- Scoring system based on pad difficulty and fuel efficiency
- Realistic Apollo LM physics and controls
- Camera scrolling to follow the lander
"""

import math
import pygame
import random
import sys
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

# -------------------------------------------------
# Config
# -------------------------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 1000  # Larger play area for better visibility
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS

# Planet properties: gravity (m/s²), linear damping (air resistance), angular damping (rotation resistance)
PLANET_PROPERTIES = {
    # Format: gravity (m/s²), damping, diameter (km), low orbit altitude (km)
    # Orbital altitude chosen to maintain similar gameplay (1.05-1.06x radius ratio)
    "mercury":  {"gravity": 3.7,   "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 4879,  "orbit_alt": 130},   # No atmosphere
    "venus":    {"gravity": 8.87,  "linear_damping": 0.35, "angular_damping": 0.4,  "diameter": 12104, "orbit_alt": 300},   # Very thick atmosphere
    "earth":    {"gravity": 9.81,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 12742, "orbit_alt": 350},   # Standard atmosphere (ISS altitude)
    "luna":     {"gravity": 1.62,  "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 3475,  "orbit_alt": 100},   # No atmosphere (Moon, Apollo ~100km)
    "mars":     {"gravity": 3.71,  "linear_damping": 0.05, "angular_damping": 0.15, "diameter": 6779,  "orbit_alt": 180},   # Thin atmosphere
    "jupiter":  {"gravity": 24.79, "linear_damping": 0.4,  "angular_damping": 0.5,  "diameter": 139820, "orbit_alt": 4200}, # Very thick atmosphere (cloud top)
    "saturn":   {"gravity": 10.44, "linear_damping": 0.3,  "angular_damping": 0.4,  "diameter": 116460, "orbit_alt": 3500}, # Thick atmosphere (cloud top)
    "uranus":   {"gravity": 8.69,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 50724, "orbit_alt": 1500},  # Moderate atmosphere
    "neptune":  {"gravity": 11.15, "linear_damping": 0.3,  "angular_damping": 0.35, "diameter": 49244, "orbit_alt": 1500},  # Thick atmosphere
}

# Parse command-line arguments for planet selection
SELECTED_PLANET = "luna"  # Default to Moon
if len(sys.argv) > 1:
    planet_arg = sys.argv[1].lower()
    if planet_arg in PLANET_PROPERTIES:
        SELECTED_PLANET = planet_arg
    else:
        print(f"Unknown planet '{sys.argv[1]}'. Available planets:")
        for planet in PLANET_PROPERTIES.keys():
            print(f"  - {planet}")
        print(f"Using default: {SELECTED_PLANET}")

# Get planet properties
PLANET = PLANET_PROPERTIES[SELECTED_PLANET]
LUNAR_GRAVITY = -PLANET["gravity"]  # Negative for downward
LINEAR_DAMPING = PLANET["linear_damping"]
ANGULAR_DAMPING = PLANET["angular_damping"]

# Calculate thrust factors based on real Apollo specifications
# Real Apollo Descent Engine: 45,040 N max / 10,246 kg = 2.71:1 thrust-to-weight ratio on Moon
# Real Apollo Ascent Engine: 15,600 N / 4,819 kg = 2.0:1 thrust-to-weight ratio on Moon
DESCENT_THRUST_FACTOR = PLANET["gravity"] * 2.71  # Matches real Apollo descent engine
ASCENT_THRUST_FACTOR = PLANET["gravity"] * 2.0   # Matches real Apollo ascent engine

print(f"🚀 Playing on {SELECTED_PLANET.upper()}")
print(f"   Gravity: {PLANET['gravity']} m/s²")
print(f"   Descent thrust: {DESCENT_THRUST_FACTOR:.2f}x mass")
print(f"   Ascent thrust: {ASCENT_THRUST_FACTOR:.2f}x mass")
print(f"   Atmosphere: {'None' if LINEAR_DAMPING == 0.0 else f'Linear damping {LINEAR_DAMPING}, Angular damping {ANGULAR_DAMPING}'}")

# Game world size - dynamically scaled based on planet diameter and orbital altitude
# Scale factor: convert real km to game meters (1:1000 ratio for playability)
SCALE_FACTOR = 1000.0  # 1km in reality = 1m in game
PLANET_DIAMETER_METERS = PLANET["diameter"] * 1000.0 / SCALE_FACTOR  # Convert km to game meters
ORBITAL_ALTITUDE_METERS = PLANET["orbit_alt"] * 1000.0 / SCALE_FACTOR  # Convert km to game meters

# Calculate how many screens wide the world needs to be
SCREEN_WIDTH_METERS = SCREEN_WIDTH / PPM
WORLD_SCREENS = PLANET_DIAMETER_METERS / SCREEN_WIDTH_METERS
WORLD_WIDTH = PLANET_DIAMETER_METERS  # Diameter of cylindrical world

print(f"   Diameter: {PLANET['diameter']} km (game: {PLANET_DIAMETER_METERS:.1f}m)")
print(f"   Orbital altitude: {PLANET['orbit_alt']} km (game: {ORBITAL_ALTITUDE_METERS:.1f}m)")
print(f"   World screens: {WORLD_SCREENS:.1f} wide")

# Fuel capacity (real Apollo specifications in kg)
# Real Apollo descent: 8,248 kg fuel, 10,246 kg total → 1,998 kg dry mass
# Real Apollo ascent: 2,353 kg fuel, 4,819 kg total → 2,466 kg dry mass
# Real Apollo CSM: 18,413 kg fuel, 30,080 kg total → 11,667 kg dry mass
MAX_DESCENT_FUEL = 8248.0  # kg (real Apollo LM descent propellant)
MAX_ASCENT_FUEL = 2353.0   # kg (real Apollo LM ascent propellant)
MAX_CSM_FUEL = 18413.0     # kg (real Apollo SPS propellant)

# Dry masses (structure without fuel)
DESCENT_DRY_MASS = 1998.0  # kg (descent stage structure)
ASCENT_DRY_MASS = 2466.0   # kg (ascent stage structure)
CSM_DRY_MASS = 11667.0     # kg (CM + SM structure)

# Fuel consumption rates (kg per second at full throttle)
# Based on real Apollo specific impulse (Isp) and thrust
# Consumption = Thrust / (Isp × g₀), where g₀ = 9.81 m/s²
DESCENT_FUEL_RATE = 45040.0 / (311.0 * 9.81)  # ~14.77 kg/s (real Apollo descent)
ASCENT_FUEL_RATE = 15600.0 / (311.0 * 9.81)   # ~5.12 kg/s (real Apollo ascent)
CSM_FUEL_RATE = 91190.0 / (314.0 * 9.81)      # ~29.59 kg/s (real Apollo SPS)
RCS_FUEL_RATE = 0.1  # kg per thruster per second (negligible for gameplay)

# Scoring
BASE_LANDING_SCORE = 500
UNDOCKING_BONUS = 200  # Bonus for successful undocking from CSM
SEPARATION_BONUS = 300  # Bonus for separating ascent from descent stage
DOCKING_BONUS = 1000  # Bonus for successful CSM docking
JETTISON_BONUS = 400  # Bonus for jettisoning ascent module after docking
ESCAPE_BONUS = 1500  # Bonus for reaching escape altitude with CSM
FUEL_BONUS_MULTIPLIER = 0.5  # points per fuel unit remaining


# -------------------------------------------------
# Game State
# -------------------------------------------------
class GameState:
    """Game state manager."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset game state for new game."""
        self.playing = True
        self.landed = False
        self.crashed = False
        self.docked = True  # Start docked to CSM
        self.undocked = False  # True after pressing U to undock
        self.deorbiting = False  # True when lander slows down enough to start descent
        self.ascending = False  # True after successful landing, flying to CSM
        self.mission_complete = False  # True after docking with CSM
        self.jettisoned = False  # True after jettisoning ascent module from CSM
        self.score = 0
        self.undocking_score = 0
        self.separation_score = 0
        self.landing_score = 0
        self.docking_score = 0
        self.jettison_score = 0
        self.escape_score = 0
        self.fuel_bonus = 0
        self.pad_multiplier = 1


# -------------------------------------------------
# Main Game
# -------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Apollo Lunar Lander - Atari Style")
    clock = pygame.time.Clock()

    # Initialize game
    game_state = GameState()
    hud = ApolloHUD(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Initialize variables that will be set by new_game()
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
    max_world_height = 136.4  # Will be set properly in new_game()
    spawn_x = 0.0
    spawn_y = 0.0
    cam_x = 0.0
    cam_y = 0.0
    descent_fuel = 0.0
    ascent_fuel = 0.0
    csm_fuel = 0.0
    descent_throttle = 0.0
    descent_gimbal_deg = 0.0

    # Start new game
    def new_game():
        """Initialize a new game session."""
        nonlocal world, terrain_gen, terrain_body, terrain_pts, pads_info
        nonlocal lander, target_pad, target_pad_index, cam_x, cam_y, csm, csm_altitude
        nonlocal descent_fuel, ascent_fuel, csm_fuel, descent_throttle, descent_gimbal_deg
        nonlocal csm_x, spawn_x, spawn_y, max_world_height

        # Reset game state
        game_state.reset()

        # Create physics world
        world = b2World(gravity=(0, LUNAR_GRAVITY))

        # Generate terrain
        difficulty = 2  # Medium difficulty
        terrain_gen = ApolloTerrain(world_width_meters=WORLD_WIDTH, difficulty=difficulty)
        terrain_body, terrain_pts, pads_info = terrain_gen.generate_terrain(world)

        # Select random target pad
        target_pad_index = random.randint(0, len(pads_info) - 1)
        target_pad = pads_info[target_pad_index]
        target_x = (target_pad["x1"] + target_pad["x2"]) / 2.0

        # Create CSM at orbital altitude based on planet properties
        # Orbital altitude comes from planet table, scaled to game meters
        csm_altitude = ORBITAL_ALTITUDE_METERS

        # Calculate max world height (CSM altitude + 10% margin for maneuvering)
        max_world_height = csm_altitude * 1.1  # 10% above orbital altitude
        csm_x = random.uniform(WORLD_WIDTH * 0.1, WORLD_WIDTH * 0.3)  # Start at left side
        csm_y = csm_altitude
        csm = ApolloCSM(world, position=(csm_x, csm_y), scale=1.125)  # Reduced by 25% from 1.5

        # Apply atmospheric damping (air resistance)
        csm.body.linearDamping = LINEAR_DAMPING
        csm.body.angularDamping = ANGULAR_DAMPING

        # Make CSM immune to gravity by setting it as kinematic-like
        csm.body.gravityScale = 0.0  # No gravity affects CSM
        csm.body.fixedRotation = True  # Prevent rotation

        # Calculate CSM docking port position in world coordinates
        # CSM is rotated 90° CCW (pointing left), so docking port is to the LEFT
        # For scale 1.125: SM extends 2.25m, CM extends 1.8m, dock extends ~0.16m
        # Total from center to docking port = 2.25 + 1.8 + 0.16 = 4.21m to the LEFT
        csm_scale = 1.125
        sm_half_h = 2.0 * csm_scale  # 3.0m
        cm_length = 1.6 * csm_scale  # 2.4m
        dock_offset = 0.3 * csm_scale * 0.7  # 0.315m
        docking_port_offset_x = -(sm_half_h + cm_length + dock_offset)  # ~-5.7m to the LEFT

        # Spawn lander docked to CSM (ascent module docking port aligned with CSM docking port)
        # Lander rotated 90° clockwise (pointing right when docked)
        # Docking port is at top of ascent module when upright (h*1.1 above center)
        # When rotated 90° CW, docking port points LEFT (toward CSM)
        lander_scale = 0.5625  # Reduced by 25% from 0.75
        lander_h = 1.5 * lander_scale
        lander_dock_offset = lander_h * 1.1  # 1.2375m from center to docking port

        # When lander is rotated 90° CW (-π/2), its docking port (originally "up") points RIGHT
        # Rotation of point (0, h*1.1) by -90° (CW): x'=+(h*1.1), y'=0 -> points RIGHT
        # CSM docking port world position: csm_x + docking_port_offset_x (to the left of CSM center)
        # Lander docking port world position when rotated 90° CW: lander_x + lander_dock_offset (to the right)
        # For docking ports to touch: csm_x + docking_port_offset_x = lander_x + lander_dock_offset
        # Solving for lander_x: lander_x = csm_x + docking_port_offset_x - lander_dock_offset
        csm_dock_world_x = csm_x + docking_port_offset_x  # CSM docking port position in world
        spawn_x = csm_dock_world_x - lander_dock_offset  # Position lander so its dock port touches CSM's
        spawn_y = csm_y  # Same altitude as CSM

        lander = ApolloLander(world, position=b2Vec2(spawn_x, spawn_y), scale=0.5625)  # Reduced by 25% from 0.75

        # Rotate lander 90° clockwise to be horizontal (pointing right, docking port toward CSM on left)
        lander.descent_stage.angle = -math.pi / 2.0  # -90° (clockwise)
        lander.ascent_stage.angle = -math.pi / 2.0

        # Recreate the weld joint with correct offset for horizontal orientation
        # When at -90° (horizontal, pointing right):
        # - Ascent octagon bottom (now left side): at body.x - h where h = 1.5*scale
        # - Descent box right edge: at body.x + half_w where half_w = 2.1*scale
        # For descent right edge to touch ascent left edge:
        #   descent.x + descent_half_w = ascent.x - ascent_h
        #   descent.x = ascent.x - ascent_h - descent_half_w
        # BUT the offset from the original spawn was 1.85*scale vertically
        # When rotated, this becomes horizontal: use the original offset
        descent_offset = 1.85 * lander.scale  # Original vertical offset, now horizontal

        # Position descent stage correctly
        lander.descent_stage.position = b2Vec2(
            lander.ascent_stage.position.x - descent_offset,
            lander.ascent_stage.position.y
        )

        # Destroy old joint and create new one at the correct anchor point
        if lander.connection_joint is not None:
            world.DestroyJoint(lander.connection_joint)

        # Create weld joint at the connection point between stages
        from Box2D import b2WeldJointDef
        jd = b2WeldJointDef()
        anchor = (lander.descent_stage.worldCenter + lander.ascent_stage.worldCenter) * 0.5
        jd.Initialize(lander.descent_stage, lander.ascent_stage, anchor)
        jd.collideConnected = False
        lander.connection_joint = world.CreateJoint(jd)

        # Apply atmospheric damping (air resistance)
        lander.descent_stage.linearDamping = LINEAR_DAMPING
        lander.descent_stage.angularDamping = ANGULAR_DAMPING
        lander.ascent_stage.linearDamping = LINEAR_DAMPING
        lander.ascent_stage.angularDamping = ANGULAR_DAMPING

        # Make lander immune to gravity while docked
        lander.descent_stage.gravityScale = 0.0
        lander.ascent_stage.gravityScale = 0.0

        # Camera follows ascent module (visual focus)
        cam_x = lander.ascent_stage.position.x
        cam_y = lander.ascent_stage.position.y

        # Fuel
        descent_fuel = MAX_DESCENT_FUEL
        ascent_fuel = MAX_ASCENT_FUEL
        csm_fuel = MAX_CSM_FUEL

        # Descent engine controls
        descent_throttle = 0.0
        descent_gimbal_deg = 0.0

    # Initialize first game
    new_game()

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Restart game
                    new_game()
                elif event.key == pygame.K_u and game_state.docked and not game_state.undocked:
                    # Undock from CSM
                    game_state.undocked = True
                    game_state.docked = False
                    game_state.undocking_score = UNDOCKING_BONUS
                    # Note: gravity will be handled by orbital mechanics system
                    # Lander will stay in orbit until it slows down horizontally
                    # Push lander away from CSM (small impulse downward to start deorbit)
                    # Apply impulse at center of mass to avoid rotation
                    lander.descent_stage.ApplyLinearImpulse(b2Vec2(0.0, -30.0), lander.descent_stage.worldCenter, True)
                    lander.ascent_stage.ApplyLinearImpulse(b2Vec2(0.0, -30.0), lander.ascent_stage.worldCenter, True)
                elif event.key == pygame.K_j and game_state.docked and not game_state.jettisoned:
                    # Jettison ascent module from CSM after docking
                    game_state.jettisoned = True
                    game_state.docked = False
                    game_state.mission_complete = False  # Game continues with CSM control
                    game_state.jettison_score = JETTISON_BONUS
                    # Re-enable gravity for jettisoned ascent module
                    lander.ascent_stage.gravityScale = 1.0
                    # Push ascent module away with small impulse
                    lander.ascent_stage.ApplyLinearImpulse(b2Vec2(0.0, -20.0), lander.ascent_stage.worldCenter, True)
                    # Enable CSM rotation and gravity for free flight
                    csm.body.fixedRotation = False  # Allow rotation
                    csm.body.gravityScale = 1.0
                elif event.key == pygame.K_SPACE and game_state.playing and game_state.undocked:
                    lander.separate()
                    game_state.separation_score = SEPARATION_BONUS

        keys = pygame.key.get_pressed()

        # Only process physics and controls if game is playing
        if game_state.playing:
            ascent_freed = lander.connection_joint is None
            descent = lander.descent_stage
            ascent = lander.ascent_stage

            # Active body for telemetry and control
            # After jettison, control passes to CSM
            if game_state.jettisoned:
                active_body = csm.body
            else:
                active_body = ascent if ascent_freed else descent

            # Get ascent data for RCS calculations
            asc_data = getattr(ascent, "userData", {}) or {}
            asc_scale = asc_data.get("scale", 1.0)

            # Only allow controls after undocking OR after jettison (CSM control)
            controls_enabled = game_state.undocked or game_state.jettisoned

            # Update descent throttle & gimbal only while attached
            if not ascent_freed and controls_enabled:
                # Throttle (real Apollo: 10% to 100% throttleable range)
                # 0% = off, then jumps to 10% minimum, then smooth to 100%
                if keys[pygame.K_UP]:
                    if descent_throttle < 0.01:
                        # Jump from 0% to 10% minimum
                        descent_throttle = 0.10
                    else:
                        # Smooth increment from 10% to 100%
                        descent_throttle = min(1.0, descent_throttle + 0.01)
                if keys[pygame.K_DOWN]:
                    if descent_throttle <= 0.10:
                        # Below 10%, go to 0% (engine off)
                        descent_throttle = 0.0
                    else:
                        # Smooth decrement from 100% to 10%
                        descent_throttle = max(0.10, descent_throttle - 0.01)

                # Gimbal left/right (comma/period) and recenter (/)
                if keys[pygame.K_COMMA]:
                    descent_gimbal_deg = max(-MAX_GIMBAL_DEG, descent_gimbal_deg - 0.15)
                if keys[pygame.K_PERIOD]:
                    descent_gimbal_deg = min(MAX_GIMBAL_DEG, descent_gimbal_deg + 0.15)
                if keys[pygame.K_SLASH]:
                    descent_gimbal_deg = 0.0
            else:
                # Once separated, descent engine is off
                descent_throttle = 0.0

            # Flags for RCS flames
            asc_tl = asc_bl = asc_tr = asc_br = False
            asc_sl = asc_sr = False
            csm_tl = csm_bl = csm_tr = csm_br = False
            csm_sl = csm_sr = False

            # Main engine flags
            main_descent_on = False
            main_ascent_on = False
            main_csm_on = False

            # Fuel consumption tracking
            active_rcs_count = 0

            # Main engines
            if not ascent_freed:
                # Descent engine (variable thrust)
                if descent_throttle > 0.0 and descent_fuel > 0 and controls_enabled:
                    main_descent_on = True
                    # Thrust with gimbal
                    rad = math.radians(descent_gimbal_deg)
                    local_dir = b2Vec2(math.sin(rad), math.cos(rad))
                    thrust_vec = descent.GetWorldVector((local_dir.x, local_dir.y))
                    descent.ApplyForce(
                        descent.mass * DESCENT_THRUST_FACTOR * descent_throttle * thrust_vec,
                        descent.worldCenter,
                        True,
                    )
                    # Consume fuel
                    descent_fuel -= DESCENT_FUEL_RATE * descent_throttle * TIME_STEP
                    descent_fuel = max(0.0, descent_fuel)

                    # Update descent stage mass (dry mass + remaining fuel)
                    new_mass = (DESCENT_DRY_MASS + descent_fuel) / MASS_SCALE
                    mass_data = descent.massData
                    mass_data.mass = new_mass
                    descent.massData = mass_data
            else:
                # Ascent engine (pulse-only) OR CSM main engine
                if keys[pygame.K_UP] and controls_enabled:
                    if game_state.jettisoned and csm_fuel > 0:
                        # CSM main engine (SPS)
                        csm.apply_main_thrust(TIME_STEP)
                        main_csm_on = True
                        # Consume CSM fuel
                        csm_fuel -= CSM_FUEL_RATE * TIME_STEP
                        csm_fuel = max(0.0, csm_fuel)

                        # Update CSM mass (dry mass + remaining fuel)
                        new_csm_mass = CSM_DRY_MASS + csm_fuel
                        csm_mass_data = csm.body.massData
                        csm_mass_data.mass = new_csm_mass
                        csm.body.massData = csm_mass_data
                    elif ascent_fuel > 0:
                        # Ascent module engine
                        main_ascent_on = True
                        up_vec = ascent.GetWorldVector((0, 1))
                        ascent.ApplyForce(
                            ascent.mass * ASCENT_THRUST_FACTOR * up_vec,
                            ascent.worldCenter,
                            True,
                        )
                        # Consume fuel
                        ascent_fuel -= ASCENT_FUEL_RATE * TIME_STEP
                        ascent_fuel = max(0.0, ascent_fuel)

                        # Update ascent stage mass (dry mass + remaining fuel)
                        new_ascent_mass = (ASCENT_DRY_MASS + ascent_fuel) / MASS_SCALE
                        ascent_mass_data = ascent.massData
                        ascent_mass_data.mass = new_ascent_mass
                        ascent.massData = ascent_mass_data

            # RCS forces (on ascent module OR CSM after jettison)
            if game_state.jettisoned:
                # Use CSM for RCS
                rcs_body = csm.body
                rcs_force_mag = csm.body.mass * RCS_THRUST_FACTOR
                # CSM RCS offsets (use CSM scale)
                csm_scale_rcs = 1.125
                rcs_x = RCS_OFFSET_X * csm_scale_rcs
                rcs_y = RCS_OFFSET_Y * csm_scale_rcs
                up_off = RCS_OFFSET_UP * csm_scale_rcs
            else:
                # Use ascent module for RCS
                rcs_body = ascent
                rcs_force_mag = ascent.mass * RCS_THRUST_FACTOR
                rcs_x = RCS_OFFSET_X * asc_scale
                rcs_y = RCS_OFFSET_Y * asc_scale
                up_off = RCS_OFFSET_UP * asc_scale

            def apply_thruster(body, local_pos: b2Vec2, local_force: b2Vec2):
                """Apply RCS force."""
                nonlocal active_rcs_count
                wp = body.GetWorldPoint(local_pos)
                wf = body.GetWorldVector(local_force)
                body.ApplyForce(wf * rcs_force_mag, wp, True)
                active_rcs_count += 1

            # RCS Roll (Q/E) - only if controls enabled
            if keys[pygame.K_q] and controls_enabled:
                rb_pos = b2Vec2(rcs_x, rcs_y - up_off)
                apply_thruster(rcs_body, rb_pos, b2Vec2(0.0, 1.0))
                if game_state.jettisoned:
                    csm_br = True
                else:
                    asc_br = True

                lt_pos = b2Vec2(-rcs_x, rcs_y + up_off)
                apply_thruster(rcs_body, lt_pos, b2Vec2(0.0, -1.0))
                if game_state.jettisoned:
                    csm_tl = True
                else:
                    asc_tl = True

            if keys[pygame.K_e] and controls_enabled:
                rt_pos = b2Vec2(rcs_x, rcs_y + up_off)
                apply_thruster(rcs_body, rt_pos, b2Vec2(0.0, -1.0))
                if game_state.jettisoned:
                    csm_tr = True
                else:
                    asc_tr = True

                lb_pos = b2Vec2(-rcs_x, rcs_y - up_off)
                apply_thruster(rcs_body, lb_pos, b2Vec2(0.0, 1.0))
                if game_state.jettisoned:
                    csm_bl = True
                else:
                    asc_bl = True

            # RCS Translation (LEFT/RIGHT)
            if keys[pygame.K_LEFT] and controls_enabled:
                r_pos = b2Vec2(rcs_x, rcs_y)
                apply_thruster(rcs_body, r_pos, b2Vec2(-1.0, 0.0))
                if game_state.jettisoned:
                    csm_sr = True
                else:
                    asc_sr = True

            if keys[pygame.K_RIGHT] and controls_enabled:
                l_pos = b2Vec2(-rcs_x, rcs_y)
                apply_thruster(rcs_body, l_pos, b2Vec2(1.0, 0.0))
                if game_state.jettisoned:
                    csm_sl = True
                else:
                    asc_sl = True

            # RCS Translation (UP/DOWN) - fire both thrusters on same side
            # W = both up thrusters (translate upward)
            if keys[pygame.K_w] and controls_enabled:
                # Fire top-left thruster downward
                lt_pos = b2Vec2(-rcs_x, rcs_y + up_off)
                apply_thruster(rcs_body, lt_pos, b2Vec2(0.0, -1.0))
                if game_state.jettisoned:
                    csm_tl = True
                else:
                    asc_tl = True

                # Fire top-right thruster downward
                rt_pos = b2Vec2(rcs_x, rcs_y + up_off)
                apply_thruster(rcs_body, rt_pos, b2Vec2(0.0, -1.0))
                if game_state.jettisoned:
                    csm_tr = True
                else:
                    asc_tr = True

            # S = both down thrusters (translate downward)
            if keys[pygame.K_s] and controls_enabled:
                # Fire bottom-left thruster upward
                lb_pos = b2Vec2(-rcs_x, rcs_y - up_off)
                apply_thruster(rcs_body, lb_pos, b2Vec2(0.0, 1.0))
                if game_state.jettisoned:
                    csm_bl = True
                else:
                    asc_bl = True

                # Fire bottom-right thruster upward
                rb_pos = b2Vec2(rcs_x, rcs_y - up_off)
                apply_thruster(rcs_body, rb_pos, b2Vec2(0.0, 1.0))
                if game_state.jettisoned:
                    csm_br = True
                else:
                    asc_br = True

            # Consume RCS fuel
            if active_rcs_count > 0:
                fuel_to_use = ascent_fuel if ascent_freed else descent_fuel
                fuel_consumed = RCS_FUEL_RATE * active_rcs_count * TIME_STEP
                fuel_to_use -= fuel_consumed
                fuel_to_use = max(0.0, fuel_to_use)

                if ascent_freed:
                    ascent_fuel = fuel_to_use
                else:
                    descent_fuel = fuel_to_use

            # ==========================================
            # Orbital Mechanics System
            # ==========================================
            # Objects at CSM altitude with ~CSM velocity are in "orbit" (immune to gravity)
            # Gravity influence ramps up/down based on deviation from orbital parameters

            def calculate_orbital_gravity_scale(altitude):
                """
                Calculate gravity scale based on orbital mechanics.

                At orbital altitude (near CSM), gravity scale = 0.0 (orbit)
                Outside orbital altitude, gravity scale = 1.0 (normal gravity)
                In between, gravity ramps smoothly.

                Orbital parameters:
                - Altitude: ~86m (CSM altitude) ± 10m
                - Velocity is NOT checked - only altitude matters

                Args:
                    altitude: Object altitude above terrain (meters)

                Returns:
                    gravity_scale: 0.0 (orbit) to 1.0 (full gravity)
                """
                # Orbital parameters
                orbital_altitude = csm_altitude  # ~86m

                # Tolerance range
                altitude_tolerance = 10.0  # ±10m

                # Calculate deviation from orbital altitude
                altitude_deviation = abs(altitude - orbital_altitude)

                # Normalize deviation (0.0 = perfect match, 1.0+ = outside tolerance)
                altitude_factor = altitude_deviation / altitude_tolerance

                # Clamp between 0.0 and 1.0
                orbital_factor = min(1.0, max(0.0, altitude_factor))

                # Gravity scale: 0.0 when at orbital altitude, 1.0 when far from it
                # Smooth transition between the two states
                return orbital_factor

            # CSM kinematic movement (before physics step)
            # CSM moves left to right at constant altitude ONLY when not jettisoned
            if not game_state.jettisoned:
                # Automatic orbit movement
                new_csm_x = csm.body.position.x + 1.5 * TIME_STEP
                csm.body.position = b2Vec2(new_csm_x, csm_altitude)
                csm.body.linearVelocity = b2Vec2(1.5, 0.0)

                # CSM wraps (left to right across world) - seamless wrapping at WORLD_WIDTH
                if csm.body.position.x > WORLD_WIDTH:
                    wrap_offset = csm.body.position.x - WORLD_WIDTH
                    new_wrapped_x = 0.0 + wrap_offset
                    csm.body.position = b2Vec2(new_wrapped_x, csm_altitude)
                    new_csm_x = new_wrapped_x
                elif csm.body.position.x < 0:
                    wrap_offset = 0 - csm.body.position.x
                    new_wrapped_x = WORLD_WIDTH - wrap_offset
                    csm.body.position = b2Vec2(new_wrapped_x, csm_altitude)
                    new_csm_x = new_wrapped_x
            else:
                # After jettison, CSM is in free flight - handle wrapping for physics body
                if csm.body.position.x > WORLD_WIDTH:
                    csm.body.position = b2Vec2(csm.body.position.x - WORLD_WIDTH, csm.body.position.y)
                elif csm.body.position.x < 0:
                    csm.body.position = b2Vec2(csm.body.position.x + WORLD_WIDTH, csm.body.position.y)

            # Apply orbital mechanics to lander (always check altitude for gravity)
            # Calculate lander altitude
            lander_altitude = active_body.position.y  # Altitude above zero (approximate)

            # Calculate gravity scale based on altitude only (velocity doesn't matter)
            gravity_scale = calculate_orbital_gravity_scale(lander_altitude)

            # Apply to both stages (but not to ascent when docked - it stays with CSM)
            lander.descent_stage.gravityScale = gravity_scale
            if not game_state.docked:
                lander.ascent_stage.gravityScale = gravity_scale

            # Calculate atmospheric damping based on altitude
            # Damping is at full value at surface (altitude 0) and reduces to 0 at lower blue line
            # Using logarithmic curve for realistic atmospheric thinning
            def calculate_atmospheric_damping(altitude, max_damping):
                """
                Calculate atmospheric damping based on altitude.

                Args:
                    altitude: Current altitude above terrain (meters)
                    max_damping: Maximum damping value at surface

                Returns:
                    damping: 0.0 (no atmosphere) to max_damping (full atmosphere)
                """
                lower_boundary = csm_altitude - 10.0  # Lower blue line (zero gravity start)

                # If at or above the lower boundary, no damping
                if altitude >= lower_boundary:
                    return 0.0

                # If at or below surface, full damping
                if altitude <= 0.0:
                    return max_damping

                # Logarithmic falloff between surface and lower boundary
                # Using natural log for smooth exponential atmosphere model
                # Scale factor: damping = max * exp(-altitude / scale_height)
                scale_height = lower_boundary / 3.0  # Atmosphere scale height
                damping_factor = math.exp(-altitude / scale_height)

                return max_damping * damping_factor

            # Apply altitude-based damping to lander stages
            terrain_height_descent = get_terrain_height_at(lander.descent_stage.position.x, terrain_pts)
            altitude_descent = lander.descent_stage.position.y - terrain_height_descent
            lander.descent_stage.linearDamping = calculate_atmospheric_damping(altitude_descent, LINEAR_DAMPING)
            lander.descent_stage.angularDamping = calculate_atmospheric_damping(altitude_descent, ANGULAR_DAMPING)

            if not game_state.docked:
                terrain_height_ascent = get_terrain_height_at(lander.ascent_stage.position.x, terrain_pts)
                altitude_ascent = lander.ascent_stage.position.y - terrain_height_ascent
                lander.ascent_stage.linearDamping = calculate_atmospheric_damping(altitude_ascent, LINEAR_DAMPING)
                lander.ascent_stage.angularDamping = calculate_atmospheric_damping(altitude_ascent, ANGULAR_DAMPING)

            # Apply altitude-based damping to CSM
            if game_state.jettisoned:
                terrain_height_csm = get_terrain_height_at(csm.body.position.x, terrain_pts)
                altitude_csm = csm.body.position.y - terrain_height_csm
                csm.body.linearDamping = calculate_atmospheric_damping(altitude_csm, LINEAR_DAMPING)
                csm.body.angularDamping = calculate_atmospheric_damping(altitude_csm, ANGULAR_DAMPING)

            # Step physics
            world.Step(TIME_STEP, 6, 2)

            # Update camera to follow active vehicle
            # Track CSM after jettison, otherwise track ascent module
            if game_state.jettisoned:
                cam_x = csm.body.position.x
                cam_y = csm.body.position.y
            else:
                cam_x = ascent.position.x
                cam_y = ascent.position.y

            # Check landing/crash conditions
            pos_x = active_body.position.x
            pos_y = active_body.position.y
            terrain_height = get_terrain_height_at(pos_x, terrain_pts)
            altitude = pos_y - terrain_height

            vel = active_body.linearVelocity
            vel_y = vel.y
            vel_x = vel.x
            angle_deg = math.degrees(active_body.angle)

            # Check if on target pad (check body center is on or near pad)
            on_target_pad = is_point_on_pad(pos_x, pos_y, target_pad, tolerance_x=2.5, tolerance_y=3.0)

            # Landing detection - check if lander is nearly stationary on ground
            # Use velocity as primary indicator (more reliable than altitude for feet contact)
            is_on_ground = altitude < 3.5  # Body center within 3.5m of terrain (feet extended ~2.4m)
            is_nearly_stopped = abs(vel_y) < 1.0 and abs(vel_x) < 1.0  # Forgiving velocity thresholds

            if is_on_ground and is_nearly_stopped and not game_state.ascending:
                # Check landing conditions
                # Successful landing: on target pad AND reasonable angle
                if on_target_pad and abs(angle_deg) < 20:
                    game_state.landed = True
                    game_state.ascending = True  # Now go to CSM!
                    game_state.pad_multiplier = target_pad["mult"]

                    # Calculate landing score (not final yet - need to dock!)
                    total_fuel = descent_fuel + ascent_fuel
                    game_state.fuel_bonus = total_fuel * FUEL_BONUS_MULTIPLIER
                    game_state.landing_score = (
                        BASE_LANDING_SCORE * game_state.pad_multiplier
                    )
                    # Don't end game - continue to docking phase!

                # Crash detection - bad angle (too tilted)
                elif abs(angle_deg) > 30:
                    game_state.playing = False
                    game_state.crashed = True
                    game_state.score = 0

                # Wrong pad - only crash if significantly off target
                # Allow gentle landings on wrong pads (just don't get full points)
                elif not on_target_pad:
                    # Use same velocity threshold as target pad, but stricter angle
                    if abs(vel_y) < 1.0 and abs(vel_x) < 1.0 and abs(angle_deg) < 20:
                        # Gentle enough - treat as successful but with penalty
                        game_state.landed = True
                        game_state.ascending = True
                        game_state.pad_multiplier = 0.5  # Half points for wrong pad
                        total_fuel = descent_fuel + ascent_fuel
                        game_state.fuel_bonus = total_fuel * FUEL_BONUS_MULTIPLIER
                        game_state.landing_score = (
                            BASE_LANDING_SCORE * game_state.pad_multiplier
                        )
                    else:
                        # Too rough for wrong pad
                        game_state.playing = False
                        game_state.crashed = True
                        game_state.score = 0

            # Docking detection (check if undocked OR ascending after landing, but NOT after jettison)
            if (game_state.undocked or game_state.ascending) and ascent_freed and not game_state.docked and not game_state.jettisoned:
                # Check if docking conditions met: distance and orientation (±45°)
                if csm.is_docked(active_body):
                    game_state.docked = True
                    game_state.mission_complete = True

                    # When docked, combined Apollo (CSM + ascent) no longer obeys gravity
                    csm.body.gravityScale = 0.0
                    lander.ascent_stage.gravityScale = 0.0

                    # Final score = landing + docking + fuel bonus (only if landed first)
                    if game_state.landed:
                        game_state.docking_score = DOCKING_BONUS
                        game_state.score = (
                            game_state.landing_score +
                            game_state.docking_score +
                            game_state.fuel_bonus
                        )
                    else:
                        # Re-docked without landing - no score bonus
                        game_state.score = 0

            # If docked, synchronize ascent module position with CSM
            if game_state.docked:
                # Get CSM docking port position in world coordinates
                csm_scale = 1.125  # Reduced by 25% from 1.5
                sm_half_h = 2.0 * csm_scale
                cm_length = 1.6 * csm_scale
                dock_offset = 0.3 * csm_scale * 0.7

                # CSM is at 90° (pointing left), so docking port is at the left tip
                # In CSM local coords: (0, sm_half_h + cm_length + dock_offset)
                # After 90° rotation: this becomes (-distance, 0) in world coords
                docking_port_offset_x = -(sm_half_h + cm_length + dock_offset)
                csm_dock_world_x = csm.body.position.x + docking_port_offset_x
                csm_dock_world_y = csm.body.position.y

                # Ascent module at -90° (pointing right) has its docking port at the top
                # In lander local coords: (0, +h*1.1)
                # After -90° rotation: this becomes (+h*1.1, 0) in world coords
                lander_h = 1.5 * 0.5625  # Reduced by 25% from 0.75
                lander_dock_offset_x = lander_h * 1.1  # Pointing right, so offset is positive X

                # Position ascent module so its docking port aligns with CSM's docking port
                lander.ascent_stage.position = b2Vec2(
                    csm_dock_world_x - lander_dock_offset_x,
                    csm_dock_world_y
                )
                # Keep ascent module at -90° (pointing right) - don't change its orientation
                lander.ascent_stage.angle = -math.pi / 2.0
                lander.ascent_stage.linearVelocity = csm.body.linearVelocity
                lander.ascent_stage.angularVelocity = 0.0

                # Descent stage remains wherever it is (on surface or wherever it was left)

            # Crash detection for high-speed impacts (lander only, before jettison)
            elif not game_state.jettisoned and is_on_ground and (abs(vel_y) > 5.0 or abs(angle_deg) > 45):
                game_state.playing = False
                game_state.crashed = True
                game_state.score = 0

            # CSM crash detection (after jettison, check if CSM hits terrain)
            if game_state.jettisoned:
                # Check if CSM is near or below terrain level
                csm_y = csm.body.position.y
                csm_x = csm.body.position.x
                terrain_height = get_terrain_height_at(csm_x, terrain_pts)
                if terrain_height is not None and csm_y <= terrain_height + 5.0:  # 5m clearance
                    game_state.playing = False
                    game_state.crashed = True
                    game_state.score = 0

            # Out of bounds check (allow margin for lander width/height)
            # Lander is about 3-4m wide, so allow 5m margin on sides
            # Allow falling far below terrain (-50m) but not too high above world
            # EXCEPT: Allow wrapping if at CSM altitude or docked
            # After jettison, CSM needs to go higher than max_world_height to reach escape altitude
            at_csm_altitude = abs(pos_y - csm_altitude) < 10.0
            if not (game_state.docked or at_csm_altitude or game_state.jettisoned):
                if pos_x < -5 or pos_x > WORLD_WIDTH + 5 or pos_y < -50 or pos_y > max_world_height + 10:
                    game_state.playing = False
                    game_state.crashed = True
                    game_state.score = 0

            # Wrap lander horizontally when at CSM altitude or docked - seamless wrapping at WORLD_WIDTH
            if game_state.docked or at_csm_altitude:
                if pos_x > WORLD_WIDTH:
                    wrap_offset = pos_x - WORLD_WIDTH
                    new_wrapped_x = 0.0 + wrap_offset
                    active_body.position = b2Vec2(new_wrapped_x, active_body.position.y)
                elif pos_x < 0:
                    wrap_offset = 0 - pos_x
                    new_wrapped_x = WORLD_WIDTH - wrap_offset
                    active_body.position = b2Vec2(new_wrapped_x, active_body.position.y)

            # Check for escape (CSM crosses 2× orbital altitude after jettison)
            if game_state.jettisoned:
                escape_altitude = csm_altitude * 2.0
                if csm.body.position.y >= escape_altitude:
                    game_state.playing = False
                    game_state.mission_complete = True
                    game_state.escape_score = ESCAPE_BONUS
                    # Calculate final total score
                    game_state.score = (
                        game_state.undocking_score +
                        game_state.separation_score +
                        game_state.landing_score +
                        game_state.docking_score +
                        game_state.jettison_score +
                        game_state.escape_score +
                        game_state.fuel_bonus
                    )

        # ==========================================
        # Rendering
        # ==========================================
        screen.fill((0, 0, 0))

        # Draw stars (fixed in world space with cylindrical wrapping)
        # Generate stars in world coordinates so they stay fixed as camera moves
        import random as rnd
        rnd.seed(42)  # Fixed seed for consistent star pattern

        # Calculate visible world area
        screen_half_w = SCREEN_WIDTH / 2 / PPM
        screen_half_h = SCREEN_HEIGHT / 2 / PPM

        # For cylindrical wrapping, we need to draw stars in a repeating pattern
        # Draw stars for the current camera position and wrapped positions
        cam_x_normalized = cam_x % WORLD_WIDTH

        # Generate stars across one world width
        for i in range(300):
            # Star position in world coordinates (one panel)
            star_base_x = (i * 73.7) % WORLD_WIDTH
            star_world_y = (i * 97.3) % 120.0 - 10

            # Draw this star in multiple panels (left, center, right of camera)
            for panel_offset in [-WORLD_WIDTH, 0, WORLD_WIDTH]:
                star_world_x = star_base_x + panel_offset

                # Check if star is visible relative to camera
                if abs(star_world_x - cam_x) < screen_half_w + 10:
                    star_screen = world_to_screen(
                        b2Vec2(star_world_x, star_world_y),
                        cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y
                    )
                    brightness = int(((i * 67) % 156) + 100)
                    if 0 <= star_screen[0] < SCREEN_WIDTH and 0 <= star_screen[1] < SCREEN_HEIGHT:
                        pygame.draw.circle(screen, (brightness, brightness, brightness), star_screen, 1)

        # Draw terrain with cylindrical wrapping
        screen_half_w = SCREEN_WIDTH / 2 / PPM

        # Draw terrain in multiple panels for seamless wrapping
        for panel_offset in [-WORLD_WIDTH, 0, WORLD_WIDTH]:
            # Check if this panel is visible
            panel_center_x = WORLD_WIDTH / 2 + panel_offset
            if abs(panel_center_x - cam_x) < screen_half_w + WORLD_WIDTH / 2:
                # Offset terrain points for this panel
                offset_terrain_pts = [(pt[0] + panel_offset, pt[1]) for pt in terrain_pts]
                offset_pads_info = []
                for pad in pads_info:
                    offset_pad = pad.copy()
                    offset_pad["x1"] = pad["x1"] + panel_offset
                    offset_pad["x2"] = pad["x2"] + panel_offset
                    offset_pads_info.append(offset_pad)

                draw_terrain_pygame(
                    screen,
                    offset_terrain_pts,
                    offset_pads_info,
                    cam_x,
                    PPM,
                    SCREEN_WIDTH,
                    SCREEN_HEIGHT,
                    cam_y,
                )

        # Highlight target pad with pulsing green circles (only before landing) with wrapping
        if game_state.playing and not game_state.ascending:
            target_center_x = (target_pad["x1"] + target_pad["x2"]) / 2.0

            # Pulsing effect
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5
            radius1 = int(20 + pulse * 15)
            radius2 = int(35 + pulse * 15)

            # Draw in all visible panels
            for panel_offset in [-WORLD_WIDTH, 0, WORLD_WIDTH]:
                wrapped_x = target_center_x + panel_offset
                if abs(wrapped_x - cam_x) < screen_half_w + 10:
                    target_screen = world_to_screen(
                        b2Vec2(wrapped_x, target_pad["y"]),
                        cam_x,
                        SCREEN_WIDTH,
                        SCREEN_HEIGHT,
                        cam_y,
                    )
                    pygame.draw.circle(screen, (0, 255, 0), target_screen, radius1, 2)
                    pygame.draw.circle(screen, (0, 255, 0), target_screen, radius2, 1)

        # Draw CSM (always visible - it's in "orbit")
        # Draw with cylindrical wrapping - draw in all visible panels
        if csm:
            # Calculate which panels we need to draw
            screen_half_w = SCREEN_WIDTH / 2 / PPM
            csm_x = csm.body.position.x

            # Draw CSM in multiple panels if needed for wrapping
            for panel_offset in [-WORLD_WIDTH, 0, WORLD_WIDTH]:
                wrapped_x = csm_x + panel_offset
                # Only draw if this panel is visible
                if abs(wrapped_x - cam_x) < screen_half_w + 10:
                    temp_pos = csm.body.position
                    csm.body.position = b2Vec2(wrapped_x, csm.body.position.y)
                    # Pass thruster flags when jettisoned
                    if game_state.jettisoned:
                        csm.draw(screen, cam_x, cam_y, SCREEN_WIDTH, SCREEN_HEIGHT,
                                main_on=main_csm_on,
                                tl=csm_tl, tr=csm_tr, bl=csm_bl, br=csm_br,
                                sl=csm_sl, sr=csm_sr)
                    else:
                        csm.draw(screen, cam_x, cam_y, SCREEN_WIDTH, SCREEN_HEIGHT)
                    csm.body.position = temp_pos

            # Draw pulsing docking indicator when ascent module is separated (but not after jettison)
            if ascent_freed and game_state.ascending and not game_state.jettisoned:
                # Calculate CSM docking port position in world coordinates
                csm_scale = 1.125  # Reduced by 25% from 1.5
                sm_half_h = 2.0 * csm_scale
                cm_length = 1.6 * csm_scale
                dock_offset = 0.3 * csm_scale * 0.7
                docking_port_offset_x = -(sm_half_h + cm_length + dock_offset)
                csm_dock_world_x = csm.body.position.x + docking_port_offset_x
                csm_dock_world_y = csm.body.position.y

                # Convert to screen coordinates
                dock_screen = world_to_screen(
                    b2Vec2(csm_dock_world_x, csm_dock_world_y),
                    cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y
                )

                # Pulsing effect (similar to landing pad)
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5

                # Docking tolerance is 2.5 meters - convert to pixels
                docking_tolerance_pixels = 2.5 * PPM  # ~55 pixels
                radius1 = int(docking_tolerance_pixels * (0.8 + pulse * 0.2))
                radius2 = int(docking_tolerance_pixels * (1.0 + pulse * 0.2))

                # Draw pulsing circles in blue (docking color)
                pygame.draw.circle(screen, (100, 149, 237), dock_screen, radius1, 2)
                pygame.draw.circle(screen, (100, 149, 237), dock_screen, radius2, 1)

            # Draw pulsing indicator around ascent module's docking port when separated (but not after jettison)
            if ascent_freed and (game_state.ascending or game_state.undocked) and not game_state.jettisoned:
                # Calculate ascent module docking port position in world coordinates
                lander_h = 1.5 * lander.scale  # 1.5 * 0.75
                lander_dock_local = b2Vec2(0.0, lander_h * 1.1)

                # Convert from local to world coordinates
                lander_angle = lander.ascent_stage.angle
                cos_a = math.cos(lander_angle)
                sin_a = math.sin(lander_angle)

                # Rotate local coordinates to world coordinates
                lander_dock_world_x = (lander.ascent_stage.position.x +
                                      lander_dock_local.x * cos_a -
                                      lander_dock_local.y * sin_a)
                lander_dock_world_y = (lander.ascent_stage.position.y +
                                      lander_dock_local.x * sin_a +
                                      lander_dock_local.y * cos_a)

                # Convert to screen coordinates
                lander_dock_screen = world_to_screen(
                    b2Vec2(lander_dock_world_x, lander_dock_world_y),
                    cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y
                )

                # Same pulsing effect as CSM indicator
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5

                # Same docking tolerance
                docking_tolerance_pixels = 2.5 * PPM
                radius1 = int(docking_tolerance_pixels * (0.8 + pulse * 0.2))
                radius2 = int(docking_tolerance_pixels * (1.0 + pulse * 0.2))

                # Draw pulsing circles in yellow (different from CSM blue)
                pygame.draw.circle(screen, (255, 255, 0), lander_dock_screen, radius1, 2)
                pygame.draw.circle(screen, (255, 255, 0), lander_dock_screen, radius2, 1)

        # Draw zero gravity zone boundaries (thin blue horizontal lines at CSM altitude ±10m)
        # These lines are continuous across the entire screen (cylindrical world wrapping)
        gravity_tolerance = 10.0
        upper_gravity_line = csm_altitude + gravity_tolerance
        lower_gravity_line = csm_altitude - gravity_tolerance

        # Convert altitude to screen Y coordinate
        upper_y = world_to_screen(b2Vec2(cam_x, upper_gravity_line), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)[1]
        lower_y = world_to_screen(b2Vec2(cam_x, lower_gravity_line), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)[1]

        # Draw continuous horizontal lines across entire screen width
        pygame.draw.line(screen, (100, 149, 237), (0, upper_y), (SCREEN_WIDTH, upper_y), 1)  # Upper boundary
        pygame.draw.line(screen, (100, 149, 237), (0, lower_y), (SCREEN_WIDTH, lower_y), 1)  # Lower boundary

        # Draw escape altitude line at 2× CSM altitude (green line for mission success after jettison)
        escape_altitude = csm_altitude * 2.0
        escape_y = world_to_screen(b2Vec2(cam_x, escape_altitude), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)[1]
        pygame.draw.line(screen, (100, 255, 100), (0, escape_y), (SCREEN_WIDTH, escape_y), 2)  # Escape line (green)

        # Draw vertical boundary lines at world edges (faint, extending from terrain to top)
        # Left edge (x=0)
        left_edge_screen = world_to_screen(b2Vec2(0, 0), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
        left_edge_top_screen = world_to_screen(b2Vec2(0, max_world_height), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
        # Right edge (x=WORLD_WIDTH)
        right_edge_screen = world_to_screen(b2Vec2(WORLD_WIDTH, 0), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
        right_edge_top_screen = world_to_screen(b2Vec2(WORLD_WIDTH, max_world_height), cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)

        # Draw faint vertical lines
        pygame.draw.line(screen, (80, 80, 100), left_edge_screen, left_edge_top_screen, 1)
        pygame.draw.line(screen, (80, 80, 100), right_edge_screen, right_edge_top_screen, 1)

        # Draw lander
        descent = lander.descent_stage
        ascent = lander.ascent_stage
        ascent_freed = lander.connection_joint is None
        active_body = ascent if ascent_freed else descent

        # Lander color based on state
        if not game_state.playing:
            if game_state.landed:
                ascent_color = (100, 255, 100)  # Green - success
                descent_color = (100, 255, 100)
            else:
                ascent_color = (255, 100, 100)  # Red - crashed
                descent_color = (255, 100, 100)
        else:
            ascent_color = (180, 180, 190)  # Gray
            descent_color = (212, 175, 55)  # Gold

        # Draw lander with cylindrical wrapping - draw in all visible panels
        screen_half_w = SCREEN_WIDTH / 2 / PPM
        lander_x = active_body.position.x

        # Draw lander in multiple panels if needed for wrapping
        for panel_offset in [-WORLD_WIDTH, 0, WORLD_WIDTH]:
            wrapped_ascent_x = ascent.position.x + panel_offset
            wrapped_descent_x = descent.position.x + panel_offset

            # Only draw if this panel is visible
            if abs(wrapped_ascent_x - cam_x) < screen_half_w + 10 or abs(wrapped_descent_x - cam_x) < screen_half_w + 10:
                temp_ascent_pos = ascent.position
                temp_descent_pos = descent.position

                ascent.position = b2Vec2(wrapped_ascent_x, ascent.position.y)
                descent.position = b2Vec2(wrapped_descent_x, descent.position.y)

                # Draw ascent FIRST so descent can cover the recessed nozzle
                draw_body(screen, ascent, ascent_color, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
                draw_rcs_pods(screen, ascent, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
                # Draw descent AFTER
                draw_body(screen, descent, descent_color, cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)

                ascent.position = temp_ascent_pos
                descent.position = temp_descent_pos

        # Thruster flames (only if playing)
        if game_state.playing:
            draw_thrusters(
                screen,
                descent,
                is_ascent=False,
                main_on=main_descent_on,
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
            )
            draw_thrusters(
                screen,
                ascent,
                is_ascent=True,
                main_on=main_ascent_on,
                tl=asc_tl,
                bl=asc_bl,
                tr=asc_tr,
                br=asc_br,
                sl=asc_sl,
                sr=asc_sr,
                gimbal_angle_deg=0.0,
                cam_x=cam_x,
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                cam_y=cam_y,
            )

        # ==========================================
        # HUD
        # ==========================================
        if game_state.jettisoned:
            mode = "CSM"
        else:
            mode = "ASCENT" if ascent_freed else "DESCENT"

        # Telemetry
        terrain_height = get_terrain_height_at(active_body.position.x, terrain_pts)
        hud.draw_telemetry(screen, active_body, terrain_height, mode)

        # Fuel gauges (moved to left side to avoid mini-map)
        hud.draw_fuel_gauge(
            screen,
            10,
            160,
            descent_fuel,
            MAX_DESCENT_FUEL,
            "DESC",
        )
        hud.draw_fuel_gauge(
            screen,
            50,
            160,
            ascent_fuel,
            MAX_ASCENT_FUEL,
            "ASC",
        )
        hud.draw_fuel_gauge(
            screen,
            90,
            160,
            csm_fuel,
            MAX_CSM_FUEL,
            "CSM",
        )

        # Navigation (show target pad before landing, CSM after landing)
        # Moved to left side to avoid mini-map
        if game_state.ascending:
            # Show navigation to CSM
            hud.draw_navigation(
                screen,
                active_body.position.x,
                csm.body.position.x,
                active_body.position.x,
                pos_x=100,
                pos_y=160,
            )
            # Also show distance to CSM
            dist_to_csm = (active_body.position - csm.body.position).length
            font = pygame.font.SysFont("Arial", 16, bold=True)
            csm_dist_text = font.render(f"CSM DISTANCE: {dist_to_csm:.1f}m", True, (100, 149, 237))
            screen.blit(csm_dist_text, (100, 130))
        else:
            # Show navigation to landing pad
            target_center_x = (target_pad["x1"] + target_pad["x2"]) / 2.0
            hud.draw_navigation(
                screen,
                active_body.position.x,
                target_center_x,
                active_body.position.x,
                pos_x=100,
                pos_y=160,
            )

        # Score (show current or final)
        if game_state.ascending and not game_state.mission_complete:
            # Show landing score + potential docking bonus
            current_score = game_state.landing_score + game_state.fuel_bonus
            hud.draw_score(screen, current_score)
            font = pygame.font.SysFont("Arial", 14)
            bonus_text = font.render(f"(+{DOCKING_BONUS} for docking)", True, (100, 255, 100))
            screen.blit(bonus_text, (SCREEN_WIDTH // 2 - 80, 40))
        else:
            hud.draw_score(screen, game_state.score)

        # Throttle/Gimbal (if in descent mode)
        if not ascent_freed and game_state.playing:
            hud.draw_throttle_indicator(screen, descent_throttle, descent_gimbal_deg)

        # ==========================================
        # Mini-map (top-right corner)
        # ==========================================
        minimap_width = 600  # Tripled from 200 for better proportions
        minimap_height = 120
        minimap_x = SCREEN_WIDTH - minimap_width - 10
        minimap_y = 10
        minimap_margin = 5

        # Draw background
        minimap_rect = pygame.Rect(minimap_x, minimap_y, minimap_width, minimap_height)
        pygame.draw.rect(screen, (0, 0, 0, 180), minimap_rect)
        pygame.draw.rect(screen, (100, 100, 100), minimap_rect, 2)

        # Calculate scale factors for mini-map
        # World is WORLD_WIDTH wide and extends to escape altitude (2× CSM altitude)
        escape_altitude = csm_altitude * 2.0
        world_height = escape_altitude  # Top of minimap = escape altitude
        map_inner_width = minimap_width - 2 * minimap_margin
        map_inner_height = minimap_height - 2 * minimap_margin
        scale_x = map_inner_width / WORLD_WIDTH
        scale_y = map_inner_height / world_height

        def world_to_minimap(world_x, world_y):
            """Convert world coordinates to minimap screen coordinates with cylindrical wrapping."""
            # Normalize X coordinate to [0, WORLD_WIDTH) for cylindrical world
            normalized_x = world_x % WORLD_WIDTH
            mx = minimap_x + minimap_margin + normalized_x * scale_x
            # Y is inverted (0 at bottom in world, at top in screen)
            my = minimap_y + minimap_height - minimap_margin - world_y * scale_y
            return int(mx), int(my)

        # Draw terrain on mini-map
        if len(terrain_pts) > 0:
            minimap_terrain_pts = [world_to_minimap(pt[0], pt[1]) for pt in terrain_pts]
            if len(minimap_terrain_pts) > 2:
                pygame.draw.lines(screen, (150, 150, 150), False, minimap_terrain_pts, 1)

        # Draw landing pads
        for i, pad in enumerate(pads_info):
            pad_x1, pad_y = pad["x1"], pad["y"]
            pad_x2 = pad["x2"]
            mp1 = world_to_minimap(pad_x1, pad_y)
            mp2 = world_to_minimap(pad_x2, pad_y)

            # Get pad difficulty color based on multiplier
            mult = pad["mult"]
            if mult >= 6:
                pad_color = (255, 100, 100)  # Red - very hard
            elif mult >= 5:
                pad_color = (255, 165, 0)    # Orange - hard
            elif mult >= 4:
                pad_color = (255, 215, 0)    # Gold - medium-hard
            elif mult >= 3:
                pad_color = (255, 255, 100)  # Yellow - medium
            else:
                pad_color = (100, 255, 100)  # Green - easy

            # Target pad gets pulsing indicator and thicker line
            if i == target_pad_index:
                thickness = 3  # Increased from 2 for better visibility

                # Add pulsing indicator for target pad (only before landing)
                if not game_state.landed:
                    pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5
                    pad_center_x = (pad_x1 + pad_x2) / 2.0
                    center_minimap = world_to_minimap(pad_center_x, pad_y)

                    # Draw pulsing circles around target pad
                    pulse_radius1 = int(3 + pulse * 3)
                    pulse_radius2 = int(5 + pulse * 3)
                    pygame.draw.circle(screen, pad_color, center_minimap, pulse_radius1, 1)
                    pygame.draw.circle(screen, pad_color, center_minimap, pulse_radius2, 1)
            else:
                thickness = 2  # Increased from 1 for better visibility of non-target pads

            # Draw the pad line
            pygame.draw.line(screen, pad_color, mp1, mp2, thickness)

            # Add small dots at pad endpoints for better visibility
            pygame.draw.circle(screen, pad_color, mp1, 2)
            pygame.draw.circle(screen, pad_color, mp2, 2)

        # Draw CSM on mini-map
        csm_minimap = world_to_minimap(csm.body.position.x, csm.body.position.y)
        pygame.draw.circle(screen, (100, 149, 237), csm_minimap, 3)  # Blue circle for CSM

        # Draw lander on mini-map
        lander_minimap = world_to_minimap(active_body.position.x, active_body.position.y)
        if game_state.playing:
            lander_color = (0, 255, 0)  # Green when playing
        elif game_state.mission_complete:
            lander_color = (100, 255, 100)  # Light green when complete
        else:
            lander_color = (255, 0, 0)  # Red when crashed
        pygame.draw.circle(screen, lander_color, lander_minimap, 3)

        # Label
        font_mini = pygame.font.SysFont("Arial", 10, bold=True)
        minimap_label = font_mini.render("MAP", True, (150, 150, 150))
        screen.blit(minimap_label, (minimap_x + 5, minimap_y + 3))

        # Controls help
        if game_state.playing:
            if not game_state.undocked and not game_state.jettisoned:
                # Show undocking instruction
                font = pygame.font.SysFont("Arial", 24, bold=True)
                undock_text = font.render("PRESS U TO UNDOCK", True, (255, 255, 100))
                undock_rect = undock_text.get_rect(centerx=SCREEN_WIDTH // 2, top=SCREEN_HEIGHT - 60)
                # Draw background
                bg_rect = undock_rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0, 200), bg_rect, 0)
                pygame.draw.rect(screen, (255, 255, 100), bg_rect, 2)
                screen.blit(undock_text, undock_rect)
            elif game_state.docked and not game_state.jettisoned:
                # Show jettison instruction when docked
                font = pygame.font.SysFont("Arial", 24, bold=True)
                jettison_text = font.render("PRESS J TO JETTISON ASCENT MODULE", True, (100, 255, 100))
                jettison_rect = jettison_text.get_rect(centerx=SCREEN_WIDTH // 2, top=SCREEN_HEIGHT - 60)
                # Draw background
                bg_rect = jettison_rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0, 200), bg_rect, 0)
                pygame.draw.rect(screen, (100, 255, 100), bg_rect, 2)
                screen.blit(jettison_text, jettison_rect)
            else:
                hud.draw_controls_help(screen, mode)

        # Game over screen
        if not game_state.playing:
            if game_state.mission_complete:
                # Mission complete - show docking success
                hud.draw_game_over(
                    screen,
                    True,  # success
                    game_state.score,
                    game_state.fuel_bonus,
                    game_state.pad_multiplier,
                )
                # Add "SUCCESSFUL MISSION" banner
                font = pygame.font.SysFont("Arial", 36, bold=True)
                banner = font.render("SUCCESSFUL MISSION!", True, (100, 255, 100))
                banner_rect = banner.get_rect(centerx=SCREEN_WIDTH // 2, top=SCREEN_HEIGHT // 2 - 200)
                screen.blit(banner, banner_rect)

                # Show breakdown
                font_small = pygame.font.SysFont("Arial", 18)
                y = SCREEN_HEIGHT // 2 - 100
                texts = [
                    f"Undocking: +{int(game_state.undocking_score)}",
                    f"Separation: +{int(game_state.separation_score)}",
                    f"Landing: +{int(game_state.landing_score)}",
                    f"Docking: +{int(game_state.docking_score)}",
                    f"Jettison: +{int(game_state.jettison_score)}",
                    f"Escape: +{int(game_state.escape_score)}",
                    f"Fuel Bonus: +{int(game_state.fuel_bonus)}",
                ]
                for text in texts:
                    rendered = font_small.render(text, True, (255, 255, 255))
                    rect = rendered.get_rect(centerx=SCREEN_WIDTH // 2, top=y)
                    screen.blit(rendered, rect)
                    y += 25
            else:
                # Regular game over (landed or crashed)
                hud.draw_game_over(
                    screen,
                    game_state.landed,
                    game_state.score,
                    game_state.fuel_bonus,
                    game_state.pad_multiplier,
                )

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
