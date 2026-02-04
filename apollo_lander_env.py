"""
Apollo Lander Gymnasium Environment
Custom environment wrapper for training AI on the Apollo lunar lander simulation.

State space (9 dimensions):
    0: x position relative to target (normalized)
    1: y position / altitude (normalized)
    2: x velocity (normalized)
    3: y velocity (normalized)
    4: angle (radians)
    5: angular velocity (rad/s)
    6: left leg contact (1 if touching ground)
    7: right leg contact (1 if touching ground)
    8: throttle level (0.1 to 1.0)

Action space (7 discrete actions) - Always-on throttleable engine:
    0: No operation (engine continues at current throttle)
    1: RCS_L - Rotate LEFT (use when tilted right, angle > 0)
    2: THR_UP - Increase throttle by 5%
    3: THR_DOWN - Decrease throttle by 5%
    4: RCS_R - Rotate RIGHT (use when tilted left, angle < 0)
    5: TRANSLATE_L - Lateral translation LEFT (both side thrusters)
    6: TRANSLATE_R - Lateral translation RIGHT (both side thrusters)

The main engine fires EVERY step at the current throttle level (like the real
Lunar Module). The agent controls throttle up/down rather than pulsing the
engine on/off. This ensures continuous thrust for descent arrest while the
agent focuses on throttle modulation and attitude control.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from Box2D import b2World, b2Vec2, b2ContactListener

from apollolander import ApolloLander, PPM, MASS_SCALE
from apollo_terrain import ApolloTerrain, get_terrain_height_at, is_point_on_pad
from spring_rewards import SpringRewardComputer


class LanderContactListener(b2ContactListener):
    """Contact listener to detect leg touches on terrain."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        # Check if either fixture belongs to the lander
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body

        # Check for lander-terrain contact
        if self.env.lander:
            descent_body = self.env.lander.descent_stage
            if body_a == descent_body or body_b == descent_body:
                # Determine which side of the lander hit
                contact_point = contact.worldManifold.points[0] if contact.worldManifold.points else None
                if contact_point:
                    lander_x = descent_body.position.x
                    if contact_point[0] < lander_x:
                        self.env.left_leg_contact = True
                    else:
                        self.env.right_leg_contact = True

    def EndContact(self, contact):
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body

        if self.env.lander:
            descent_body = self.env.lander.descent_stage
            if body_a == descent_body or body_b == descent_body:
                contact_point = contact.worldManifold.points[0] if contact.worldManifold.points else None
                if contact_point:
                    lander_x = descent_body.position.x
                    if contact_point[0] < lander_x:
                        self.env.left_leg_contact = False
                    else:
                        self.env.right_leg_contact = False


class ApolloLanderEnv(gym.Env):
    """
    Gymnasium environment for the Apollo lunar lander.

    Provides realistic physics-based simulation for training
    AI agents to land on the Moon.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, terrain_roughness=1.0, spawn_altitude=50.0,
                 world_width=None, gravity=-1.62, max_episode_steps=2000,
                 num_actions=7,
                 spring_k_rotation=20.0, spring_k_horizontal=0.0, spring_k_vertical=0.0,
                 spring_c_angular=30.0, spring_c_descent=3.0, spring_c_ascend=5.0,
                 spring_c_horizontal_vel=0.0, descent_max_rate=2.5,
                 spring_proximity_gain=0.0,
                 spring_circle_radius=5.0, spring_alpha_deg=10.0):
        """
        Initialize the Apollo lander environment.

        Args:
            render_mode: "human" for pygame display, "rgb_array" for image output, None for headless
            terrain_roughness: Terrain bumpiness (higher = rougher)
            spawn_altitude: Starting altitude in game meters (= km real)
            world_width: Width of the cylindrical world (auto-calculated if None)
            gravity: Lunar gravity (default -1.62 m/s²)
            max_episode_steps: Maximum steps per episode before truncation
            num_actions: Number of discrete actions (5 = no lateral, 7 = with lateral RCS)
            spring_k_rotation: Spring constant for rotational stabilization
            spring_k_horizontal: Spring constant for horizontal centering toward pad
            spring_k_vertical: Spring constant for vertical descent (altitude)
            spring_c_angular: Damping coefficient for angular velocity
            spring_c_descent: Descent profile tracking coefficient
            spring_c_ascend: Penalty for ascending (positive vel_y)
            spring_c_horizontal_vel: Damping for horizontal velocity
            descent_max_rate: Target descent rate at spawn altitude (m/s)
            spring_proximity_gain: Extra vertical spring tightening near surface
            spring_circle_radius: Radius of imaginary circle for rotation spring anchors
            spring_alpha_deg: Angle offset (degrees) for top springs from vertical
        """
        super().__init__()

        self.render_mode = render_mode
        self.terrain_roughness = terrain_roughness
        self.spawn_altitude = spawn_altitude
        self.gravity = gravity
        self.max_episode_steps = max_episode_steps

        # World dimensions - match the game exactly
        # Game uses: SCREEN_WIDTH_METERS * max(3, int(WORLD_SCREENS))
        # where SCREEN_WIDTH_METERS = SCREEN_WIDTH / PPM = 1600 / 22 ≈ 72.7
        # and WORLD_SCREENS = PLANET_DIAMETER / SCREEN_WIDTH_METERS
        if world_width is None:
            # Match game calculation for Luna
            screen_width_meters = 1600.0 / PPM  # ~72.7 meters
            planet_diameter_meters = 3475.0  # Luna: 3475 km -> 3475 game meters at 1:1000 scale
            world_screens = planet_diameter_meters / screen_width_meters
            self.world_width = screen_width_meters * max(3, int(world_screens))  # ~3417 meters
        else:
            self.world_width = world_width

        # Action space: 5 or 7 discrete actions (always-on throttleable engine)
        # 0: No-op, 1: RCS_L, 2: THR_UP, 3: THR_DOWN, 4: RCS_R
        # 5: TRANSLATE_L, 6: TRANSLATE_R (only if num_actions=7)
        self.num_actions = num_actions
        self.action_space = spaces.Discrete(num_actions)

        # Observation space: 9 continuous values
        # All normalized to roughly [-1, 1] range for neural network
        # Includes throttle level so agent knows current engine state
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, -1.5, -np.pi, -5.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, np.pi, 5.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Physics world
        self.world = None
        self.lander = None
        self.terrain_body = None
        self.terrain_pts = []
        self.pads_info = []
        self.target_pad = None
        self.target_pad_index = 0

        # Contact detection
        self.left_leg_contact = False
        self.right_leg_contact = False

        # Fuel tracking
        self.fuel = 328.0  # Quadrupled fuel like in game
        self.max_fuel = 328.0

        # Throttle control (always-on engine with variable throttle)
        self.throttle = 0.4  # Start at 40% (just above ~37% hover point)
        self.min_throttle = 0.1  # 10% minimum (like real LM DPS)
        self.max_throttle = 1.0
        self.throttle_step = 0.05  # 5% per action for fine control

        # Episode tracking
        self.steps = 0
        self.prev_shaping = None

        # Thrust parameters
        self.descent_thrust_factor = abs(gravity) * 2.71
        # RCS thrust increased 3x for faster stabilization
        self.rcs_thrust = 15.0 * 445.0 / MASS_SCALE  # 15x RCS for meaningful control authority

        # Thruster state tracking for visualization
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_left_side = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_right_side = False

        # Spring-based reward shaping via shared SpringRewardComputer
        self.spring_computer = SpringRewardComputer(
            k_rotation=spring_k_rotation,
            k_horizontal=spring_k_horizontal,
            k_vertical=spring_k_vertical,
            c_angular=spring_c_angular,
            c_descent=spring_c_descent,
            c_ascend=spring_c_ascend,
            c_horizontal_vel=spring_c_horizontal_vel,
            descent_max_rate=descent_max_rate,
            proximity_gain=spring_proximity_gain,
            circle_radius=spring_circle_radius,
            alpha_deg=spring_alpha_deg,
            spawn_altitude=spawn_altitude,
        )

        # Rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.screen_width = 1200
        self.screen_height = 800

        # Continuous angle tracking (initialized in reset)
        self.prev_raw_angle = 0.0
        self.cumulative_angle = 0.0

        # Episode counter for HUD display
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Create physics world
        self.world = b2World(gravity=(0, self.gravity), doSleep=True)

        # Set up contact listener
        self.contact_listener = LanderContactListener(self)
        self.world.contactListener = self.contact_listener

        # Generate terrain
        terrain_gen = ApolloTerrain(world_width_meters=self.world_width, roughness=self.terrain_roughness)
        self.terrain_body, self.terrain_pts, self.pads_info = terrain_gen.generate_terrain(self.world)

        # Select random target pad
        self.target_pad_index = self.np_random.integers(0, len(self.pads_info))
        self.target_pad = self.pads_info[self.target_pad_index]
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0

        # Spawn lander with random offset from center over the pad
        # Offset is +/- 5 to 20 meters from center (never directly over pad)
        offset_magnitude = self.np_random.uniform(5.0, 20.0)
        offset_sign = self.np_random.choice([-1, 1])
        spawn_offset = offset_magnitude * offset_sign
        spawn_x = target_x + spawn_offset
        spawn_y = self.spawn_altitude

        self.lander = ApolloLander(self.world, position=b2Vec2(spawn_x, spawn_y), scale=0.75)

        # Apply atmospheric damping (Luna has no atmosphere, so 0.0)
        # This matches the game's behavior exactly
        self.lander.descent_stage.linearDamping = 0.0
        self.lander.descent_stage.angularDamping = 0.0
        self.lander.ascent_stage.linearDamping = 0.0
        self.lander.ascent_stage.angularDamping = 0.0

        # Apply initial velocity (slight descent) - matches game's random initial state
        initial_vx = self.np_random.uniform(-3.0, 3.0)
        initial_vy = self.np_random.uniform(-5.0, -2.0)
        self.lander.descent_stage.linearVelocity = b2Vec2(initial_vx, initial_vy)
        self.lander.ascent_stage.linearVelocity = b2Vec2(initial_vx, initial_vy)

        # Apply random initial rotation (+/- 90 degrees = +/- pi/2 radians)
        # This forces the agent to learn to stabilize first before maneuvering
        initial_angle = self.np_random.uniform(-np.pi/2, np.pi/2)
        self.lander.descent_stage.angle = initial_angle
        self.lander.ascent_stage.angle = initial_angle

        # Also add some initial angular velocity to make it more challenging
        initial_angular_vel = self.np_random.uniform(-0.5, 0.5)
        self.lander.descent_stage.angularVelocity = initial_angular_vel
        self.lander.ascent_stage.angularVelocity = initial_angular_vel

        # Reset state
        self.fuel = self.max_fuel
        self.throttle = 0.4  # Reset to 40% (just above hover)
        self.steps = 0
        self.episode_count += 1  # Increment episode counter
        self.prev_shaping = None
        self.left_leg_contact = False
        self.right_leg_contact = False

        # Track cumulative angle for continuous observation
        # This prevents discontinuity when crossing 0 degrees
        self.prev_raw_angle = initial_angle
        self.cumulative_angle = -initial_angle  # Negate for intuitive convention (+ = right)

        # Reset thruster states
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_left_side = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_right_side = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Integer action (0-3)

        Returns:
            observation: New state
            reward: Step reward
            terminated: Whether episode ended (landed/crashed)
            truncated: Whether episode was cut short (max steps/out of fuel)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Apply action
        self._apply_action(action)

        # Step physics
        self.world.Step(1.0/60.0, 6, 2)
        self.steps += 1

        # Get new state
        observation = self._get_observation()

        # Check termination conditions
        terminated, truncated, landing_status = self._check_termination()

        # Calculate reward
        reward = self._calculate_reward(landing_status, terminated, truncated)

        info = self._get_info()
        info['landing_status'] = landing_status

        # Debug: Log early terminations (before max steps)
        if (terminated or truncated) and self.steps < self.max_episode_steps - 10:
            pos = self.lander.descent_stage.position if self.lander else None
            vel = self.lander.descent_stage.linearVelocity if self.lander else None
            angle_deg = math.degrees(self.cumulative_angle)
            terrain_h = get_terrain_height_at(pos.x, self.terrain_pts) if pos else 0
            altitude = pos.y - terrain_h if pos else 0
            print(f"  [TERM] Ep {self.episode_count} Step {self.steps}: {landing_status} | "
                  f"Alt={altitude:.1f}m Vel=({vel.x:.1f},{vel.y:.1f}) Angle={angle_deg:.1f}° "
                  f"Legs=({self.left_leg_contact},{self.right_leg_contact}) Fuel={self.fuel:.0f}")

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        """Apply the given action to the lander.

        Engine is ALWAYS ON at the current throttle level. The agent controls
        attitude (RCS) and throttle level (THR_UP/THR_DOWN), not whether the
        engine fires. This mirrors the real Lunar Module descent.
        """
        # Reset thruster visualization states
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_left_side = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_right_side = False

        if self.lander is None:
            return

        # Get lander properties
        descent = self.lander.descent_stage
        ascent = self.lander.ascent_stage
        total_mass = descent.mass + ascent.mass
        lander_angle = descent.angle

        # Get RCS pod positions
        scale = 0.75
        rcs_offset_x = 2.1 * scale
        rcs_center_y = 0.0

        left_pod_pos = b2Vec2(-rcs_offset_x, rcs_center_y)
        right_pod_pos = b2Vec2(rcs_offset_x, rcs_center_y)
        left_pod_world = ascent.GetWorldPoint(left_pod_pos)
        right_pod_world = ascent.GetWorldPoint(right_pod_pos)

        # --- Process discrete action (RCS or throttle adjustment) ---
        if action == 0:
            # No-op: engine continues at current throttle, no RCS
            pass

        elif action == 1:
            # RCS_L: Rotate LEFT (CCW in world view) - makes angle MORE NEGATIVE
            # Use this when tilted RIGHT (angle > 0) to correct back to vertical
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, -1.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, left_pod_world, True)
                self.rcs_left_down = True

                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, 1.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, right_pod_world, True)
                self.rcs_right_up = True

                self.fuel = max(0.0, self.fuel - 0.02)

        elif action == 2:
            # THR_UP: Increase throttle (+5%)
            self.throttle = min(self.max_throttle, self.throttle + self.throttle_step)

        elif action == 3:
            # THR_DOWN: Decrease throttle (-5%)
            self.throttle = max(self.min_throttle, self.throttle - self.throttle_step)

        elif action == 4:
            # RCS_R: Rotate RIGHT (CW in world view) - makes angle MORE POSITIVE
            # Use this when tilted LEFT (angle < 0) to correct back to vertical
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, -1.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, right_pod_world, True)
                self.rcs_right_down = True

                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, 1.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, left_pod_world, True)
                self.rcs_left_up = True

                self.fuel = max(0.0, self.fuel - 0.02)

        elif action == 5 and self.num_actions >= 7:
            # TRANSLATE_L: Both side thrusters fire to push craft LEFT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(-1.0, 0.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, left_pod_world, True)
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, right_pod_world, True)
                self.rcs_left_side = True
                self.rcs_right_side = True
                self.fuel = max(0.0, self.fuel - 0.02)

        elif action == 6 and self.num_actions >= 7:
            # TRANSLATE_R: Both side thrusters fire to push craft RIGHT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(1.0, 0.0))
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, left_pod_world, True)
                ascent.ApplyForce(thrust_dir * self.rcs_thrust, right_pod_world, True)
                self.rcs_left_side = True
                self.rcs_right_side = True
                self.fuel = max(0.0, self.fuel - 0.02)

        # --- ALWAYS-ON ENGINE: fires every step at current throttle ---
        if self.fuel > 0:
            thrust_magnitude = self.descent_thrust_factor * total_mass * self.throttle

            thrust_x = -thrust_magnitude * math.sin(lander_angle)
            thrust_y = thrust_magnitude * math.cos(lander_angle)

            descent_pos = descent.worldCenter
            ascent_pos = ascent.worldCenter
            combined_com_x = (descent.mass * descent_pos.x + ascent.mass * ascent_pos.x) / total_mass
            combined_com_y = (descent.mass * descent_pos.y + ascent.mass * ascent_pos.y) / total_mass
            combined_com = b2Vec2(combined_com_x, combined_com_y)

            descent.ApplyForce((thrust_x, thrust_y), combined_com, True)
            self.main_engine_on = True

            self.fuel = max(0.0, self.fuel - 0.12 * self.throttle)

    def _get_observation(self):
        """Get the current observation/state."""
        if self.lander is None:
            return np.zeros(9, dtype=np.float32)

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        raw_angle = self.lander.descent_stage.angle
        angular_vel = self.lander.descent_stage.angularVelocity

        # CONTINUOUS ANGLE TRACKING: Instead of normalizing (which causes jumps at 0),
        # we track the cumulative angle change to maintain continuity.
        # This way, rotating from +5° through 0° to -5° is smooth, not a jump.
        #
        # Convention: positive = tilted RIGHT, negative = tilted LEFT
        # Box2D uses CCW as positive, so we negate.

        # Calculate angle delta (handling wrap-around)
        angle_delta = raw_angle - self.prev_raw_angle
        # Handle wrap-around: if delta is > pi, we wrapped the other way
        if angle_delta > math.pi:
            angle_delta -= 2 * math.pi
        elif angle_delta < -math.pi:
            angle_delta += 2 * math.pi

        # Update cumulative angle (negate because Box2D CCW is positive)
        self.cumulative_angle -= angle_delta
        self.prev_raw_angle = raw_angle

        # Use cumulative angle for observation
        angle = self.cumulative_angle

        # Get target position
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        target_y = self.target_pad["y"]

        # Calculate position relative to target (normalized)
        rel_x = (pos.x - target_x) / (self.world_width / 2.0)
        rel_y = (pos.y - target_y) / 50.0  # Normalize by spawn altitude

        # Normalize velocities
        vel_x = vel.x / 10.0
        vel_y = vel.y / 10.0

        # Clamp values
        rel_x = np.clip(rel_x, -1.5, 1.5)
        rel_y = np.clip(rel_y, -1.5, 1.5)
        vel_x = np.clip(vel_x, -1.5, 1.5)
        vel_y = np.clip(vel_y, -1.5, 1.5)
        angular_vel = np.clip(angular_vel, -5.0, 5.0)

        # IMPORTANT: Normalize angle to [-π, π] for the neural network observation
        # The cumulative tracking (self.cumulative_angle) prevents jumps during rotation,
        # but we normalize the OUTPUT to keep it in the expected range for the network.
        # This preserves continuity (no jumps) while keeping values bounded.
        normalized_angle = math.atan2(math.sin(angle), math.cos(angle))

        state = np.array([
            rel_x,
            rel_y,
            vel_x,
            vel_y,
            normalized_angle,  # Normalized to [-π, π] for neural network
            -angular_vel,  # Negate to match angle convention (+ = rotating right)
            1.0 if self.left_leg_contact else 0.0,
            1.0 if self.right_leg_contact else 0.0,
            self.throttle,  # Current throttle level (0.1 to 1.0)
        ], dtype=np.float32)

        return state

    def _check_termination(self):
        """
        Check if episode should end.

        Returns:
            terminated: True if landed or crashed
            truncated: True if max steps reached or out of fuel
            landing_status: 'landed', 'crashed', 'flying', or 'out_of_fuel'
        """
        if self.lander is None:
            return True, False, 'crashed'

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        # Normalize angle and negate for intuitive convention
        # (termination uses abs(angle_deg) so sign doesn't matter, but keep consistent)
        raw_angle = self.lander.descent_stage.angle
        normalized_angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))
        angle_deg = -math.degrees(normalized_angle)

        # Get terrain height at lander position
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)
        altitude = pos.y - terrain_height

        # Check leg contact - this is the primary landing detection
        # Legs are about 1.8m below the main body (2.4 * 0.75 scale)
        legs_touching = self.left_leg_contact or self.right_leg_contact
        both_legs_touching = self.left_leg_contact and self.right_leg_contact

        # Very close to ground (foot pads are about 0.12m thick at scale 0.75)
        # Use a smaller threshold that accounts for leg geometry
        is_very_close = altitude < 2.0

        # Landing/crash detection based on leg contact OR very close proximity
        if legs_touching or (is_very_close and abs(vel.y) < 0.5):
            # Check landing conditions
            speed = math.sqrt(vel.x**2 + vel.y**2)

            if speed < 2.0 and abs(angle_deg) < 20:
                # Soft landing - good!
                return True, False, 'landed'
            elif speed > 4.0 or abs(angle_deg) > 35:
                # Hard crash - too fast or too tilted
                return True, False, 'crashed'
            elif speed < 3.0 and abs(angle_deg) < 30:
                # Marginal but acceptable landing
                return True, False, 'landed'
            else:
                # Crash - bad angle or speed
                return True, False, 'crashed'

        # Check for out of bounds (fell off world)
        if pos.y < -10:
            return True, False, 'crashed'

        # Truncation conditions
        if self.steps >= self.max_episode_steps:
            return False, True, 'flying'

        if self.fuel <= 0:
            return False, True, 'out_of_fuel'

        return False, False, 'flying'

    def _compute_spring_energy(self):
        """Compute total spring potential energy via shared SpringRewardComputer."""
        if self.lander is None:
            return 0.0

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        angle = self.lander.descent_stage.angle
        angular_vel = self.lander.descent_stage.angularVelocity

        pad_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        pad_y = self.target_pad["y"]
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts) if self.terrain_pts else 0.0

        return self.spring_computer.compute_energy(
            lander_x=pos.x, lander_y=pos.y,
            angle=angle, angular_vel=angular_vel,
            pad_x=pad_x, pad_y=pad_y,
            vel_y=vel.y, vel_x=vel.x, terrain_height=terrain_height
        )

    def _calculate_reward(self, landing_status, terminated, truncated):
        """Calculate reward via shared SpringRewardComputer."""
        if self.lander is None:
            return -100.0

        E_new = self._compute_spring_energy()

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity

        on_target = False
        pad_mult = 1
        if landing_status == 'landed':
            on_target = is_point_on_pad(pos.x, pos.y, self.target_pad,
                                        tolerance_x=2.5, tolerance_y=3.0)
            pad_mult = self.target_pad.get("mult", 1)

        rcs_active = (self.rcs_left_up or self.rcs_left_down or
                      self.rcs_right_up or self.rcs_right_down or
                      self.rcs_left_side or self.rcs_right_side)

        reward = self.spring_computer.compute_reward(
            energy_old=self.prev_shaping,
            energy_new=E_new,
            rcs_active=rcs_active,
            cumulative_angle=self.cumulative_angle,
            landing_status=landing_status,
            fuel=self.fuel,
            max_fuel=self.max_fuel,
            on_target=on_target,
            pad_mult=pad_mult,
            vel_x=vel.x,
            vel_y=vel.y,
        )

        self.prev_shaping = E_new
        return reward

    def _get_info(self):
        """Get additional information about current state."""
        if self.lander is None:
            return {}

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        angle_deg = math.degrees(self.lander.descent_stage.angle)

        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)

        return {
            'altitude': pos.y - terrain_height,
            'velocity_x': vel.x,
            'velocity_y': vel.y,
            'angle': angle_deg,
            'fuel': self.fuel,
            'fuel_percent': 100.0 * self.fuel / self.max_fuel,
            'steps': self.steps,
            'on_target_pad': is_point_on_pad(pos.x, pos.y, self.target_pad, tolerance_x=2.5, tolerance_y=3.0),
            'left_leg_contact': self.left_leg_contact,
            'right_leg_contact': self.right_leg_contact,
        }

    def render(self):
        """Render the environment using the same visualization as the game."""
        if self.render_mode is None:
            return None

        # Lazy import pygame and rendering functions
        try:
            import pygame
            from apollo_terrain import draw_terrain_pygame
            from apollolander import draw_body, draw_rcs_pods, draw_thrusters
        except ImportError:
            return None

        # Initialize pygame and create screen
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 36)

            if self.render_mode == "human":
                pygame.display.set_caption("Apollo Lander - AI Training")
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # Handle pygame events to prevent window from freezing
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Clear screen with space background
        self.screen.fill((0, 0, 20))

        if self.lander:
            # Calculate camera position (follow lander)
            cam_x = self.lander.ascent_stage.position.x
            cam_y = self.lander.ascent_stage.position.y

            # Draw terrain with landing pads
            draw_terrain_pygame(
                self.screen, self.terrain_pts, self.pads_info,
                cam_x, PPM, self.screen_width, self.screen_height,
                cam_y, self.world_width
            )

            # Draw lander components (same as game)
            ascent_color = (200, 200, 200)  # Light gray for ascent
            descent_color = (180, 150, 100)  # Gold for descent

            # Draw ascent stage first
            draw_body(self.screen, self.lander.ascent_stage, ascent_color,
                     cam_x, self.screen_width, self.screen_height, cam_y)
            draw_rcs_pods(self.screen, self.lander.ascent_stage,
                         cam_x, self.screen_width, self.screen_height, cam_y)

            # Draw descent stage
            draw_body(self.screen, self.lander.descent_stage, descent_color,
                     cam_x, self.screen_width, self.screen_height, cam_y)

            # Draw main engine flame
            draw_thrusters(
                self.screen,
                self.lander.descent_stage,
                is_ascent=False,
                main_on=self.main_engine_on,
                tl=False, bl=False, tr=False, br=False, sl=False, sr=False,
                gimbal_angle_deg=0.0,
                cam_x=cam_x,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                cam_y=cam_y,
                throttle=self.throttle if self.main_engine_on else 0.0,
            )

            # Draw RCS thruster flames
            draw_thrusters(
                self.screen,
                self.lander.ascent_stage,
                is_ascent=True,
                main_on=False,
                tl=self.rcs_left_up,
                bl=self.rcs_left_down,
                tr=self.rcs_right_up,
                br=self.rcs_right_down,
                sl=self.rcs_left_side,
                sr=self.rcs_right_side,
                gimbal_angle_deg=0.0,
                cam_x=cam_x,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                cam_y=cam_y,
            )

            # Draw HUD overlay
            self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

        return None

    def _draw_hud(self):
        """Draw heads-up display with flight data."""
        import pygame

        if self.lander is None or self.font is None:
            return

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        # Use cumulative angle for continuous display (matches what agent sees)
        # positive = tilted RIGHT, negative = tilted LEFT
        angle_deg = math.degrees(self.cumulative_angle)
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)
        altitude = pos.y - terrain_height

        # HUD background
        hud_x, hud_y = 10, 10
        hud_width, hud_height = 280, 240
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 150))
        self.screen.blit(hud_surface, (hud_x, hud_y))

        # Draw HUD text
        white = (255, 255, 255)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        red = (255, 100, 100)
        cyan = (0, 255, 255)
        orange = (255, 165, 0)

        lines = [
            ("AI TRAINING MODE", yellow),
            (f"Episode: {self.episode_count}  Step: {self.steps}", white),
            ("", white),
            (f"Altitude: {altitude:.1f} m", white),
            (f"Vel X: {vel.x:+.1f} m/s", green if abs(vel.x) < 2 else red),
            (f"Vel Y: {vel.y:+.1f} m/s", green if abs(vel.y) < 2 else red),
            (f"Fuel: {100*self.fuel/self.max_fuel:.0f}%", green if self.fuel > 50 else red),
        ]

        y_offset = hud_y + 10
        for text, color in lines:
            if text:
                rendered = self.font.render(text, True, color)
                self.screen.blit(rendered, (hud_x + 10, y_offset))
            y_offset += 20

        # --- ANGLE INDICATOR ---
        y_offset += 5
        angle_color = green if abs(angle_deg) < 15 else (orange if abs(angle_deg) < 45 else red)
        angle_label = self.font.render("ANGLE:", True, white)
        self.screen.blit(angle_label, (hud_x + 10, y_offset))
        # Large numeric display for angle
        angle_value = self.font_large.render(f"{angle_deg:+.1f}°", True, angle_color)
        self.screen.blit(angle_value, (hud_x + 70, y_offset - 5))
        y_offset += 28

        # Visual angle bar: shows -90 to +90 degrees
        bar_x = hud_x + 10
        bar_width = 180
        bar_height = 16
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, y_offset, bar_width, bar_height))
        # Center line (vertical = 0 degrees)
        center_x = bar_x + bar_width // 2
        pygame.draw.line(self.screen, green, (center_x, y_offset), (center_x, y_offset + bar_height), 2)
        # Angle indicator (clamped to -90 to +90)
        clamped_angle = max(-90, min(90, angle_deg))
        indicator_x = center_x + int((clamped_angle / 90.0) * (bar_width // 2))
        indicator_color = green if abs(angle_deg) < 15 else (orange if abs(angle_deg) < 45 else red)
        pygame.draw.rect(self.screen, indicator_color, (indicator_x - 3, y_offset, 6, bar_height))
        # Labels
        left_label = self.font.render("-90", True, white)
        right_label = self.font.render("+90", True, white)
        self.screen.blit(left_label, (bar_x - 5, y_offset + bar_height + 2))
        self.screen.blit(right_label, (bar_x + bar_width - 20, y_offset + bar_height + 2))
        y_offset += bar_height + 22

        # --- THROTTLE INDICATOR ---
        throttle_pct = int(self.throttle * 100)
        throttle_label = self.font.render(f"THROTTLE: {throttle_pct}%", True, cyan)
        self.screen.blit(throttle_label, (hud_x + 10, y_offset))
        y_offset += 22

        # Visual throttle bar
        bar_x = hud_x + 10
        bar_width = 180
        bar_height = 14
        # Background
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, y_offset, bar_width, bar_height))
        # Filled portion
        fill_width = int(bar_width * self.throttle)
        throttle_color = cyan if self.throttle < 0.8 else orange
        pygame.draw.rect(self.screen, throttle_color, (bar_x, y_offset, fill_width, bar_height))

        # Draw target indicator
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        rel_x = target_x - pos.x
        if abs(rel_x) > self.screen_width / (2 * PPM):
            # Target is off-screen, draw arrow
            arrow_x = self.screen_width - 30 if rel_x > 0 else 30
            arrow_y = self.screen_height // 2
            arrow_text = "TARGET -->" if rel_x > 0 else "<-- TARGET"
            arrow_rendered = self.font.render(arrow_text, True, (255, 200, 0))
            self.screen.blit(arrow_rendered, (arrow_x - 40 if rel_x > 0 else arrow_x, arrow_y))

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.font_large = None


# Register the environment with Gymnasium
def register_apollo_env():
    """Register the Apollo lander environment with Gymnasium."""
    from gymnasium.envs.registration import register

    try:
        register(
            id='ApolloLander-v0',
            entry_point='apollo_lander_env:ApolloLanderEnv',
            max_episode_steps=1000,
        )
    except Exception:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    print("Testing Apollo Lander Environment...")

    env = ApolloLanderEnv(render_mode=None)

    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Test a few random steps
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"Landing status: {info.get('landing_status', 'unknown')}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")

    env.close()
    print("Test complete!")
