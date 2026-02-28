"""
In-Game Training for Apollo Lander (v2 — Lateral RCS + 2-Stage Curriculum)

Trains the AI agent using the game's exact Box2D physics (solver iterations 10,10)
rather than the gym environment's (6,2). This eliminates physics mismatches between
training and deployment.

2-Stage Curriculum:
  Stage 1: Stabilize + Hover (5 actions — rotation RCS only)
  Stage 2: Full Landing with lateral RCS (7 actions — adds TRANSLATE_L/R)

Usage:
    python train_in_game.py --seed 999                    # Headless (fast)
    python train_in_game.py --seed 999 --visualize        # Watch training live
    python train_in_game.py --seed 999 --stage 2          # Start from stage 2
    python train_in_game.py --seed 999 --episodes 2000    # Custom episode count
    python train_in_game.py --seed 999 --single-stage --stage 2
"""

import argparse
import numpy as np
import os
import json
import math
import random
from collections import deque
import time

from Box2D import b2World, b2Vec2, b2ContactListener

from apollolander import ApolloLander, PPM, MASS_SCALE, check_contact_probes
from apollo_terrain import ApolloTerrain, get_terrain_height_at, is_point_on_pad
from spring_rewards import SpringRewardComputer
from double_dqn_agent import DoubleDQNAgent


# Game-identical constants (from apollolandergame_with_ai.py, Luna)
GRAVITY = -1.62
SPAWN_ALTITUDE = 50.0
TERRAIN_ROUGHNESS = 1.0
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000
SCREEN_WIDTH_METERS = SCREEN_WIDTH / PPM
PLANET_DIAMETER_METERS = 3475.0  # Luna diameter in game meters
WORLD_SCREENS = PLANET_DIAMETER_METERS / SCREEN_WIDTH_METERS
WORLD_WIDTH = SCREEN_WIDTH_METERS * max(3, int(WORLD_SCREENS))

# Physics parameters matching the game exactly
SOLVER_VELOCITY_ITERATIONS = 10
SOLVER_POSITION_ITERATIONS = 10
TIME_STEP = 1.0 / 60.0

# Engine/RCS parameters (matching apollo_lander_env.py)
DESCENT_THRUST_FACTOR = abs(GRAVITY) * 2.71
RCS_THRUST = 15.0 * 445.0 / MASS_SCALE  # 66.75 force units
FUEL_MAX = 250.0
FUEL_MAIN_RATE = 0.12   # per step * throttle
FUEL_RCS_RATE = 0.02    # per RCS firing
THROTTLE_INIT = 0.6
THROTTLE_MIN = 0.1
THROTTLE_MAX = 1.0
THROTTLE_STEP = 0.05


class GameContactListener(b2ContactListener):
    """Contact listener to detect leg touches on terrain."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body
        if self.env.lander:
            descent_body = self.env.lander.descent_stage
            if body_a == descent_body or body_b == descent_body:
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


class GameTrainingEnv:
    """
    Training environment using game-identical Box2D physics.

    Key difference from ApolloLanderEnv: uses solver iterations (10, 10)
    matching the actual game, instead of (6, 2).
    """

    def __init__(self, render_mode=None, spring_computer=None, spawn_altitude=SPAWN_ALTITUDE,
                 num_actions=5, max_episode_steps=2000):
        self.render_mode = render_mode
        self.spawn_altitude = spawn_altitude
        self.spring_computer = spring_computer
        self.num_actions = num_actions
        self._max_episode_steps = max_episode_steps

        # Observation/action space dimensions (matches ApolloLanderEnv)
        self.state_size = 9
        self.action_size = num_actions

        # Gym-compatible space objects (for DoubleDQNAgent compatibility)
        class SimpleSpace:
            def __init__(self, shape=None, n=None):
                self.shape = shape
                self.n = n
            def contains(self, x):
                return True
            def sample(self):
                if self.n is not None:
                    return random.randint(0, self.n - 1)
                return None

        self.observation_space = SimpleSpace(shape=(self.state_size,))
        self.action_space = SimpleSpace(n=self.action_size)

        # Physics state
        self.world = None
        self.lander = None
        self.terrain_body = None
        self.terrain_pts = []
        self.pads_info = []
        self.target_pad = None

        # Contact detection
        self.left_leg_contact = False
        self.right_leg_contact = False
        self.contact_light = False  # Contact probes touched surface (cuts engine, like real Apollo)

        # Engine state
        self.fuel = FUEL_MAX
        self.throttle = THROTTLE_INIT

        # Thruster visualization state
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_left_side = False
        self.rcs_right_side = False

        # Episode tracking
        self.steps = 0
        self.max_episode_steps = self._max_episode_steps
        self.prev_energy = None

        # Continuous angle tracking
        self.prev_raw_angle = 0.0
        self.cumulative_angle = 0.0

        # Episode counter
        self.episode_count = 0

        # RNG
        self.np_random = np.random.default_rng()

        # Rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.font_large = None

    def reset(self, seed=None):
        """Reset environment to initial state. Returns (observation, info)."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Create physics world with GAME-IDENTICAL solver iterations
        self.world = b2World(gravity=(0, GRAVITY), doSleep=True)
        self.contact_listener = GameContactListener(self)
        self.world.contactListener = self.contact_listener

        # Generate terrain
        terrain_gen = ApolloTerrain(world_width_meters=WORLD_WIDTH, roughness=TERRAIN_ROUGHNESS)
        self.terrain_body, self.terrain_pts, self.pads_info = terrain_gen.generate_terrain(self.world)

        # Select random target pad
        self.target_pad_index = int(self.np_random.integers(0, len(self.pads_info)))
        self.target_pad = self.pads_info[self.target_pad_index]
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0

        # Spawn lander with random offset from pad center
        offset_magnitude = float(self.np_random.uniform(5.0, 20.0))
        offset_sign = self.np_random.choice([-1, 1])
        spawn_x = target_x + offset_magnitude * offset_sign
        spawn_y = self.spawn_altitude

        self.lander = ApolloLander(self.world, position=b2Vec2(spawn_x, spawn_y), scale=0.75)

        # Zero damping (Luna has no atmosphere)
        self.lander.descent_stage.linearDamping = 0.0
        self.lander.descent_stage.angularDamping = 0.0
        self.lander.ascent_stage.linearDamping = 0.0
        self.lander.ascent_stage.angularDamping = 0.0

        # Random initial velocity
        initial_vx = float(self.np_random.uniform(-3.0, 3.0))
        initial_vy = float(self.np_random.uniform(-5.0, -2.0))
        self.lander.descent_stage.linearVelocity = b2Vec2(initial_vx, initial_vy)
        self.lander.ascent_stage.linearVelocity = b2Vec2(initial_vx, initial_vy)

        # Random initial rotation
        initial_angle = float(self.np_random.uniform(-np.pi/2, np.pi/2))
        self.lander.descent_stage.angle = initial_angle
        self.lander.ascent_stage.angle = initial_angle

        # Random initial angular velocity
        initial_angular_vel = float(self.np_random.uniform(-0.5, 0.5))
        self.lander.descent_stage.angularVelocity = initial_angular_vel
        self.lander.ascent_stage.angularVelocity = initial_angular_vel

        # Reset state
        self.fuel = FUEL_MAX
        self.throttle = THROTTLE_INIT
        self.steps = 0
        self.episode_count += 1
        self.prev_energy = None
        self.left_leg_contact = False
        self.right_leg_contact = False
        self.contact_light = False

        # Angle tracking
        self.prev_raw_angle = initial_angle
        self.cumulative_angle = -initial_angle

        # Thruster states
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_left_side = False
        self.rcs_right_side = False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """Execute one physics step. Returns (obs, reward, terminated, truncated, info)."""
        # Apply action
        self._apply_action(action)

        # Step physics with GAME solver iterations (10, 10)
        self.world.Step(TIME_STEP, SOLVER_VELOCITY_ITERATIONS, SOLVER_POSITION_ITERATIONS)
        self.steps += 1

        # Contact probe check: cuts throttle to 0 on surface contact, like real Apollo
        if not self.contact_light and not (self.left_leg_contact or self.right_leg_contact):
            if check_contact_probes(self.lander.descent_stage,
                                    lambda x: get_terrain_height_at(x, self.terrain_pts),
                                    WORLD_WIDTH):
                self.contact_light = True
                self.throttle = 0.0

        obs = self._get_observation()
        terminated, truncated, landing_status = self._check_termination()

        # Compute reward via SpringRewardComputer
        reward = self._calculate_reward(landing_status, terminated, truncated)

        info = self._get_info()
        info['landing_status'] = landing_status

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        """Apply action to lander. Identical to ApolloLanderEnv._apply_action."""
        self.main_engine_on = False
        self.rcs_left_up = False
        self.rcs_left_down = False
        self.rcs_right_up = False
        self.rcs_right_down = False
        self.rcs_left_side = False
        self.rcs_right_side = False

        if self.lander is None:
            return

        descent = self.lander.descent_stage
        ascent = self.lander.ascent_stage
        total_mass = descent.mass + ascent.mass
        lander_angle = descent.angle

        # RCS pod positions
        scale = 0.75
        rcs_offset_x = 2.1 * scale
        rcs_center_y = 0.0
        left_pod_pos = b2Vec2(-rcs_offset_x, rcs_center_y)
        right_pod_pos = b2Vec2(rcs_offset_x, rcs_center_y)
        left_pod_world = ascent.GetWorldPoint(left_pod_pos)
        right_pod_world = ascent.GetWorldPoint(right_pod_pos)

        if action == 0:
            pass  # No-op

        elif action == 1:
            # RCS_L: Rotate LEFT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, -1.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, left_pod_world, True)
                self.rcs_left_down = True

                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, 1.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, right_pod_world, True)
                self.rcs_right_up = True

                self.fuel = max(0.0, self.fuel - FUEL_RCS_RATE)

        elif action == 2:
            # THR_UP
            self.throttle = min(THROTTLE_MAX, self.throttle + THROTTLE_STEP)

        elif action == 3:
            # THR_DOWN
            self.throttle = max(THROTTLE_MIN, self.throttle - THROTTLE_STEP)

        elif action == 4:
            # RCS_R: Rotate RIGHT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, -1.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, right_pod_world, True)
                self.rcs_right_down = True

                thrust_dir = ascent.GetWorldVector(b2Vec2(0.0, 1.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, left_pod_world, True)
                self.rcs_left_up = True

                self.fuel = max(0.0, self.fuel - FUEL_RCS_RATE)

        elif action == 5 and self.num_actions >= 7:
            # TRANSLATE_L: Both pods fire to push craft LEFT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(-1.0, 0.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, left_pod_world, True)
                ascent.ApplyForce(thrust_dir * RCS_THRUST, right_pod_world, True)
                self.rcs_left_side = True
                self.rcs_right_side = True
                self.fuel = max(0.0, self.fuel - FUEL_RCS_RATE)

        elif action == 6 and self.num_actions >= 7:
            # TRANSLATE_R: Both pods fire to push craft RIGHT
            if self.fuel > 0:
                thrust_dir = ascent.GetWorldVector(b2Vec2(1.0, 0.0))
                ascent.ApplyForce(thrust_dir * RCS_THRUST, left_pod_world, True)
                ascent.ApplyForce(thrust_dir * RCS_THRUST, right_pod_world, True)
                self.rcs_left_side = True
                self.rcs_right_side = True
                self.fuel = max(0.0, self.fuel - FUEL_RCS_RATE)

        # Always-on engine
        if self.fuel > 0:
            thrust_magnitude = DESCENT_THRUST_FACTOR * total_mass * self.throttle
            thrust_x = -thrust_magnitude * math.sin(lander_angle)
            thrust_y = thrust_magnitude * math.cos(lander_angle)

            descent_pos = descent.worldCenter
            ascent_pos = ascent.worldCenter
            combined_com_x = (descent.mass * descent_pos.x + ascent.mass * ascent_pos.x) / total_mass
            combined_com_y = (descent.mass * descent_pos.y + ascent.mass * ascent_pos.y) / total_mass
            combined_com = b2Vec2(combined_com_x, combined_com_y)

            descent.ApplyForce((thrust_x, thrust_y), combined_com, True)
            self.main_engine_on = True

            self.fuel = max(0.0, self.fuel - FUEL_MAIN_RATE * self.throttle)

    def _get_observation(self):
        """Get 9-dim observation. Identical to ApolloLanderEnv._get_observation."""
        if self.lander is None:
            return np.zeros(self.state_size, dtype=np.float32)

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        raw_angle = self.lander.descent_stage.angle
        angular_vel = self.lander.descent_stage.angularVelocity

        # Continuous angle tracking
        angle_delta = raw_angle - self.prev_raw_angle
        if angle_delta > math.pi:
            angle_delta -= 2 * math.pi
        elif angle_delta < -math.pi:
            angle_delta += 2 * math.pi
        self.cumulative_angle -= angle_delta
        self.prev_raw_angle = raw_angle

        angle = self.cumulative_angle
        normalized_angle = math.atan2(math.sin(angle), math.cos(angle))

        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        target_y = self.target_pad["y"]

        rel_x = np.clip((pos.x - target_x) / (WORLD_WIDTH / 2.0), -1.5, 1.5)
        rel_y = np.clip((pos.y - target_y) / 50.0, -1.5, 1.5)
        vel_x = np.clip(vel.x / 10.0, -1.5, 1.5)
        vel_y = np.clip(vel.y / 10.0, -1.5, 1.5)
        angular_vel = np.clip(angular_vel, -5.0, 5.0)

        return np.array([
            rel_x, rel_y, vel_x, vel_y, normalized_angle,
            -angular_vel,
            0.0,  # left leg contact: always 0.0 to match game (no contact listener in game)
            0.0,  # right leg contact: always 0.0 to match game
            self.throttle,
        ], dtype=np.float32)

    def _check_termination(self):
        """Check if episode should end. Mirrors apollolandergame_with_ai.py exactly."""
        if self.lander is None:
            return True, False, 'crashed'

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        raw_angle = self.lander.descent_stage.angle
        normalized_angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))
        angle_deg = -math.degrees(normalized_angle)

        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)
        altitude = pos.y - terrain_height

        on_target_pad = is_point_on_pad(pos.x, pos.y, self.target_pad, tolerance_x=2.5, tolerance_y=3.0)

        # Mirror game: ground detection at 3.5m (not 2.0m)
        is_on_ground = altitude < 3.5
        # Mirror game: nearly stopped requires both axes < 1.0 m/s
        is_nearly_stopped = abs(vel.y) < 1.0 and abs(vel.x) < 1.0

        if is_on_ground and is_nearly_stopped:
            # Mirror game: success = on target AND angle < 20°
            if on_target_pad and abs(angle_deg) < 20:
                return True, False, 'landed'
            # Mirror game: bad angle (> 30°) = crash
            elif abs(angle_deg) > 30:
                return True, False, 'crashed'
            # Mirror game: wrong pad but angle < 20° = landed (off-target)
            elif not on_target_pad and abs(angle_deg) < 20:
                return True, False, 'landed'
            # Mirror game: 20-30° angle range = crash
            else:
                return True, False, 'crashed'

        # Mirror game: high-speed ground impact = crash
        if is_on_ground and (abs(vel.y) > 3.0 or abs(vel.x) > 3.0):
            return True, False, 'crashed'

        if pos.y < -10:
            return True, False, 'crashed'

        if self.steps >= self.max_episode_steps:
            return False, True, 'flying'

        if self.fuel <= 0:
            return False, True, 'out_of_fuel'

        return False, False, 'flying'

    def _calculate_reward(self, landing_status, terminated, truncated):
        """Calculate reward using SpringRewardComputer."""
        if self.lander is None:
            return -100.0

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        angle = self.lander.descent_stage.angle
        angular_vel = self.lander.descent_stage.angularVelocity

        pad_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        pad_y = self.target_pad["y"]
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts) if self.terrain_pts else 0.0

        # Compute current energy
        E_new = self.spring_computer.compute_energy(
            lander_x=pos.x, lander_y=pos.y,
            angle=angle, angular_vel=angular_vel,
            pad_x=pad_x, pad_y=pad_y,
            vel_y=vel.y, vel_x=vel.x, terrain_height=terrain_height
        )

        # Determine terminal info
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
            energy_old=self.prev_energy,
            energy_new=E_new,
            rcs_active=rcs_active,
            cumulative_angle=self.cumulative_angle,
            landing_status=landing_status,
            fuel=self.fuel,
            max_fuel=FUEL_MAX,
            on_target=on_target,
            pad_mult=pad_mult,
            vel_x=vel.x,
            vel_y=vel.y,
        )

        self.prev_energy = E_new
        return reward

    def _get_info(self):
        """Get additional state information."""
        if self.lander is None:
            return {}

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)

        return {
            'altitude': pos.y - terrain_height,
            'velocity_x': vel.x,
            'velocity_y': vel.y,
            'angle': math.degrees(self.lander.descent_stage.angle),
            'fuel': self.fuel,
            'fuel_percent': 100.0 * self.fuel / FUEL_MAX,
            'steps': self.steps,
            'on_target_pad': is_point_on_pad(pos.x, pos.y, self.target_pad,
                                             tolerance_x=2.5, tolerance_y=3.0),
            'left_leg_contact': self.left_leg_contact,
            'right_leg_contact': self.right_leg_contact,
        }

    def render(self):
        """Render the environment using pygame (game-style display)."""
        if self.render_mode is None:
            return None

        try:
            import pygame
            from apollo_terrain import draw_terrain_pygame
            from apollolander import draw_body, draw_rcs_pods, draw_thrusters
        except ImportError:
            return None

        if self.screen is None:
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
            if not pygame.font.get_init():
                pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 36)

            if self.render_mode == "human":
                pygame.display.set_caption("Apollo Lander - In-Game Training")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self.screen.fill((0, 0, 20))

        if self.lander:
            cam_x = self.lander.ascent_stage.position.x
            cam_y = self.lander.ascent_stage.position.y

            draw_terrain_pygame(
                self.screen, self.terrain_pts, self.pads_info,
                cam_x, PPM, SCREEN_WIDTH, SCREEN_HEIGHT,
                cam_y, WORLD_WIDTH
            )

            ascent_color = (200, 200, 200)
            descent_color = (180, 150, 100)

            draw_body(self.screen, self.lander.ascent_stage, ascent_color,
                     cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
            draw_rcs_pods(self.screen, self.lander.ascent_stage,
                         cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)
            draw_body(self.screen, self.lander.descent_stage, descent_color,
                     cam_x, SCREEN_WIDTH, SCREEN_HEIGHT, cam_y)

            # Main engine flame
            draw_thrusters(
                self.screen, self.lander.descent_stage,
                is_ascent=False, main_on=self.main_engine_on,
                tl=False, bl=False, tr=False, br=False, sl=False, sr=False,
                gimbal_angle_deg=0.0, cam_x=cam_x,
                screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                cam_y=cam_y, throttle=self.throttle if self.main_engine_on else 0.0,
            )

            # RCS flames
            draw_thrusters(
                self.screen, self.lander.ascent_stage,
                is_ascent=True, main_on=False,
                tl=self.rcs_left_up, bl=self.rcs_left_down,
                tr=self.rcs_right_up, br=self.rcs_right_down,
                sl=self.rcs_left_side, sr=self.rcs_right_side,
                gimbal_angle_deg=0.0, cam_x=cam_x,
                screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                cam_y=cam_y,
            )

            self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)

        return None

    def _draw_hud(self):
        """Draw training HUD overlay."""
        import pygame

        if self.lander is None or self.font is None:
            return

        pos = self.lander.descent_stage.position
        vel = self.lander.descent_stage.linearVelocity
        angle_deg = math.degrees(self.cumulative_angle)
        terrain_height = get_terrain_height_at(pos.x, self.terrain_pts)
        altitude = pos.y - terrain_height

        # HUD background
        hud_x, hud_y = 10, 10
        hud_width, hud_height = 300, 260
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 150))
        self.screen.blit(hud_surface, (hud_x, hud_y))

        white = (255, 255, 255)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        red = (255, 100, 100)
        cyan = (0, 255, 255)
        orange = (255, 165, 0)

        lines = [
            ("IN-GAME TRAINING", yellow),
            (f"Episode: {self.episode_count}  Step: {self.steps}", white),
            ("", white),
            (f"Altitude: {altitude:.1f} m", white),
            (f"Vel X: {vel.x:+.1f} m/s", green if abs(vel.x) < 2 else red),
            (f"Vel Y: {vel.y:+.1f} m/s", green if abs(vel.y) < 2 else red),
            (f"Fuel: {100*self.fuel/FUEL_MAX:.0f}%", green if self.fuel > 50 else red),
        ]

        y_offset = hud_y + 10
        for text, color in lines:
            if text:
                rendered = self.font.render(text, True, color)
                self.screen.blit(rendered, (hud_x + 10, y_offset))
            y_offset += 20

        # Angle display
        y_offset += 5
        angle_color = green if abs(angle_deg) < 15 else (orange if abs(angle_deg) < 45 else red)
        angle_label = self.font.render("ANGLE:", True, white)
        self.screen.blit(angle_label, (hud_x + 10, y_offset))
        angle_value = self.font_large.render(f"{angle_deg:+.1f}", True, angle_color)
        self.screen.blit(angle_value, (hud_x + 80, y_offset - 5))
        y_offset += 28

        # Angle bar
        bar_x = hud_x + 10
        bar_width = 200
        bar_height = 14
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, y_offset, bar_width, bar_height))
        center_x = bar_x + bar_width // 2
        pygame.draw.line(self.screen, green, (center_x, y_offset), (center_x, y_offset + bar_height), 2)
        clamped_angle = max(-90, min(90, angle_deg))
        indicator_x = center_x + int((clamped_angle / 90.0) * (bar_width // 2))
        pygame.draw.rect(self.screen, angle_color, (indicator_x - 3, y_offset, 6, bar_height))
        y_offset += bar_height + 18

        # Throttle
        throttle_pct = int(self.throttle * 100)
        throttle_label = self.font.render(f"THROTTLE: {throttle_pct}%", True, cyan)
        self.screen.blit(throttle_label, (hud_x + 10, y_offset))
        y_offset += 22
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, y_offset, bar_width, bar_height))
        fill_width = int(bar_width * self.throttle)
        throttle_color = cyan if self.throttle < 0.8 else orange
        pygame.draw.rect(self.screen, throttle_color, (bar_x, y_offset, fill_width, bar_height))

        # Target indicator
        target_x = (self.target_pad["x1"] + self.target_pad["x2"]) / 2.0
        rel_x = target_x - pos.x
        if abs(rel_x) > SCREEN_WIDTH / (2 * PPM):
            arrow_x = SCREEN_WIDTH - 30 if rel_x > 0 else 30
            arrow_y = SCREEN_HEIGHT // 2
            arrow_text = "TARGET -->" if rel_x > 0 else "<-- TARGET"
            arrow_rendered = self.font.render(arrow_text, True, (255, 200, 0))
            self.screen.blit(arrow_rendered, (arrow_x - 40 if rel_x > 0 else arrow_x, arrow_y))

    def close(self):
        """Clean up rendering resources without tearing down pygame globally."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.font_large = None


# ============================================================================
# 2-Stage Curriculum wrapper (Lateral RCS redesign)
# ============================================================================

class CurriculumGameEnv:
    """2-stage curriculum wrapper around GameTrainingEnv.

    Stage 1: Stabilize + Hover (5 actions — rotation RCS only)
    Stage 2: Full Landing with lateral RCS (7 actions — adds TRANSLATE_L/R)
    """

    ACTION_NAMES_5 = {0: "NOOP", 1: "RCS_L", 2: "THR_UP", 3: "THR_DN", 4: "RCS_R"}
    ACTION_NAMES_7 = {0: "NOOP", 1: "RCS_L", 2: "THR_UP", 3: "THR_DN", 4: "RCS_R",
                      5: "TR_L", 6: "TR_R"}

    def __init__(self, stage=1, render_mode=None):
        # Spring configs per stage (new parameter names)
        spring_configs = {
            1: {'k_rotation': 25.0, 'k_horizontal': 0.0, 'k_vertical': 0.0,
                'c_angular': 40.0, 'c_descent': 2.0, 'c_ascend': 4.0,
                'c_horizontal_vel': 0.0, 'descent_max_rate': 0.0,
                'proximity_gain': 0.0},
            2: {'k_rotation': 20.0, 'k_horizontal': 3.0, 'k_vertical': 1.2,
                'c_angular': 30.0, 'c_descent': 3.0, 'c_ascend': 5.0,
                'c_horizontal_vel': 1.0, 'descent_max_rate': 2.5,
                'proximity_gain': 5.0},
            3: {'k_rotation': 20.0, 'k_horizontal': 3.0, 'k_vertical': 2.0,
                'c_angular': 30.0, 'c_descent': 5.0, 'c_ascend': 5.0,
                'c_horizontal_vel': 1.0, 'descent_max_rate': 2.5,
                'proximity_gain': 0.0, 'vertical_log_scale': 50.0,
                'time_penalty': 0.3},
        }
        sc = spring_configs[stage]

        computer = SpringRewardComputer(
            k_rotation=sc['k_rotation'],
            k_horizontal=sc['k_horizontal'],
            k_vertical=sc['k_vertical'],
            c_angular=sc['c_angular'],
            c_descent=sc['c_descent'],
            c_ascend=sc['c_ascend'],
            c_horizontal_vel=sc['c_horizontal_vel'],
            descent_max_rate=sc['descent_max_rate'],
            proximity_gain=sc.get('proximity_gain', 0.0),
            vertical_log_scale=sc.get('vertical_log_scale', 0.0),
            time_penalty=sc.get('time_penalty', 0.1),
            spawn_altitude=SPAWN_ALTITUDE,
        )

        num_actions = 5 if stage == 1 else 7
        max_steps_per_stage = {1: 1500, 2: 2000, 3: 3000}
        self.env = GameTrainingEnv(render_mode=render_mode, spring_computer=computer,
                                   num_actions=num_actions,
                                   max_episode_steps=max_steps_per_stage.get(stage, 2000))
        self.stage = stage
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_names = self.ACTION_NAMES_5 if stage == 1 else self.ACTION_NAMES_7

        # Stage 3 uses same reset as stage 2 (no constrained initial conditions)

        # Stage-specific parameters
        self.stage_configs = {
            1: {
                'max_angle': np.pi/2,
                'max_steps': 1500,
                'success_angle': 0.14,          # ~8 degrees
                'success_angular_vel': 0.3,
                'success_vel_y': 1.5,
                'success_reward': 200.0,
                'graduation_threshold': 95.0,
                'min_episodes': 200,
                'sustained_steps': 50,
            },
            2: {
                'max_angle': np.pi/3,
                'max_steps': 2000,
                'success_angle': 0.26,          # ~15 degrees
                'success_vel_y': 2.0,
                'success_dist_x': 15.0,
                'success_angular_vel': 0.3,
                'success_altitude': 20.0,
                'approach_reward': 5.0,
                'graduation_threshold': 70.0,
                'min_episodes': 1000,
            },
            3: {
                'max_angle': np.pi/3,
                'max_steps': 3000,
                'success_angle': 0.26,          # ~15 degrees
                'success_vel_y': 2.0,
                'success_dist_x': 15.0,
                'success_angular_vel': 0.3,
                'success_altitude': 20.0,
                'approach_reward': 5.0,
                'landing_required': True,
                'graduation_threshold': 50.0,
                'min_episodes': 1500,
            }
        }

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        config = self.stage_configs[self.stage]

        # Stage 1: constrained initial conditions for stabilization learning
        if self.stage == 1:
            max_angle = config['max_angle']
            initial_angle = float(self.env.np_random.uniform(-max_angle, max_angle))

            descent = self.env.lander.descent_stage
            ascent = self.env.lander.ascent_stage

            ascent_offset_y = ascent.position.y - descent.position.y
            ascent_offset_x = ascent.position.x - descent.position.x

            descent.angle = initial_angle
            ascent.angle = initial_angle

            sin_a = math.sin(initial_angle)
            cos_a = math.cos(initial_angle)
            new_offset_x = ascent_offset_x * cos_a - ascent_offset_y * sin_a
            new_offset_y = ascent_offset_x * sin_a + ascent_offset_y * cos_a
            ascent.position = b2Vec2(
                descent.position.x + new_offset_x,
                descent.position.y + new_offset_y
            )

            self.env.cumulative_angle = -initial_angle
            self.env.prev_raw_angle = initial_angle
            self.env.prev_energy = None

        self.steps = 0
        self.max_steps = config['max_steps']
        self._logged_success = False
        self._action_counts = {i: 0 for i in range(self.env.num_actions)}
        self._stable_steps = 0

        return self.env._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        self._action_counts[action] = self._action_counts.get(action, 0) + 1

        # Periodic logging
        log_interval = 50
        if self.steps % log_interval == 0 or self.steps <= 10:
            angle = self.env.cumulative_angle
            angle_deg = math.degrees(angle)
            angular_vel = -self.env.lander.descent_stage.angularVelocity
            vel = self.env.lander.descent_stage.linearVelocity
            pos = self.env.lander.descent_stage.position
            terrain_h = get_terrain_height_at(pos.x, self.env.terrain_pts)
            altitude = pos.y - terrain_h

            action_name = self.action_names.get(action, f"?{action}")
            throttle_pct = int(self.env.throttle * 100)
            total_actions = sum(self._action_counts.values())
            action_dist = " ".join([f"{self.action_names[i]}:{100*self._action_counts[i]//total_actions}%"
                                    for i in range(self.env.num_actions) if self._action_counts[i] > 0])

            target_x = (self.env.target_pad["x1"] + self.env.target_pad["x2"]) / 2.0
            dist_x = pos.x - target_x

            print(f"  [S{self.steps:4d}] Act={action_name:5s} | "
                  f"Ang={angle_deg:+6.1f} AngV={angular_vel:+5.2f} | "
                  f"Alt={altitude:5.1f}m VelY={vel.y:+5.1f} DistX={dist_x:+6.1f} | "
                  f"Thr={throttle_pct}% | {action_dist}")

        config = self.stage_configs[self.stage]

        # Stage-specific success checks
        if self.stage == 1:
            sustained_steps = config.get('sustained_steps', 50)
            angle = abs(self.env.lander.descent_stage.angle)
            angular_vel = abs(self.env.lander.descent_stage.angularVelocity)
            vel_y = abs(self.env.lander.descent_stage.linearVelocity.y)
            success_ang_vel = config.get('success_angular_vel', 0.3)
            success_vel_y = config.get('success_vel_y', 1.5)

            if (angle < config['success_angle'] and
                    angular_vel < success_ang_vel and
                    vel_y < success_vel_y):
                self._stable_steps += 1
                reward += config['success_reward'] / 100.0
                info['stage_success'] = True
                if not self._logged_success:
                    print(f"  [STABLE] Ep {self.env.episode_count} Step {self.steps}: "
                          f"Angle={math.degrees(angle):.1f} AngVel={angular_vel:.2f} "
                          f"VelY={vel_y:.2f} (streak: {self._stable_steps})")
                    self._logged_success = True
            else:
                self._stable_steps = 0
                self._logged_success = False

            if self._stable_steps >= sustained_steps:
                reward += config['success_reward']
                terminated = True
                info['stage_success'] = True
                print(f"  [GRADUATED] Ep {self.env.episode_count} Step {self.steps}: "
                      f"SUSTAINED STABILITY+HOVER for {sustained_steps} steps!")

        elif self.stage == 2:
            # Stage 2: approach bonus + large landing reward from spring computer
            angle = abs(self.env.lander.descent_stage.angle)
            vel_y = self.env.lander.descent_stage.linearVelocity.y
            angular_vel = abs(self.env.lander.descent_stage.angularVelocity)
            success_ang_vel = config.get('success_angular_vel', 0.3)

            pos = self.env.lander.descent_stage.position
            target_x = (self.env.target_pad["x1"] + self.env.target_pad["x2"]) / 2.0
            dist_x = abs(pos.x - target_x)
            terrain_h = get_terrain_height_at(pos.x, self.env.terrain_pts)
            altitude = pos.y - terrain_h
            max_alt = config.get('success_altitude', 20.0)

            # Approach bonus: guides agent toward landing configuration
            if (angle < config['success_angle'] and
                    abs(vel_y) < config['success_vel_y'] and
                    dist_x < config['success_dist_x'] and
                    angular_vel < success_ang_vel and
                    altitude < max_alt):
                self._stable_steps += 1
                approach_reward = config.get('approach_reward', 5.0)
                reward += approach_reward
                info['stage_success'] = True
                if not self._logged_success:
                    print(f"  [APPROACH] Ep {self.env.episode_count} Step {self.steps}: "
                          f"ON APPROACH | Angle={math.degrees(angle):.1f} DistX={dist_x:.1f}m "
                          f"Alt={altitude:.1f}m VelY={vel_y:.1f} (streak: {self._stable_steps})")
                    self._logged_success = True
            else:
                self._stable_steps = 0
                self._logged_success = False

        elif self.stage == 3:
            # Stage 3: same approach bonus as stage 2, but success = actual landing
            angle = abs(self.env.lander.descent_stage.angle)
            vel_y = self.env.lander.descent_stage.linearVelocity.y
            angular_vel = abs(self.env.lander.descent_stage.angularVelocity)
            success_ang_vel = config.get('success_angular_vel', 0.3)

            pos = self.env.lander.descent_stage.position
            target_x = (self.env.target_pad["x1"] + self.env.target_pad["x2"]) / 2.0
            dist_x = abs(pos.x - target_x)
            terrain_h = get_terrain_height_at(pos.x, self.env.terrain_pts)
            altitude = pos.y - terrain_h
            max_alt = config.get('success_altitude', 20.0)

            # Approach bonus (same as stage 2)
            if (angle < config['success_angle'] and
                    abs(vel_y) < config['success_vel_y'] and
                    dist_x < config['success_dist_x'] and
                    angular_vel < success_ang_vel and
                    altitude < max_alt):
                approach_reward = config.get('approach_reward', 5.0)
                reward += approach_reward
                if not self._logged_success:
                    print(f"  [APPROACH] Ep {self.env.episode_count} Step {self.steps}: "
                          f"ON APPROACH | Angle={math.degrees(angle):.1f} DistX={dist_x:.1f}m "
                          f"Alt={altitude:.1f}m VelY={vel_y:.1f}")
                    self._logged_success = True
            else:
                self._logged_success = False

            # Success only on actual landing
            landing_status = info.get('landing_status', 'flying')
            if landing_status == 'landed':
                info['stage_success'] = True
                print(f"  [LANDED] Ep {self.env.episode_count} Step {self.steps}: "
                      f"TOUCHDOWN | Angle={math.degrees(angle):.1f} DistX={dist_x:.1f}m "
                      f"Alt={altitude:.1f}m VelY={vel_y:.1f}")

        # Truncate at max steps
        if self.steps >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ============================================================================
# Training functions (2-stage curriculum with lateral RCS)
# ============================================================================

def train_stage(stage, seed, episodes, save_dir='models', visualize=False,
                load_previous=True):
    """Train a single curriculum stage using game physics."""

    print(f"\n{'='*70}")
    print(f"LATERAL RCS CURRICULUM STAGE {stage} - SEED {seed}")
    print(f"{'='*70}")

    stage_names = {
        1: "STABILIZE + HOVER (5 actions — rotation RCS only)",
        2: "APPROACH PAD (7 actions — with lateral RCS)",
        3: "PRECISION LANDING (7 actions — touchdown required)"
    }
    num_actions = 5 if stage == 1 else 7
    print(f"Stage: {stage_names[stage]}")
    print(f"Actions: {num_actions}")
    print(f"Episodes: {episodes}")
    print(f"Physics: Game-identical (solver 10/10)")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    env = CurriculumGameEnv(stage=stage, render_mode='human' if visualize else None)

    # Stage-specific agent hyperparameters
    agent_configs = {
        1: {'lr': 5e-4, 'buffer_size': int(1e5)},
        2: {'lr': 3e-4, 'buffer_size': int(2e5)},
        3: {'lr': 1e-4, 'buffer_size': int(3e5)},
    }
    ac = agent_configs[stage]

    agent = DoubleDQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        seed=seed,
        hidden_size=256,
        lr=ac['lr'],
        batch_size=128,
        gamma=0.99,
        tau=1e-3,
        buffer_size=ac['buffer_size'],
        update_every=4
    )

    # Load weights from previous stage
    if load_previous and stage == 2:
        # Stage 2 loads from Stage 1 with expanded action space (5 -> 7)
        prev_path = os.path.join(save_dir, f'lateral_stage1_seed{seed}_best.pth')
        if os.path.exists(prev_path):
            agent.load_with_expanded_actions(prev_path, old_action_size=5)
            print(f"Loaded Stage 1 weights with 5->7 action expansion")
        else:
            print(f"WARNING: No Stage 1 model found at {prev_path}")
    elif load_previous and stage == 3:
        # Stage 3 loads from Stage 2 (same action space, 7 -> 7)
        prev_path = os.path.join(save_dir, f'lateral_stage2_seed{seed}_best.pth')
        if os.path.exists(prev_path):
            agent.load(prev_path)
            print(f"Loaded Stage 2 weights (7 actions)")
        else:
            print(f"WARNING: No Stage 2 model found at {prev_path}")

    scores = []
    scores_window = deque(maxlen=100)
    successes = deque(maxlen=100)

    eps_configs = {
        1: {'start': 1.0, 'end': 0.05, 'decay': 0.997},
        2: {'start': 0.3, 'end': 0.02, 'decay': 0.998},
        3: {'start': 0.15, 'end': 0.01, 'decay': 0.999},
    }
    ec = eps_configs[stage]
    eps = ec['start']
    eps_end = ec['end']
    eps_decay = ec['decay']

    best_success_rate = 0
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if visualize:
                env.render()

        # Track success
        success = info.get('stage_success', False) or info.get('landing_status') == 'landed'
        successes.append(1 if success else 0)

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        avg = np.mean(scores_window)
        success_rate = 100 * sum(successes) / len(successes)

        # Save best model
        if episode >= 50 and success_rate > best_success_rate:
            best_success_rate = success_rate
            agent.save(os.path.join(save_dir, f'lateral_stage{stage}_seed{seed}_best.pth'))

        if episode % 50 == 0:
            elapsed = time.time() - start_time
            config = env.stage_configs[stage]
            grad_thresh = config.get('graduation_threshold', 70.0)
            print(f'Episode {episode:4d}/{episodes} | Avg: {avg:7.1f} | '
                  f'Success: {success_rate:5.1f}% | Best: {best_success_rate:5.1f}% | '
                  f'Grad@{grad_thresh:.0f}% | Eps: {eps:.3f} | {elapsed/60:.1f}m')

        # Early graduation
        config = env.stage_configs[stage]
        grad_thresh = config.get('graduation_threshold', 70.0)
        min_eps = config.get('min_episodes', 200)
        if episode >= min_eps and success_rate >= grad_thresh:
            print(f"\n{'='*70}")
            print(f"STAGE {stage} GRADUATED at episode {episode}!")
            print(f"Success Rate: {success_rate:.1f}% >= {grad_thresh:.0f}% threshold")
            print(f"{'='*70}")
            break

    # Save final model
    agent.save(os.path.join(save_dir, f'lateral_stage{stage}_seed{seed}_final.pth'))
    env.close()

    elapsed = time.time() - start_time
    final_success = 100 * sum(successes) / len(successes)

    print(f'\n{"="*70}')
    print(f'STAGE {stage} COMPLETE - SEED {seed}')
    print(f'{"="*70}')
    print(f'Episodes: {episode}')
    print(f'Final Success Rate: {final_success:.1f}%')
    print(f'Best Success Rate: {best_success_rate:.1f}%')
    print(f'Training Time: {elapsed/60:.1f} minutes')
    print(f'{"="*70}')

    return {
        'stage': stage,
        'seed': seed,
        'episodes': episode,
        'final_success_rate': final_success,
        'best_success_rate': best_success_rate,
        'train_time': elapsed
    }


def train_full_curriculum(seed, episodes_per_stage=1000, save_dir='models',
                          visualize=False, start_stage=1):
    """Train through both curriculum stages using game physics."""

    print("\n" + "="*70)
    print("LATERAL RCS 3-STAGE CURRICULUM TRAINING")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Episodes per stage: {episodes_per_stage}")
    print(f"Starting from stage: {start_stage}")
    print(f"Physics: Game-identical (solver 10/10)")
    print(f"Stage 1: Stabilize+Hover (5 actions)")
    print(f"Stage 2: Approach Pad with lateral RCS (7 actions)")
    print(f"Stage 3: Precision Landing (7 actions — touchdown required)")
    print("="*70)

    all_results = []
    total_start = time.time()

    for stage in range(start_stage, 4):
        result = train_stage(
            stage=stage,
            seed=seed,
            episodes=episodes_per_stage,
            save_dir=save_dir,
            visualize=visualize,
            load_previous=(stage > start_stage or stage == 2)
        )
        all_results.append(result)

        if result['best_success_rate'] < 50:
            print(f"\nWARNING: Stage {stage} not mastered (only {result['best_success_rate']:.1f}%)")
            print("Consider more training before proceeding.")

    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("LATERAL RCS 3-STAGE CURRICULUM TRAINING SUMMARY")
    print("="*70)
    for r in all_results:
        actions = 5 if r['stage'] == 1 else 7
        print(f"  Stage {r['stage']} ({actions} actions): {r['best_success_rate']:.1f}% success in {r['episodes']} episodes")
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print("="*70)

    # Save summary
    summary_path = os.path.join(save_dir, f'lateral_summary_seed{seed}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'seed': seed,
            'total_time_minutes': total_time / 60,
            'physics': 'game_identical_10_10',
            'curriculum': '3_stage_lateral_rcs',
            'results': all_results
        }, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lateral RCS training for Apollo Lander')
    parser.add_argument('--seed', type=int, default=999, help='Random seed')
    parser.add_argument('--stage', type=int, default=1, help='Start from this stage (1-3)')
    parser.add_argument('--episodes', type=int, default=1000, help='Episodes per stage')
    parser.add_argument('--single-stage', action='store_true', help='Only train specified stage')
    parser.add_argument('--visualize', action='store_true', help='Watch training live')
    parser.add_argument('--save-dir', type=str, default='models', help='Model directory')

    args = parser.parse_args()

    if args.single_stage:
        train_stage(
            stage=args.stage,
            seed=args.seed,
            episodes=args.episodes,
            save_dir=args.save_dir,
            visualize=args.visualize
        )
    else:
        train_full_curriculum(
            seed=args.seed,
            episodes_per_stage=args.episodes,
            save_dir=args.save_dir,
            visualize=args.visualize,
            start_stage=args.stage
        )
