"""
Spring-Based Reward Computer (v2 — Lateral RCS Redesign)

Computes spring potential energy for reward shaping in lunar lander training.
Uses potential-based reward shaping: reward = -(E_new - E_old), so reducing
total energy is rewarded and increasing it is penalized.

Springs:
  A (Rotation): 3 springs on imaginary circle — zero energy when upright
  B (Horizontal): Centering spring toward pad — zero energy when directly above pad
  C (Vertical): Altitude spring — zero energy at touchdown
      Quadratic mode (default): 0.5*k*h² — gradient decreases near surface
      Log mode (vertical_log_scale>0): k*scale*log(1+h) — gradient increases near surface

Dampers:
  D (Angular velocity): Penalizes rotational speed
  E (Descent rate): Tracks target descent profile
  F (Ascend penalty): Penalizes upward velocity (wastes fuel)
  G (Horizontal velocity): Penalizes lateral speed (smooth translation)
"""

import math


class SpringRewardComputer:
    """Computes spring potential energy and reward deltas for lunar lander training."""

    def __init__(self, k_rotation=20.0, k_horizontal=0.0, k_vertical=0.0,
                 c_angular=30.0, c_descent=3.0, c_ascend=5.0,
                 c_horizontal_vel=0.0, descent_max_rate=2.5,
                 proximity_gain=0.0, vertical_log_scale=0.0,
                 circle_radius=5.0, alpha_deg=10.0,
                 h_top=3.35, h_bot=-1.836,
                 time_penalty=0.1, spawn_altitude=50.0,
                 rotation_deadzone_deg=10.0):
        """
        Args:
            k_rotation: Spring constant for rotational springs (A)
            k_horizontal: Spring constant for horizontal centering (B)
            k_vertical: Spring constant for vertical descent spring (C)
            c_angular: Damping coefficient for angular velocity (D)
            c_descent: Descent profile tracking coefficient (E)
            c_ascend: Penalty for ascending / positive vel_y (F)
            c_horizontal_vel: Damping for horizontal velocity (G)
            descent_max_rate: Target descent rate at spawn altitude (m/s)
            proximity_gain: Extra vertical spring tightening near surface (quadratic mode)
            vertical_log_scale: If >0, use log vertical spring instead of quadratic.
                E = k_vertical * scale * log(1 + altitude). Gradient = k*scale/(1+h),
                so reward per meter INCREASES near the surface (commitment reward).
            circle_radius: Radius of imaginary circle for rotation spring anchors
            alpha_deg: Angle offset (degrees) for top springs from vertical
            h_top: Height of lander top above descent stage center
            h_bot: Height of lander bottom (negative = below center)
            time_penalty: Per-step time penalty
            spawn_altitude: Reference altitude for normalization
            rotation_deadzone_deg: Tilt angle below which rotational energy is zero
        """
        self.k_rotation = k_rotation
        self.k_horizontal = k_horizontal
        self.k_vertical = k_vertical
        self.c_angular = c_angular
        self.c_descent = c_descent
        self.c_ascend = c_ascend
        self.c_horizontal_vel = c_horizontal_vel
        self.descent_max_rate = descent_max_rate
        self.proximity_gain = proximity_gain
        self.vertical_log_scale = vertical_log_scale
        self.circle_radius = circle_radius
        self.alpha_rad = math.radians(alpha_deg)
        self.h_top = h_top
        self.h_bot = h_bot
        self.time_penalty = time_penalty
        self.spawn_altitude = spawn_altitude
        self.rotation_deadzone_rad = math.radians(rotation_deadzone_deg)

        # Precompute rest lengths and dead zone threshold
        self._precompute_rest_lengths()

    def _precompute_rest_lengths(self):
        """Compute rest lengths for rotation springs at theta=0 (upright)."""
        R = self.circle_radius
        alpha = self.alpha_rad
        h_top = self.h_top
        h_bot = self.h_bot

        # Spring 1 (top-left): anchor at (+R*sin(alpha), +R*cos(alpha)), top at (0, h_top)
        dx1 = R * math.sin(alpha)
        dy1 = R * math.cos(alpha) - h_top
        self.rest_1 = math.sqrt(dx1**2 + dy1**2)

        # Spring 2 (top-right): mirror of spring 1
        self.rest_2 = self.rest_1

        # Spring 3 (bottom): anchor at (0, -R), bottom at (0, h_bot)
        self.rest_3 = abs(-R - h_bot)

        # Rotation dead zone: precompute E_rot at the dead zone angle
        dz = self.rotation_deadzone_rad
        if dz > 0:
            sin_dz = math.sin(dz)
            cos_dz = math.cos(dz)
            dz_top_x = -h_top * sin_dz
            dz_top_y = h_top * cos_dz
            dz_bot_x = -h_bot * sin_dz
            dz_bot_y = h_bot * cos_dz

            def _se(ax, ay, bx, by, rest_len):
                ddx = ax - bx
                ddy = ay - by
                length = math.sqrt(ddx * ddx + ddy * ddy)
                ext = length - rest_len
                return 0.5 * self.k_rotation * ext * ext

            E1_dz = _se(R * math.sin(alpha), R * math.cos(alpha), dz_top_x, dz_top_y, self.rest_1)
            E2_dz = _se(-R * math.sin(alpha), R * math.cos(alpha), dz_top_x, dz_top_y, self.rest_2)
            E3_dz = _se(0, -R, dz_bot_x, dz_bot_y, self.rest_3)
            self.E_rot_deadzone = E1_dz + E2_dz + E3_dz
        else:
            self.E_rot_deadzone = 0.0

    def compute_energy(self, lander_x, lander_y, angle, angular_vel,
                       pad_x, pad_y, vel_y=0.0, vel_x=0.0, terrain_height=0.0):
        """
        Compute total spring potential energy for the current lander state.

        Args:
            lander_x, lander_y: Lander position (world coords)
            angle: Lander angle (radians, Box2D convention)
            angular_vel: Angular velocity (rad/s)
            pad_x, pad_y: Landing pad center position
            vel_y: Vertical velocity
            vel_x: Horizontal velocity
            terrain_height: Terrain height at lander x position

        Returns:
            Total potential energy (float)
        """
        R = self.circle_radius
        alpha = self.alpha_rad
        h_top = self.h_top
        h_bot = self.h_bot

        sin_t = math.sin(angle)
        cos_t = math.cos(angle)

        # --- Spring A: Rotation (3 springs on imaginary circle) ---
        # Lander attachment points (rotate with lander)
        top_x = lander_x - h_top * sin_t
        top_y = lander_y + h_top * cos_t
        bot_x = lander_x - h_bot * sin_t
        bot_y = lander_y + h_bot * cos_t

        # Circle anchor points (follow lander position, NOT rotation)
        a1_x = lander_x + R * math.sin(alpha)
        a1_y = lander_y + R * math.cos(alpha)
        a2_x = lander_x - R * math.sin(alpha)
        a2_y = lander_y + R * math.cos(alpha)
        a3_x = lander_x
        a3_y = lander_y - R

        def spring_e(ax, ay, bx, by, rest_len, k):
            dx = ax - bx
            dy = ay - by
            length = math.sqrt(dx * dx + dy * dy)
            ext = length - rest_len
            return 0.5 * k * ext * ext

        E1 = spring_e(a1_x, a1_y, top_x, top_y, self.rest_1, self.k_rotation)
        E2 = spring_e(a2_x, a2_y, top_x, top_y, self.rest_2, self.k_rotation)
        E3 = spring_e(a3_x, a3_y, bot_x, bot_y, self.rest_3, self.k_rotation)

        # Apply rotation dead zone
        E_rot = max(0.0, E1 + E2 + E3 - self.E_rot_deadzone)

        # --- Spring B: Horizontal centering ---
        # Two opposing springs anchored at pad center, zero energy when directly above
        E_horizontal = 0.0
        if self.k_horizontal > 0:
            dist_x = lander_x - pad_x
            E_horizontal = 0.5 * self.k_horizontal * dist_x * dist_x

        # --- Spring C: Vertical descent ---
        altitude = max(0.1, lander_y - terrain_height)
        E_vertical = 0.0
        if self.k_vertical > 0:
            if self.vertical_log_scale > 0:
                # Log mode: E = k * scale * log(1 + h)
                # Gradient dE/dh = k * scale / (1 + h) — increases near surface
                E_vertical = self.k_vertical * self.vertical_log_scale * math.log(1.0 + altitude)
            else:
                # Quadratic mode: E = 0.5 * k_eff * h² — gradient decreases near surface
                k_eff = self.k_vertical * (1.0 + self.proximity_gain / max(altitude, 1.0))
                E_vertical = 0.5 * k_eff * altitude * altitude

        # --- Damper D: Angular velocity ---
        E_ang_damp = 0.5 * self.c_angular * angular_vel * angular_vel

        # --- Damper E: Descent rate tracking ---
        target_vy = -self.descent_max_rate * min(1.0, altitude / self.spawn_altitude)
        vel_error = vel_y - target_vy
        E_descent = 0.5 * self.c_descent * vel_error * vel_error

        # --- Damper F: Ascend penalty ---
        E_ascend = 0.0
        if vel_y > 0:
            E_ascend = 0.5 * self.c_ascend * vel_y * vel_y

        # --- Damper G: Horizontal velocity ---
        E_hv_damp = 0.0
        if self.c_horizontal_vel > 0:
            E_hv_damp = 0.5 * self.c_horizontal_vel * vel_x * vel_x

        return E_rot + E_horizontal + E_vertical + E_ang_damp + E_descent + E_ascend + E_hv_damp

    def compute_reward(self, energy_old, energy_new, rcs_active=False,
                       cumulative_angle=0.0, landing_status='flying',
                       fuel=0.0, max_fuel=1.0, on_target=False,
                       pad_mult=1, vel_x=0.0, vel_y=0.0):
        """
        Compute reward from energy delta and terminal conditions.

        Args:
            energy_old: Previous step's total energy (None for first step)
            energy_new: Current step's total energy
            rcs_active: Whether RCS thrusters fired this step
            cumulative_angle: Current cumulative angle (for RCS penalty)
            landing_status: 'flying', 'landed', 'crashed', 'out_of_fuel'
            fuel: Current fuel remaining
            max_fuel: Maximum fuel capacity
            on_target: Whether lander is on the target pad
            pad_mult: Landing pad multiplier
            vel_x: Horizontal velocity (for landing bonus)
            vel_y: Vertical velocity (for landing bonus)

        Returns:
            Reward (float)
        """
        # Potential-based shaping: reward = -(E_new - E_old)
        # Clip to prevent catastrophic weight updates from large energy swings
        if energy_old is not None:
            delta = energy_new - energy_old
            delta = max(-100.0, min(100.0, delta))
            reward = -delta
        else:
            reward = 0.0

        # Time penalty
        reward -= self.time_penalty

        # RCS penalty when already nearly upright
        if rcs_active and abs(cumulative_angle) < 0.15:
            reward -= 0.03

        # Terminal rewards (large to dominate over shaping/approach bonuses)
        if landing_status == 'landed':
            if on_target:
                reward += 4000.0
                reward += 750.0 * pad_mult
            else:
                reward += 500.0

            # Fuel efficiency bonus
            reward += 200.0 * (fuel / max_fuel) if max_fuel > 0 else 0.0

            # Upright landing bonus
            if abs(cumulative_angle) < 0.1:
                reward += 400.0
            elif abs(cumulative_angle) < 0.2:
                reward += 200.0

            # Soft landing bonus
            landing_speed = math.sqrt(vel_x**2 + vel_y**2)
            if landing_speed < 1.0:
                reward += 400.0
            elif landing_speed < 2.0:
                reward += 200.0

        elif landing_status == 'crashed':
            reward = -1500.0

        elif landing_status == 'out_of_fuel':
            reward -= 800.0

        return reward
