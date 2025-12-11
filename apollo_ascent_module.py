"""
Apollo Ascent Module - WorldObject Implementation

The ascent stage of the Apollo Lunar Module as a standalone WorldObject.
Includes RCS pods, docking port, and all physics/rendering.
"""

import math
import pygame
from Box2D import b2PolygonShape, b2Vec2
from world import WorldObject
from apollo_rcs_pod import RCSPod
from apollolander import (
    REAL_ASCENT_MASS_KG,
    MASS_SCALE,
    ASCENT_MAIN_THRUST_FACTOR,
    RCS_THRUST_FACTOR,
)


def rotate_vec(vec, angle):
    """Rotate a 2D vector by angle (radians)."""
    ca, sa = math.cos(angle), math.sin(angle)
    return b2Vec2(vec.x * ca - vec.y * sa, vec.x * sa + vec.y * ca)


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class ApolloAscent(WorldObject):
    """
    Apollo Ascent Module as a WorldObject.

    Features:
    - RCS pods for attitude control
    - Docking port at top
    - Ascent engine
    - Realistic mass and physics
    """

    def __init__(
        self,
        world,
        position,
        scale=1.0,
        ascent_density=0.6,
        friction=0.9,
        restitution=0.1,
    ):
        """
        Initialize ascent module.

        Args:
            world: ApolloWorld instance
            position: b2Vec2 position in meters
            scale: Scale factor
            ascent_density: Density for body
            friction: Friction coefficient
            restitution: Bounce coefficient
        """
        super().__init__(world)
        self.scale = scale
        self.color = (180, 180, 180)  # Gray color

        # Create Box2D body in the world's physics world
        self.body = self._create_body(
            world.b2world,
            position,
            scale,
            ascent_density,
            friction,
            restitution,
        )

        # Apply mass scaling to match real Apollo ascent mass
        self._apply_mass_scaling(self.body, REAL_ASCENT_MASS_KG)

        # Create RCS pods (left and right) using RCSPod class
        h = 1.5 * scale
        w = 2.0 * scale
        rcs_y = 0.3 * scale  # RCS_OFFSET_Y from apollolander.py
        rcs_center_offset_x = w + (0.67 / 2.0) * scale  # Body edge + half pod width

        self.rcs_pod_left = RCSPod(
            center=b2Vec2(-rcs_center_offset_x, rcs_y),
            side_direction=-1,  # Left pod, side thruster points left (inward)
            scale=scale
        )
        self.rcs_pod_right = RCSPod(
            center=b2Vec2(rcs_center_offset_x, rcs_y),
            side_direction=1,  # Right pod, side thruster points right (inward)
            scale=scale
        )

        # Build thruster dictionary for physics (maps thruster names to positions/directions)
        self.rcs_thrusters = {
            "L_UP": self.rcs_pod_left.thrusters["UP"],
            "L_DOWN": self.rcs_pod_left.thrusters["DOWN"],
            "L_SIDE": self.rcs_pod_left.thrusters["SIDE"],
            "R_UP": self.rcs_pod_right.thrusters["UP"],
            "R_DOWN": self.rcs_pod_right.thrusters["DOWN"],
            "R_SIDE": self.rcs_pod_right.thrusters["SIDE"],
        }

        # Fuel system (real Apollo ascent stage: 2,353 kg propellant)
        self.max_fuel_kg = 2353.0  # Maximum fuel capacity in kg
        self.fuel_kg = 2353.0  # Current fuel in kg (starts full)

        self.active_thrusters = set()
        self.main_thrusting = False

    def _create_body(self, b2world, position, s, density, friction, restitution):
        """Create ascent stage Box2D body with RCS pods."""
        body = b2world.CreateDynamicBody(
            position=position,
            angle=0.0,
            linearDamping=0.1,
            angularDamping=0.25,
            bullet=True,
        )

        h = 1.5 * s
        w = 2.0 * s

        # Octagonal hull
        hull_vertices = [
            b2Vec2(-w, -h * 0.3),
            b2Vec2(-w * 0.4, -h),
            b2Vec2(w * 0.4, -h),
            b2Vec2(w, -h * 0.3),
            b2Vec2(w, h * 0.3),
            b2Vec2(w * 0.4, h),
            b2Vec2(-w * 0.4, h),
            b2Vec2(-w, h * 0.3),
        ]

        hull_shape = b2PolygonShape(vertices=hull_vertices)
        body.CreateFixture(
            shape=hull_shape,
            density=density,
            friction=friction,
            restitution=restitution,
        )

        # Ascent engine nozzle
        bell_height = 0.9 * s
        bell_top_y = -h * 0.85
        bell_bottom_y = bell_top_y - bell_height
        bell_top_half_w = 0.28 * s
        bell_bottom_half_w = 0.7 * s

        ascent_bell_vertices = [
            b2Vec2(-bell_top_half_w, bell_top_y),
            b2Vec2(bell_top_half_w, bell_top_y),
            b2Vec2(bell_bottom_half_w, bell_bottom_y),
            b2Vec2(-bell_bottom_half_w, bell_bottom_y),
        ]
        ascent_bell_shape = b2PolygonShape(vertices=ascent_bell_vertices)
        body.CreateFixture(
            shape=ascent_bell_shape,
            density=density * 0.5,
            friction=friction,
            restitution=restitution,
        )

        bell_base_local = b2Vec2(0.0, bell_bottom_y)

        # Docking block
        dock_half_w = 0.5 * s
        dock_half_h = 0.3 * s
        dock_center = b2Vec2(0.0, h * 1.1)
        dock_shape = b2PolygonShape(
            box=(dock_half_w, dock_half_h, dock_center, 0.0)
        )
        body.CreateFixture(
            shape=dock_shape,
            density=density * 0.7,
            friction=friction,
            restitution=restitution,
        )

        # Note: RCS pods are now created using RCSPod class in __init__
        # They are visual/logical components, not physics fixtures

        body.userData = {
            "type": "ascent",
            "scale": s,
            "bell_vertices": ascent_bell_vertices,
            "bell_base_local": bell_base_local,
            "hull_w": w,
            "hull_h": h,
        }

        return body

    def _apply_mass_scaling(self, body, real_mass_kg):
        """Rescale body mass to match real Apollo ascent mass."""
        md = body.massData
        if md.mass <= 0:
            return
        target_mass = real_mass_kg / MASS_SCALE
        factor = target_mass / md.mass
        md.mass *= factor
        md.I *= factor
        body.massData = md

    def _update_body_mass(self):
        """Update Box2D body mass based on current fuel level."""
        # Calculate current total mass: dry mass + fuel mass
        # REAL_ASCENT_MASS_KG includes full fuel (4,798 kg total, 2,353 kg fuel)
        dry_mass_kg = REAL_ASCENT_MASS_KG - self.max_fuel_kg  # 2,445 kg dry mass
        current_mass_kg = dry_mass_kg + self.fuel_kg

        # Convert to Box2D mass units (scaled)
        target_mass = current_mass_kg / MASS_SCALE

        # Update Box2D mass
        md = self.body.massData
        if md.mass > 0:
            factor = target_mass / md.mass
            md.mass *= factor
            md.I *= factor
            self.body.massData = md

    def apply_main_thrust(self, dt):
        """
        Apply main ascent engine thrust (15,600 N real Apollo specs).

        Includes fuel consumption and dynamic mass updates.
        """
        # Check if fuel is available
        if self.fuel_kg <= 0:
            self.main_thrusting = False
            return

        # Thrust along +Y in body frame (upward)
        world_dir = rotate_vec(b2Vec2(0.0, 1.0), self.body.angle)
        # Force = mass * acceleration (thrust factor is acceleration in m/s²)
        force_magnitude = self.body.mass * ASCENT_MAIN_THRUST_FACTOR
        force = force_magnitude * world_dir

        # Apply at engine bell base for realistic torque
        bell_base_local = self.body.userData["bell_base_local"]
        attach_world = local_to_world(self.body, bell_base_local)
        self.body.ApplyForce(force, attach_world, wake=True)
        self.main_thrusting = True

        # Consume fuel based on thrust and time
        # Fuel flow rate = Thrust / (Specific Impulse * g0)
        # Apollo Ascent Engine: Isp = 311s, g0 = 9.81 m/s²
        # Fuel flow = 15,600 N / (311 * 9.81) = 5.11 kg/s
        fuel_flow_rate = 15600.0 / (311.0 * 9.81)  # kg/s
        fuel_consumed = fuel_flow_rate * dt
        self.fuel_kg = max(0.0, self.fuel_kg - fuel_consumed)

        # Update body mass to reflect fuel consumption
        self._update_body_mass()

    def apply_rcs_thrust(self, thruster_names, dt):
        """Apply RCS thrust from specific thrusters."""
        for name in thruster_names:
            if name in self.rcs_thrusters:
                t = self.rcs_thrusters[name]
                pos_world = local_to_world(self.body, t["pos"])
                dir_world = rotate_vec(t["dir"], self.body.angle)
                # Force = mass * acceleration (thrust factor is acceleration in m/s²)
                force_magnitude = self.body.mass * RCS_THRUST_FACTOR
                force = force_magnitude * dir_world
                self.body.ApplyForce(force, pos_world, wake=True)
                self.active_thrusters.add(name)

    def update(self, dt):
        """Update ascent module physics."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

        # Note: Do NOT clear thrust states here - they're managed by the game loop
        # The game loop should clear them at the start of each frame before reading input

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw ascent module to screen."""
        PPM = self.world.ppm

        # Colors
        BODY_COLOR = self.color
        NOZZLE_COLOR = (40, 40, 55)
        OUTLINE = (0, 0, 0)
        DOCK_COLOR = (70, 70, 80)

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * PPM
            y = screen_height / 2 - (pos.y - cam_y) * PPM
            return int(x), int(y)

        # Draw main octagonal hull
        hull_w = self.body.userData["hull_w"]
        hull_h = self.body.userData["hull_h"]
        hull_vertices = [
            b2Vec2(-hull_w, -hull_h * 0.3),
            b2Vec2(-hull_w * 0.4, -hull_h),
            b2Vec2(hull_w * 0.4, -hull_h),
            b2Vec2(hull_w, -hull_h * 0.3),
            b2Vec2(hull_w, hull_h * 0.3),
            b2Vec2(hull_w * 0.4, hull_h),
            b2Vec2(-hull_w * 0.4, hull_h),
            b2Vec2(-hull_w, hull_h * 0.3),
        ]
        hull_world = [local_to_world(self.body, v) for v in hull_vertices]
        pygame.draw.polygon(surface, BODY_COLOR, [w2s(p) for p in hull_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in hull_world], 2)

        # Draw ascent engine nozzle
        bell_vertices = self.body.userData["bell_vertices"]
        bell_world = [local_to_world(self.body, v) for v in bell_vertices]
        pygame.draw.polygon(surface, NOZZLE_COLOR, [w2s(p) for p in bell_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in bell_world], 2)

        # Draw docking block
        dock_half_w = 0.5 * self.scale
        dock_half_h = 0.3 * self.scale
        dock_center = b2Vec2(0.0, hull_h * 1.1)
        dock_vertices = [
            dock_center + b2Vec2(-dock_half_w, -dock_half_h),
            dock_center + b2Vec2(dock_half_w, -dock_half_h),
            dock_center + b2Vec2(dock_half_w, dock_half_h),
            dock_center + b2Vec2(-dock_half_w, dock_half_h),
        ]
        dock_world = [local_to_world(self.body, v) for v in dock_vertices]
        pygame.draw.polygon(surface, DOCK_COLOR, [w2s(p) for p in dock_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in dock_world], 2)

        # Draw RCS pods using RCSPod class
        # Determine which thrusters are active for each pod
        left_active = {t.replace("L_", "") for t in self.active_thrusters if t.startswith("L_")}
        right_active = {t.replace("R_", "") for t in self.active_thrusters if t.startswith("R_")}

        self.rcs_pod_left.draw(surface, self.body, left_active, cam_x, cam_y,
                               screen_width, screen_height, PPM)
        self.rcs_pod_right.draw(surface, self.body, right_active, cam_x, cam_y,
                                screen_width, screen_height, PPM)

        # Draw ascent module distinctive features
        self._draw_ascent_details(surface, hull_w, hull_h, cam_x, cam_y, screen_width, screen_height, PPM)

    def _draw_ascent_details(self, surface, hull_w, hull_h, cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw distinctive features: windows, hatch, antenna, stripes."""
        s = self.scale

        # Colors
        LINE_COLOR = (0, 0, 0)
        HATCH_COLOR = (100, 100, 100)
        WINDOW_COLOR = (50, 50, 80)
        MAST_COLOR = (80, 80, 80)
        DISH_COLOR = (255, 255, 255)

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        # Vertical black stripes on sides
        top_y = hull_h * 0.6
        bot_y = hull_h * 0.0

        # Left stripe
        left_top = b2Vec2(-0.4 * s, top_y)
        left_bot = b2Vec2(-0.9 * s, bot_y)
        lt_w = w2s(local_to_world(self.body, left_top))
        lb_w = w2s(local_to_world(self.body, left_bot))
        pygame.draw.line(surface, LINE_COLOR, lt_w, lb_w, 2)

        # Right stripe
        right_top = b2Vec2(0.4 * s, top_y)
        right_bot = b2Vec2(0.9 * s, bot_y)
        rt_w = w2s(local_to_world(self.body, right_top))
        rb_w = w2s(local_to_world(self.body, right_bot))
        pygame.draw.line(surface, LINE_COLOR, rt_w, rb_w, 2)

        # Hatch (rectangular panel on lower front)
        hatch_center = b2Vec2(0.0, -hull_h * 0.25)
        hatch_half_w = 0.55 * s
        hatch_half_h = 0.35 * s
        hatch_verts = [
            hatch_center + b2Vec2(-hatch_half_w, -hatch_half_h),
            hatch_center + b2Vec2(hatch_half_w, -hatch_half_h),
            hatch_center + b2Vec2(hatch_half_w, hatch_half_h),
            hatch_center + b2Vec2(-hatch_half_w, hatch_half_h),
        ]
        hatch_world = [local_to_world(self.body, v) for v in hatch_verts]
        pygame.draw.polygon(surface, HATCH_COLOR, [w2s(p) for p in hatch_world])

        # Triangular windows (left and right)
        win_y_mid = hull_h * 0.25
        win_height = 0.6 * s

        # Left window
        left_outer = b2Vec2(-hull_w * 0.9, win_y_mid)
        left_inner_top = b2Vec2(-0.55 * s, win_y_mid + win_height / 2)
        left_inner_bot = b2Vec2(-0.55 * s, win_y_mid - win_height / 2)
        left_tri = [left_outer, left_inner_top, left_inner_bot]
        left_tri_world = [local_to_world(self.body, v) for v in left_tri]
        pygame.draw.polygon(surface, WINDOW_COLOR, [w2s(p) for p in left_tri_world])

        # Right window
        right_outer = b2Vec2(hull_w * 0.9, win_y_mid)
        right_inner_top = b2Vec2(0.55 * s, win_y_mid + win_height / 2)
        right_inner_bot = b2Vec2(0.55 * s, win_y_mid - win_height / 2)
        right_tri = [right_outer, right_inner_bot, right_inner_top]
        right_tri_world = [local_to_world(self.body, v) for v in right_tri]
        pygame.draw.polygon(surface, WINDOW_COLOR, [w2s(p) for p in right_tri_world])

        # Dish antenna (mast + circular dish on top)
        dish_base = b2Vec2(0.25 * s, hull_h * 1.05)
        dish_tip = b2Vec2(0.25 * s, hull_h * 1.35)
        base_world = w2s(local_to_world(self.body, dish_base))
        tip_world = w2s(local_to_world(self.body, dish_tip))
        pygame.draw.line(surface, MAST_COLOR, base_world, tip_world, 2)

        # Dish (circular)
        dish_center_local = dish_tip + b2Vec2(0.0, 0.35 * s)
        dish_center_screen = w2s(local_to_world(self.body, dish_center_local))
        radius = int(8 * s)
        pygame.draw.circle(surface, DISH_COLOR, dish_center_screen, radius, 0)

        # Crosshairs on dish
        cx, cy = dish_center_screen
        arm = int(radius * 0.5)
        pygame.draw.line(surface, LINE_COLOR, (cx - arm, cy), (cx + arm, cy), 1)
        pygame.draw.line(surface, LINE_COLOR, (cx, cy - arm), (cx, cy + arm), 1)

    def draw_engine(
        self,
        surface,
        main_on,
        tl, bl, tr, br, sl, sr,
        cam_x,
        cam_y,
        screen_width,
        screen_height,
    ):
        """Draw main engine flame if active (RCS handled by draw() method)."""
        # Clear active thrusters from previous frame
        self.active_thrusters.clear()

        # Track RCS thruster activity for rendering in draw() method
        if tl:
            self.active_thrusters.add("L_UP")
        if bl:
            self.active_thrusters.add("L_DOWN")
        if tr:
            self.active_thrusters.add("R_UP")
        if br:
            self.active_thrusters.add("R_DOWN")
        if sl:
            self.active_thrusters.add("L_SIDE")
        if sr:
            self.active_thrusters.add("R_SIDE")

        # Draw main engine flame if active
        if not main_on:
            return

        PPM = self.world.ppm
        FLAME_COLOR = (255, 190, 60)

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * PPM
            y = screen_height / 2 - (pos.y - cam_y) * PPM
            return int(x), int(y)

        # Get engine exit position from bell vertices
        bell_base_local = self.body.userData["bell_base_local"]
        bell_exit_world = local_to_world(self.body, bell_base_local)

        # Flame direction is opposite to body up direction
        # Body angle 0 = pointing right, so engine points down
        flame_dir = rotate_vec(b2Vec2(0.0, -1.0), self.body.angle)

        # Flame dimensions
        flame_len = 2.0 * self.scale
        flame_w = 0.8 * self.scale

        tip_flame = bell_exit_world + flame_dir * flame_len
        perp = b2Vec2(-flame_dir.y, flame_dir.x)
        p1 = bell_exit_world + perp * (flame_w / 2)
        p2 = bell_exit_world - perp * (flame_w / 2)

        pygame.draw.polygon(surface, FLAME_COLOR, [w2s(p1), w2s(p2), w2s(tip_flame)])
