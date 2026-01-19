"""
Apollo Service Module (SM) - WorldObject Implementation

The Service Module as a standalone WorldObject with its own physics body.
Features SPS engine with gimbal and RCS thrusters.
"""

import math
import pygame
from Box2D import b2PolygonShape, b2Vec2
from world import WorldObject
from apollo_rcs_pod import RCSPod


# SM-specific thrust values (real Apollo specifications)
SM_SPS_THRUST = 91190.0  # Service Propulsion System thrust in Newtons
SM_RCS_THRUST = 445.0  # RCS thruster force in Newtons (real Apollo spec)

# Real Apollo Service Module mass (kg)
# Dry mass: 6,150 kg + Fuel: 18,410 kg = 24,560 kg total
REAL_SM_MASS_KG = 24560.0  # Full mass with fuel
SM_MASS_SCALE = 100.0  # 100 kg → 1 Box2D mass unit


def rotate_vec(vec, angle):
    """Rotate a 2D vector by angle (radians)."""
    ca, sa = math.cos(angle), math.sin(angle)
    return b2Vec2(vec.x * ca - vec.y * sa, vec.x * sa + vec.y * ca)


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class ServiceModule(WorldObject):
    """
    Apollo Service Module as a WorldObject.

    Features:
    - Cylindrical body with realistic mass (~24,520 kg)
    - SPS main engine with gimbal (±6°)
    - RCS thruster pods (left and right)
    - Independent physics body
    """

    def __init__(self, world, position, scale=1.2):
        """
        Initialize Service Module.

        Args:
            world: ApolloWorld instance
            position: (x, y) tuple or b2Vec2 position in meters
            scale: Scale factor (default 1.2 to be larger than lander)
        """
        super().__init__(world)

        self.scale = scale

        # Convert position to tuple if it's a b2Vec2
        if hasattr(position, 'x') and hasattr(position, 'y'):
            position = (position.x, position.y)

        # Create Box2D body
        self.body = world.b2world.CreateDynamicBody(
            position=position,
            angle=0.0,  # Upright orientation (engine pointing down)
            linearDamping=world.linear_damping,
            angularDamping=world.angular_damping,
        )

        # Service Module dimensions (real Apollo SM: 3.9m diameter, 7.6m length)
        self.sm_half_w = (3.9 / 2.0) * scale  # 1.95m radius
        self.sm_half_h = (7.6 / 2.0) * scale  # 3.8m half-length
        self.sm_center = b2Vec2(0.0, 0.0)

        sm_vertices = [
            b2Vec2(self.sm_center.x - self.sm_half_w, self.sm_center.y - self.sm_half_h),
            b2Vec2(self.sm_center.x + self.sm_half_w, self.sm_center.y - self.sm_half_h),
            b2Vec2(self.sm_center.x + self.sm_half_w, self.sm_center.y + self.sm_half_h),
            b2Vec2(self.sm_center.x - self.sm_half_w, self.sm_center.y + self.sm_half_h),
        ]

        sm_top_y = self.sm_center.y + self.sm_half_h
        sm_bottom_y = self.sm_center.y - self.sm_half_h
        sm_height = 2.0 * self.sm_half_h

        # Mass settings via density
        # Real Apollo SM: 24,560 kg fully fueled (using module constant)
        sm_area = (2.0 * self.sm_half_w) * (2.0 * self.sm_half_h)
        sm_density = 1.0  # Will be rescaled after fixture creation

        # Create SM fixture
        self.fixture = self.body.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=sm_vertices),
            density=sm_density,
            friction=0.3,
            restitution=0.1,
        )

        # Apply mass scaling to match real Apollo SM mass
        self._apply_mass_scaling(self.body, REAL_SM_MASS_KG)

        # Store top Y position for CM attachment
        self.top_y = sm_top_y

        # Main engine nozzle (SPS) - gimballed (real Apollo SPS: 2.5m exit diameter, 3.88m length)
        nozzle_attach_y = sm_bottom_y
        attach_half_w = (0.8) * scale  # Throat ~0.8m radius estimate
        exit_half_w = (2.5 / 2.0) * scale  # 1.25m exit radius
        nozzle_length = 3.88 * scale  # 3.88m length

        # Pivot point (throat region)
        self.nozzle_attach_center = b2Vec2(0.0, nozzle_attach_y)

        # Canonical nozzle shape, pointing straight down in LOCAL frame
        self.nozzle_shape_local = [
            b2Vec2(-attach_half_w, 0.0),
            b2Vec2(+attach_half_w, 0.0),
            b2Vec2(+exit_half_w, -nozzle_length),
            b2Vec2(-exit_half_w, -nozzle_length),
        ]

        # Gimbal state
        self.nozzle_gimbal_angle = 0.0  # radians
        self.max_gimbal = math.radians(6.0)  # ±6°

        # Create nozzle fixture for collision
        # Note: Nozzle shape is relative to attachment point
        nozzle_vertices_for_fixture = [
            b2Vec2(-attach_half_w, nozzle_attach_y),
            b2Vec2(+attach_half_w, nozzle_attach_y),
            b2Vec2(+exit_half_w, nozzle_attach_y - nozzle_length),
            b2Vec2(-exit_half_w, nozzle_attach_y - nozzle_length),
        ]
        self.nozzle_fixture = self.body.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=nozzle_vertices_for_fixture),
            density=sm_density * 0.3,  # Lighter than main body
            friction=0.3,
            restitution=0.1,
        )

        # White bands on SM
        narrow_h = 0.3 * scale
        wide_h = 0.6 * scale
        band_margin = 0.15 * scale

        self.sm_band_top = {
            "center": b2Vec2(
                self.sm_center.x,
                sm_top_y - band_margin - narrow_h / 2.0
            ),
            "half_w": self.sm_half_w,
            "half_h": narrow_h / 2.0,
        }
        self.sm_band_bottom = {
            "center": b2Vec2(
                self.sm_center.x,
                sm_bottom_y + band_margin + wide_h / 2.0
            ),
            "half_w": self.sm_half_w,
            "half_h": wide_h / 2.0,
        }

        # SM forward gray panel
        self.sm_forward_panel = {
            "center": b2Vec2(self.sm_center.x, self.sm_center.y + 0.5 * scale),
            "half_w": 0.8 * scale,
            "half_h": 0.8 * scale,
        }

        # RCS pods (left and right) - using real-world scaled RCSPod class
        pod_half_w = (0.67 / 2.0) * scale  # Real Apollo quad dimensions
        pod_half_h = (0.82 / 2.0) * scale
        rcs_pod_scale = 1.0 * scale  # Full scale - RCSPod now has real dimensions

        # Position RCS pods approximately 1/4 of the SM length from the top (docking end)
        pod_center_y = sm_top_y - (sm_height / 4.0)
        # Position pods flush against SM edges (no gap)
        left_center_x = self.sm_center.x - (self.sm_half_w + pod_half_w)
        right_center_x = self.sm_center.x + (self.sm_half_w + pod_half_w)

        # Create RCSPod instances
        self.rcs_pod_left = RCSPod(
            center=b2Vec2(left_center_x, pod_center_y),
            side_direction=-1,  # Left pod, side thruster points left (inward)
            scale=rcs_pod_scale
        )
        self.rcs_pod_right = RCSPod(
            center=b2Vec2(right_center_x, pod_center_y),
            side_direction=1,  # Right pod, side thruster points right (inward)
            scale=rcs_pod_scale
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

        # Fuel system (real Apollo Service Module: 18,410 kg propellant for SPS)
        self.max_fuel_kg = 18410.0  # Maximum fuel capacity in kg
        self.fuel_kg = 18410.0  # Current fuel in kg (starts full)

        self.active_thrusters = set()
        self.main_thrusting = False

    def _apply_mass_scaling(self, body, real_mass_kg):
        """Rescale body mass to match real Apollo Service Module mass."""
        md = body.massData
        if md.mass <= 0:
            return
        target_mass = real_mass_kg / SM_MASS_SCALE
        factor = target_mass / md.mass
        md.mass *= factor
        md.I *= factor
        body.massData = md

    def _update_body_mass(self):
        """Update Box2D body mass based on current fuel level."""
        # Calculate current total mass: dry mass + fuel mass
        # REAL_SM_MASS_KG includes full fuel (24,560 kg total, 18,410 kg fuel)
        dry_mass_kg = REAL_SM_MASS_KG - self.max_fuel_kg  # 6,150 kg dry mass
        current_mass_kg = dry_mass_kg + self.fuel_kg

        # Convert to Box2D mass units (scaled)
        target_mass = current_mass_kg / SM_MASS_SCALE

        # Update Box2D mass
        md = self.body.massData
        if md.mass > 0:
            factor = target_mass / md.mass
            md.mass *= factor
            md.I *= factor
            self.body.massData = md

    def set_nozzle_angle(self, angle):
        """Set nozzle gimbal angle (radians), clamped to ±max_gimbal."""
        self.nozzle_gimbal_angle = max(
            -self.max_gimbal,
            min(self.max_gimbal, angle),
        )

    def adjust_nozzle(self, delta):
        """Adjust nozzle gimbal by delta radians."""
        self.set_nozzle_angle(self.nozzle_gimbal_angle + delta)

    def center_nozzle(self):
        """Reset nozzle gimbal to center (0°)."""
        self.nozzle_gimbal_angle = 0.0

    def apply_main_thrust(self, dt):
        """
        Apply main SPS thrust with gimbal angle (91,190 N real Apollo specs).

        Includes fuel consumption and dynamic mass updates.
        """
        # Check if fuel is available
        if self.fuel_kg <= 0:
            self.main_thrusting = False
            return

        # Thrust along +Y in engine frame, rotated by gimbal relative to body
        local_engine_dir = rotate_vec(b2Vec2(0.0, 1.0), self.nozzle_gimbal_angle)
        world_dir = rotate_vec(local_engine_dir, self.body.angle)
        force = SM_SPS_THRUST * world_dir

        # Apply at nozzle attach point to get realistic torque
        attach_world = local_to_world(self.body, self.nozzle_attach_center)
        self.body.ApplyForce(force, attach_world, wake=True)
        self.main_thrusting = True

        # Consume fuel based on thrust and time
        # Fuel flow rate = Thrust / (Specific Impulse * g0)
        # Apollo SPS Engine: Isp = 314s, g0 = 9.81 m/s²
        # Fuel flow = 91,190 N / (314 * 9.81) = 29.60 kg/s
        fuel_flow_rate = 91190.0 / (314.0 * 9.81)  # kg/s
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
                force = SM_RCS_THRUST * dir_world
                self.body.ApplyForce(force, pos_world, wake=True)
                self.active_thrusters.add(name)

    def update(self, dt):
        """Update SM physics."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

        # Note: Do NOT clear thrust states here - they're managed by the game loop
        # The game loop should clear them at the start of each frame before reading input

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw Service Module to screen."""
        PPM = self.world.ppm

        # Colors
        SM_COLOR = (200, 200, 230)
        NOZZLE_COLOR = (40, 40, 55)
        OUTLINE = (0, 0, 0)
        BAND_FILL = (240, 240, 250)
        PANEL_FILL = (210, 210, 220)
        FLAME_COLOR = (255, 190, 60)

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * PPM
            y = screen_height / 2 - (pos.y - cam_y) * PPM
            return int(x), int(y)

        # Draw SM body
        sm_world = [
            self.body.transform * v
            for v in self.fixture.shape.vertices
        ]
        pygame.draw.polygon(surface, SM_COLOR, [w2s(p) for p in sm_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in sm_world], 2)

        # Draw gimballed SPS nozzle
        nozzle_world = []
        for v in self.nozzle_shape_local:
            v_rot = rotate_vec(v, self.nozzle_gimbal_angle)
            p_local = self.nozzle_attach_center + v_rot
            nozzle_world.append(local_to_world(self.body, p_local))

        pygame.draw.polygon(surface, NOZZLE_COLOR, [w2s(p) for p in nozzle_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in nozzle_world], 2)

        # Draw main SPS engine flame if firing
        if self.main_thrusting:
            nozzle_exit_left = self.nozzle_shape_local[3]
            nozzle_exit_right = self.nozzle_shape_local[2]
            nozzle_exit_center = b2Vec2(
                (nozzle_exit_left.x + nozzle_exit_right.x) / 2,
                (nozzle_exit_left.y + nozzle_exit_right.y) / 2
            )

            cos_g = math.cos(self.nozzle_gimbal_angle)
            sin_g = math.sin(self.nozzle_gimbal_angle)
            nozzle_exit_gimbaled = b2Vec2(
                nozzle_exit_center.x * cos_g - nozzle_exit_center.y * sin_g,
                nozzle_exit_center.x * sin_g + nozzle_exit_center.y * cos_g
            )

            nozzle_exit_body = self.nozzle_attach_center + nozzle_exit_gimbaled
            nozzle_exit_world = local_to_world(self.body, nozzle_exit_body)

            thrust_dir_gimbaled = b2Vec2(-sin_g, cos_g)
            flame_dir = rotate_vec(-thrust_dir_gimbaled, self.body.angle)

            flame_len = 2.0 * self.scale
            flame_w = 1.0 * self.scale
            tip_flame = nozzle_exit_world + flame_dir * flame_len
            perp = b2Vec2(-flame_dir.y, flame_dir.x)
            p1 = nozzle_exit_world + perp * (flame_w / 2)
            p2 = nozzle_exit_world - perp * (flame_w / 2)
            pygame.draw.polygon(surface, FLAME_COLOR, [w2s(p1), w2s(p2), w2s(tip_flame)])

        # Draw white bands
        self._draw_panel_rect(surface, self.sm_band_top, BAND_FILL, OUTLINE, 1,
                              cam_x, cam_y, screen_width, screen_height, PPM)
        self._draw_panel_rect(surface, self.sm_band_bottom, BAND_FILL, OUTLINE, 1,
                              cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw forward panel
        self._draw_panel_rect(surface, self.sm_forward_panel, PANEL_FILL, OUTLINE, 1,
                              cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw RCS pods using RCSPod class
        # Determine which thrusters are active for each pod
        left_active = {t.replace("L_", "") for t in self.active_thrusters if t.startswith("L_")}
        right_active = {t.replace("R_", "") for t in self.active_thrusters if t.startswith("R_")}

        self.rcs_pod_left.draw(surface, self.body, left_active, cam_x, cam_y,
                               screen_width, screen_height, PPM)
        self.rcs_pod_right.draw(surface, self.body, right_active, cam_x, cam_y,
                                screen_width, screen_height, PPM)

        # Draw Service Module distinctive features
        self._draw_sm_details(surface, cam_x, cam_y, screen_width, screen_height, PPM)

    def _draw_panel_rect(self, surface, panel, fill, outline, line_width,
                         cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw a rectangular panel on the SM."""
        hw = panel["half_w"]
        hh = panel["half_h"]
        c = panel["center"]
        verts_local = [
            b2Vec2(c.x - hw, c.y - hh),
            b2Vec2(c.x + hw, c.y - hh),
            b2Vec2(c.x + hw, c.y + hh),
            b2Vec2(c.x - hw, c.y + hh),
        ]
        verts_world = [local_to_world(self.body, v) for v in verts_local]

        def w2s(pos):
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        verts_screen = [w2s(v) for v in verts_world]
        pygame.draw.polygon(surface, fill, verts_screen)
        pygame.draw.polygon(surface, outline, verts_screen, line_width)

    def _draw_sm_details(self, surface, cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw Service Module distinctive features: radiators, sectors, antennas."""
        s = self.scale

        # Colors
        RADIATOR_COLOR = (250, 250, 255)  # Bright white for radiators
        SECTOR_LINE = (40, 40, 50)  # Dark lines for sector divisions
        ANTENNA_COLOR = (200, 200, 200)  # Light gray for antennas
        DISH_COLOR = (255, 255, 255)  # White for high-gain antenna dish

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        # Draw sector dividing lines (6 radial sectors)
        # The SM was divided into 6 pie-slice sections
        sm_top = self.sm_center.y + self.sm_half_h
        sm_bottom = self.sm_center.y - self.sm_half_h
        sm_height = self.sm_half_h * 2.0

        # Vertical center line
        top_world = w2s(local_to_world(self.body, b2Vec2(0.0, sm_top)))
        bottom_world = w2s(local_to_world(self.body, b2Vec2(0.0, sm_bottom)))
        pygame.draw.line(surface, SECTOR_LINE, top_world, bottom_world, 1)

        # Draw two large radiator panels (on opposite sides)
        # Real SM had two large radiator panels for thermal control
        # Keep panels within SM body bounds (overlaid on the cylinder sides)
        radiator_height = self.sm_half_h * 1.6  # 80% of SM height
        radiator_width = self.sm_half_w * 0.3  # Narrow panels on the sides
        radiator_y_center = self.sm_center.y

        # Left radiator panel (inset from left edge)
        left_radiator = {
            "center": b2Vec2(-self.sm_half_w + radiator_width / 2, radiator_y_center),
            "half_w": radiator_width / 2,
            "half_h": radiator_height / 2,
        }
        self._draw_panel_rect(surface, left_radiator, RADIATOR_COLOR, SECTOR_LINE, 2,
                              cam_x, cam_y, screen_width, screen_height, ppm)

        # Right radiator panel (inset from right edge)
        right_radiator = {
            "center": b2Vec2(self.sm_half_w - radiator_width / 2, radiator_y_center),
            "half_w": radiator_width / 2,
            "half_h": radiator_height / 2,
        }
        self._draw_panel_rect(surface, right_radiator, RADIATOR_COLOR, SECTOR_LINE, 2,
                              cam_x, cam_y, screen_width, screen_height, ppm)

        # Draw radiator panel lines (simulate cooling tubes)
        num_lines = 8
        for i in range(num_lines):
            y_offset = -radiator_height / 2 + (i + 1) * (radiator_height / (num_lines + 1))

            # Left radiator lines
            left_p1 = b2Vec2(-self.sm_half_w, radiator_y_center + y_offset)
            left_p2 = b2Vec2(-self.sm_half_w + radiator_width, radiator_y_center + y_offset)
            p1_world = w2s(local_to_world(self.body, left_p1))
            p2_world = w2s(local_to_world(self.body, left_p2))
            pygame.draw.line(surface, SECTOR_LINE, p1_world, p2_world, 1)

            # Right radiator lines
            right_p1 = b2Vec2(self.sm_half_w - radiator_width, radiator_y_center + y_offset)
            right_p2 = b2Vec2(self.sm_half_w, radiator_y_center + y_offset)
            p1_world = w2s(local_to_world(self.body, right_p1))
            p2_world = w2s(local_to_world(self.body, right_p2))
            pygame.draw.line(surface, SECTOR_LINE, p1_world, p2_world, 1)

        # High-gain antenna removed - it's been detached for now

        # Draw S-band omni antennas (4 antennas in array near bottom, on the side)
        # Real SM had 4 omni antennas positioned as a cluster on the side near bottom
        antenna_array_y = sm_bottom + (sm_height * 0.25)  # 25% up from bottom
        antenna_array_x = self.sm_half_w  # On right side

        # 4 antennas in a rough cluster pattern
        omni_positions = [
            b2Vec2(antenna_array_x, antenna_array_y + 0.3 * s),  # Upper
            b2Vec2(antenna_array_x, antenna_array_y - 0.3 * s),  # Lower
            b2Vec2(antenna_array_x, antenna_array_y + 0.1 * s),  # Middle-upper
            b2Vec2(antenna_array_x, antenna_array_y - 0.1 * s),  # Middle-lower
        ]

        for pos in omni_positions:
            # Draw small antenna stub protruding from side
            antenna_tip = pos + b2Vec2(0.25 * s, 0.0)

            p1 = w2s(local_to_world(self.body, pos))
            p2 = w2s(local_to_world(self.body, antenna_tip))
            pygame.draw.line(surface, ANTENNA_COLOR, p1, p2, 2)

            # Small ball at tip
            tip_screen = w2s(local_to_world(self.body, antenna_tip))
            pygame.draw.circle(surface, ANTENNA_COLOR, tip_screen, int(3 * s), 0)
