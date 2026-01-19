"""
Apollo Command and Service Module (CSM) - Modular version

Uses separate ServiceModule and CommandModule components for better code organization.
Maintains backwards compatibility with existing code.
"""

import math
from Box2D import b2Vec2
from apollo_service_module import ServiceModule
from apollo_command_module import CommandModule


# CSM-specific thrust values (real Apollo specifications)
CSM_MAIN_THRUST = 91190.0  # Service Propulsion System (SPS) thrust in Newtons
CSM_RCS_THRUST = 445.0  # RCS thruster force in Newtons


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class ApolloCSM:
    """
    Apollo Command and Service Module with modular architecture.

    Features:
    - Gimballed SPS main engine (±6°)
    - RCS thrusters for translation and rotation
    - Realistic mass distribution (SM + CM)
    - Docking port at apex of Command Module
    """

    def __init__(self, world, position=(10, 10), scale=0.75):
        """
        Initialize CSM in the physics world.

        Args:
            world: Box2D world instance
            position: (x, y) spawn position in meters
            scale: Scale factor for CSM size (default 0.75 to match lander)
        """
        self.scale = scale

        # Create single Box2D body for both modules
        self.body = world.CreateDynamicBody(
            position=position,
            angle=math.pi / 2.0,  # Rotated 90 degrees counter-clockwise (pointing left)
            linearDamping=0.0,
            angularDamping=0.5,
        )

        # Create Service Module component (attaches fixtures to body)
        self.service_module = ServiceModule(self.body, scale=scale)

        # Create Command Module component (attaches fixtures to body)
        # CM sits on top of SM, so we pass the SM's top_y position
        self.command_module = CommandModule(
            self.body,
            sm_top_y=self.service_module.top_y,
            scale=scale
        )

        # Expose commonly used properties for backwards compatibility
        self.active_thrusters = set()
        self.thrusting = False

        # Expose fixture references for backwards compatibility
        self.service_module_fixture = self.service_module.fixture
        self.command_module_fixture = self.command_module.fixture

    # -------------------------------------------------
    # Gimbal helpers (delegate to Service Module)
    # -------------------------------------------------
    def set_nozzle_angle(self, angle):
        """Set nozzle gimbal angle (radians), clamped to ±max_gimbal."""
        self.service_module.set_nozzle_angle(angle)

    def adjust_nozzle(self, delta):
        """Adjust nozzle gimbal by delta radians."""
        self.service_module.adjust_nozzle(delta)

    def center_nozzle(self):
        """Reset nozzle gimbal to center (0°)."""
        self.service_module.center_nozzle()

    @property
    def nozzle_gimbal_angle(self):
        """Get current nozzle gimbal angle."""
        return self.service_module.nozzle_gimbal_angle

    @property
    def max_gimbal(self):
        """Get maximum gimbal angle."""
        return self.service_module.max_gimbal

    @property
    def nozzle_attach_center(self):
        """Get nozzle attachment point."""
        return self.service_module.nozzle_attach_center

    # -------------------------------------------------
    # Docking port (delegate to Command Module)
    # -------------------------------------------------
    @property
    def docking_port_center(self):
        """Get docking port center position (local coordinates)."""
        return self.command_module.docking_port_center

    # -------------------------------------------------
    # Physics / controls
    # -------------------------------------------------
    def apply_main_thrust(self, dt):
        """Apply main SPS thrust with gimbal angle."""
        from apollo_service_module import rotate_vec

        # Thrust along +Y in engine frame, rotated by gimbal relative to body
        local_engine_dir = rotate_vec(b2Vec2(0.0, 1.0), self.service_module.nozzle_gimbal_angle)
        world_dir = rotate_vec(local_engine_dir, self.body.angle)
        force = CSM_MAIN_THRUST * world_dir

        # Apply at nozzle attach point to get realistic torque
        attach_world = local_to_world(self.body, self.service_module.nozzle_attach_center)
        self.body.ApplyForce(force, attach_world, wake=True)

    def clear_rcs_activity(self):
        """Clear all active RCS thrusters (call once per frame)."""
        self.active_thrusters.clear()

    def _fire_thruster(self, name):
        """Fire a specific RCS thruster by name."""
        from apollo_service_module import rotate_vec

        t = self.service_module.rcs_thrusters[name]
        pos_world = local_to_world(self.body, t["pos"])
        dir_world = rotate_vec(t["dir"], self.body.angle)
        force = CSM_RCS_THRUST * dir_world
        self.body.ApplyForce(force, pos_world, wake=True)
        self.active_thrusters.add(name)

    def rcs_translate_left(self):
        """Fire RCS to translate left."""
        self._fire_thruster("R_SIDE")

    def rcs_translate_right(self):
        """Fire RCS to translate right."""
        self._fire_thruster("L_SIDE")

    def rcs_roll_ccw(self):
        """Fire RCS to roll counter-clockwise."""
        self._fire_thruster("L_UP")
        self._fire_thruster("R_DOWN")

    def rcs_roll_cw(self):
        """Fire RCS to roll clockwise."""
        self._fire_thruster("L_DOWN")
        self._fire_thruster("R_UP")

    # -------------------------------------------------
    # Docking check
    # -------------------------------------------------
    def is_docked(self, lander_body, tolerance=5.0, angle_tolerance=None):
        """
        Check if lander has docked with CSM.

        Args:
            lander_body: b2Body of the lander (ascent module)
            tolerance: Docking distance tolerance in meters (default 5.0m)
            angle_tolerance: Angular alignment tolerance in degrees (None = no angle check)

        Returns:
            True if docking port indicators overlap (distance <= tolerance)
        """
        # Check distance between docking ports
        docking_port_world = local_to_world(self.body, self.docking_port_center)

        # Calculate lander's docking port position
        lander_h = 1.5 * 0.5625  # lander height * scale
        lander_dock_local = b2Vec2(0.0, lander_h * 1.1)
        lander_dock_world = local_to_world(lander_body, lander_dock_local)

        distance = (lander_dock_world - docking_port_world).length
        return distance <= tolerance

    # -------------------------------------------------
    # Drawing
    # -------------------------------------------------
    def draw(self, surface, cam_x, cam_y, screen_width, screen_height,
             main_on=False, tl=False, tr=False, bl=False, br=False, sl=False, sr=False):
        """
        Draw CSM with all components and optional thruster effects.

        Args:
            surface: Pygame surface
            cam_x: Camera X position in world coordinates
            cam_y: Camera Y position in world coordinates
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            main_on: Main SPS engine firing
            tl, tr, bl, br: RCS thruster flags (top-left, top-right, bottom-left, bottom-right)
            sl, sr: RCS side thrusters (left, right)
        """
        PPM = 22.0  # Pixels per meter

        # Map thruster flags to thruster names for active RCS set
        active_rcs = set()
        if tl:
            active_rcs.add("L_UP")
        if tr:
            active_rcs.add("R_UP")
        if bl:
            active_rcs.add("L_DOWN")
        if br:
            active_rcs.add("R_DOWN")
        if sl:
            active_rcs.add("L_SIDE")
        if sr:
            active_rcs.add("R_SIDE")

        # Draw Service Module (includes SPS nozzle and RCS pods)
        self.service_module.draw(
            surface, cam_x, cam_y, screen_width, screen_height,
            ppm=PPM, main_on=main_on, active_rcs=active_rcs
        )

        # Draw Command Module (includes docking port and hatch)
        self.command_module.draw(
            surface, cam_x, cam_y, screen_width, screen_height,
            ppm=PPM
        )

    def apply_rcs_thrust(self, thruster_names, dt):
        """
        Apply RCS thrust from specific thrusters.

        Args:
            thruster_names: List of thruster names to fire
            dt: Time step (not used, kept for compatibility)
        """
        for name in thruster_names:
            self._fire_thruster(name)
