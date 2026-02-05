"""
Apollo Command/Service Module - WorldObject Implementation

The CSM as a standalone WorldObject.
Wraps the existing ApolloCSM class to integrate with the world architecture.
"""

from world import WorldObject
from apollocsm import ApolloCSM


class ApolloCSMObject(WorldObject):
    """
    Apollo Command/Service Module as a WorldObject.

    This is a thin wrapper around the existing ApolloCSM class
    to integrate it with the world-object architecture.

    Features:
    - Gimballed SPS main engine
    - RCS thrusters
    - Command Module with docking port
    - Service Module
    """

    def __init__(self, world, position, scale=0.75):
        """
        Initialize CSM.

        Args:
            world: ApolloWorld instance
            position: (x, y) tuple or b2Vec2 position in meters
            scale: Scale factor (default 0.75 to match lander)
        """
        super().__init__(world)

        # Convert position to tuple if it's a b2Vec2
        if hasattr(position, 'x') and hasattr(position, 'y'):
            position = (position.x, position.y)

        # Create the CSM using the existing class
        self.csm = ApolloCSM(world.b2world, position=position, scale=scale)

        # Expose the body for compatibility
        self.body = self.csm.body
        self.scale = scale

    def update(self, dt):
        """Update CSM physics."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw CSM to screen."""
        # Use the CSM's existing draw method
        self.csm.draw(surface, cam_x, cam_y, screen_width, screen_height)

    def apply_main_thrust(self, dt):
        """Apply main SPS thrust."""
        self.csm.apply_main_thrust(dt)

    def apply_rcs_thrust(self, thruster_names, dt):
        """Apply RCS thrust from specific thrusters."""
        self.csm.apply_rcs_thrust(thruster_names, dt)

    def set_nozzle_angle(self, angle):
        """Set SPS nozzle gimbal angle."""
        self.csm.set_nozzle_angle(angle)

    def adjust_nozzle(self, delta):
        """Adjust SPS nozzle gimbal by delta."""
        self.csm.adjust_nozzle(delta)

    def center_nozzle(self):
        """Reset SPS nozzle to center."""
        self.csm.center_nozzle()

    @property
    def docking_port_center(self):
        """Get docking port center position (local coordinates)."""
        return self.csm.docking_port_center

    @property
    def nozzle_gimbal_angle(self):
        """Get current nozzle gimbal angle."""
        return self.csm.nozzle_gimbal_angle
