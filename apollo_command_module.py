"""
Apollo Command Module (CM) - WorldObject Implementation

The Command Module as a standalone WorldObject with its own physics body.
Features docking port and hatch.
"""

import math
import pygame
from Box2D import b2PolygonShape, b2Vec2
from world import WorldObject


def rotate_vec(vec, angle):
    """Rotate a 2D vector by angle (radians)."""
    ca, sa = math.cos(angle), math.sin(angle)
    return b2Vec2(vec.x * ca - vec.y * sa, vec.x * sa + vec.y * ca)


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class CommandModule(WorldObject):
    """
    Apollo Command Module as a WorldObject.

    Features:
    - Conical body with realistic mass (~5,560 kg)
    - Docking port at apex
    - CM hatch
    - Independent physics body
    """

    def __init__(self, world, position, scale=1.2):
        """
        Initialize Command Module.

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
            angle=0.0,  # Upright orientation (heat shield at bottom)
            linearDamping=world.linear_damping,
            angularDamping=world.angular_damping,
        )

        # Command Module dimensions (real Apollo CM: 3.9m base diameter, 3.65m height)
        cm_base_half_w = (3.9 / 2.0) * scale  # 1.95m base radius
        cm_height = 3.65 * scale  # 3.65m height

        self.cm_base_y = 0.0
        self.cm_tip_y = cm_height
        self.cm_base_half_w = cm_base_half_w

        cm_vertices = [
            b2Vec2(-cm_base_half_w, self.cm_base_y),
            b2Vec2(+cm_base_half_w, self.cm_base_y),
            b2Vec2(0.0, self.cm_tip_y),
        ]

        # Mass settings via density
        # Real Apollo CM: ~5,560 kg at reentry
        cm_area = 2.0 * cm_base_half_w * cm_height * 0.5  # triangular
        REAL_CM_MASS_KG = 5560.0
        cm_density = REAL_CM_MASS_KG / cm_area

        # Create CM fixture
        self.fixture = self.body.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=cm_vertices),
            density=cm_density,
            friction=0.3,
            restitution=0.1,
        )

        # Docking tunnel and probe (at top of CM)
        # Real Apollo docking mechanism extends about 0.5m above CM apex
        tunnel_height = 0.5 * scale
        tunnel_width = 0.9 * scale

        # Docking tunnel (cylindrical section)
        self.docking_tunnel_center = b2Vec2(0.0, self.cm_tip_y + tunnel_height / 2)
        self.docking_tunnel_half_w = tunnel_width / 2.0
        self.docking_tunnel_half_h = tunnel_height / 2.0

        # Docking port location (for rendezvous calculations)
        self.docking_port_center = b2Vec2(0.0, self.cm_tip_y + tunnel_height)

        # Probe (extended drogue capture assembly)
        probe_length = 0.6 * scale
        probe_width = 0.3 * scale
        self.docking_probe_center = b2Vec2(0.0, self.cm_tip_y + tunnel_height + probe_length / 2)
        self.docking_probe_half_w = probe_width / 2.0
        self.docking_probe_half_h = probe_length / 2.0

        # CM windows (5 windows around cone at ~60% height)
        # Real Apollo CM had 5 windows positioned around the cone
        # Windows must be positioned INSIDE the cone boundaries
        window_height = self.cm_tip_y * 0.6  # 60% from base to tip

        # Calculate cone radius at window height (linear interpolation)
        # At base (y=0): radius = cm_base_half_w
        # At tip (y=cm_tip_y): radius = 0
        # At window_height: radius = cm_base_half_w * (1 - window_height/cm_tip_y)
        cone_radius_at_window = cm_base_half_w * (1 - window_height / self.cm_tip_y)

        self.windows = []
        num_windows = 5
        window_inset = 0.15 * scale  # Inset from cone edge
        window_radius = max(0.2 * scale, cone_radius_at_window - window_inset)  # Keep windows inside

        for i in range(num_windows):
            angle = (i * 2 * math.pi / num_windows) - math.pi / 2  # Start at front
            x_offset = window_radius * math.cos(angle)
            # Windows positioned on surface of cone, well within boundaries
            window_center = b2Vec2(x_offset, window_height)
            self.windows.append({
                "center": window_center,
                "half_w": 0.2 * scale,
                "half_h": 0.25 * scale,
            })

        # Side hatch (unified crew hatch on side of CM)
        # Position at mid-height and keep within cone boundaries
        hatch_height = self.cm_tip_y * 0.4  # 40% from base
        cone_radius_at_hatch = cm_base_half_w * (1 - hatch_height / self.cm_tip_y)
        hatch_x = max(0.3 * scale, cone_radius_at_hatch * 0.6)  # 60% of radius at that height

        self.side_hatch_rect = {
            "center": b2Vec2(hatch_x, hatch_height),
            "half_w": 0.35 * scale,
            "half_h": 0.45 * scale,
        }

        # Forward hatch (at base, centered)
        self.forward_hatch_rect = {
            "center": b2Vec2(0.0, self.cm_base_y + 0.5 * scale),
            "half_w": 0.35 * scale,
            "half_h": 0.35 * scale,
        }

        # Heat shield panels (divided sections on cone surface)
        self.heat_shield_divisions = 6  # Number of vertical divisions

        # RCS thruster blocks (small rectangular protrusions on CM sides)
        # Keep these INSIDE the cone boundaries
        rcs_height = self.cm_tip_y * 0.7  # 70% from base
        cone_radius_at_rcs = cm_base_half_w * (1 - rcs_height / self.cm_tip_y)
        rcs_x = cone_radius_at_rcs * 0.7  # 70% of radius at that height

        self.rcs_blocks = [
            {"center": b2Vec2(-rcs_x, rcs_height), "half_w": 0.12 * scale, "half_h": 0.15 * scale},
            {"center": b2Vec2(rcs_x, rcs_height), "half_w": 0.12 * scale, "half_h": 0.15 * scale},
        ]

    def update(self, dt):
        """Update CM physics."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw Command Module to screen."""
        PPM = self.world.ppm

        # Colors
        CM_COLOR = (220, 220, 220)  # Light gray for CM body
        OUTLINE = (0, 0, 0)
        DOCK_TUNNEL_COLOR = (80, 80, 90)  # Dark gray for docking tunnel
        DOCK_PROBE_COLOR = (60, 60, 70)  # Darker for probe
        WINDOW_COLOR = (40, 60, 100)  # Dark blue for windows
        HATCH_COLOR = (180, 180, 180)  # Mid gray for hatches
        RCS_BLOCK_COLOR = (160, 160, 160)  # Lighter gray for RCS blocks
        HEAT_SHIELD_LINE = (100, 100, 100)  # Gray for heat shield divisions

        def w2s(pos):
            """World to screen conversion."""
            x = screen_width / 2 + (pos.x - cam_x) * PPM
            y = screen_height / 2 - (pos.y - cam_y) * PPM
            return int(x), int(y)

        # Draw CM cone
        cm_world = [
            self.body.transform * v
            for v in self.fixture.shape.vertices
        ]
        pygame.draw.polygon(surface, CM_COLOR, [w2s(p) for p in cm_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in cm_world], 2)

        # Draw heat shield panel divisions (radial lines on cone)
        self._draw_heat_shield_divisions(surface, cam_x, cam_y, screen_width, screen_height, PPM, HEAT_SHIELD_LINE)

        # Draw windows
        for window in self.windows:
            self._draw_panel_rect(surface, window, WINDOW_COLOR, OUTLINE, 1,
                                  cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw hatches
        self._draw_panel_rect(surface, self.side_hatch_rect, HATCH_COLOR, OUTLINE, 2,
                              cam_x, cam_y, screen_width, screen_height, PPM)
        self._draw_panel_rect(surface, self.forward_hatch_rect, HATCH_COLOR, OUTLINE, 1,
                              cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw RCS thruster blocks
        for rcs_block in self.rcs_blocks:
            self._draw_panel_rect(surface, rcs_block, RCS_BLOCK_COLOR, OUTLINE, 1,
                                  cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw docking tunnel (cylindrical section)
        tunnel_rect = {
            "center": self.docking_tunnel_center,
            "half_w": self.docking_tunnel_half_w,
            "half_h": self.docking_tunnel_half_h,
        }
        self._draw_panel_rect(surface, tunnel_rect, DOCK_TUNNEL_COLOR, OUTLINE, 2,
                              cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw docking probe (extended capture mechanism)
        probe_rect = {
            "center": self.docking_probe_center,
            "half_w": self.docking_probe_half_w,
            "half_h": self.docking_probe_half_h,
        }
        self._draw_panel_rect(surface, probe_rect, DOCK_PROBE_COLOR, OUTLINE, 2,
                              cam_x, cam_y, screen_width, screen_height, PPM)

        # Draw probe tip (small circle at end)
        probe_tip_local = b2Vec2(0.0, self.cm_tip_y + 0.5 * self.scale + 0.6 * self.scale)
        probe_tip_world = local_to_world(self.body, probe_tip_local)
        probe_tip_screen = w2s(probe_tip_world)
        pygame.draw.circle(surface, DOCK_PROBE_COLOR, probe_tip_screen, int(4 * self.scale), 0)
        pygame.draw.circle(surface, OUTLINE, probe_tip_screen, int(4 * self.scale), 1)

    def _draw_panel_rect(self, surface, panel, fill, outline, line_width,
                         cam_x, cam_y, screen_width, screen_height, ppm):
        """Draw a rectangular panel on the CM."""
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

    def _draw_heat_shield_divisions(self, surface, cam_x, cam_y, screen_width, screen_height, ppm, line_color):
        """Draw heat shield panel division lines on the CM cone."""
        def w2s(pos):
            x = screen_width / 2 + (pos.x - cam_x) * ppm
            y = screen_height / 2 - (pos.y - cam_y) * ppm
            return int(x), int(y)

        # Draw radial division lines from base to apex
        for i in range(self.heat_shield_divisions):
            # Angle for this division line
            angle_fraction = i / self.heat_shield_divisions

            # Start at base edge
            base_x = self.cm_base_half_w * (1 - 2 * angle_fraction)  # -w to +w
            base_point_local = b2Vec2(base_x, self.cm_base_y)

            # End at apex
            apex_point_local = b2Vec2(0.0, self.cm_tip_y)

            # Transform to world coordinates
            base_world = local_to_world(self.body, base_point_local)
            apex_world = local_to_world(self.body, apex_point_local)

            # Draw line
            pygame.draw.line(surface, line_color, w2s(base_world), w2s(apex_world), 1)

        # Draw circumferential line at mid-height
        mid_height = self.cm_tip_y * 0.5
        # Calculate cone radius at mid-height (linearly interpolated)
        mid_radius = self.cm_base_half_w * (1 - 0.5)  # 50% up = 50% narrower

        # Draw line segments to approximate circle around cone
        num_segments = 8
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi

            p1_local = b2Vec2(mid_radius * math.cos(angle1), mid_height)
            p2_local = b2Vec2(mid_radius * math.cos(angle2), mid_height)

            p1_world = local_to_world(self.body, p1_local)
            p2_world = local_to_world(self.body, p2_local)

            pygame.draw.line(surface, line_color, w2s(p1_world), w2s(p2_world), 1)
