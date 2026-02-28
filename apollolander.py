"""
Apollo Lunar Module - Physics and Rendering

Two-stage spacecraft with realistic Apollo LM characteristics:
- Descent stage with landing legs and variable throttle engine
- Ascent stage with RCS pods and engine
- Beautiful gold/gray rendering with detailed graphics
- All drawing functions for the lander
"""

import math
import pygame
from Box2D import (
    b2World,
    b2PolygonShape,
    b2CircleShape,
    b2Vec2,
    b2WeldJointDef,
)

# -------------------------------------------------
# Constants
# -------------------------------------------------
PPM = 22.0  # pixels per meter (for rendering)

# Thrust factors (multiplied by mass to get force)
DESCENT_MAIN_THRUST_FACTOR = 4.35
ASCENT_MAIN_THRUST_FACTOR  = 3.25

# Real-world masses (kg) - includes fuel at full capacity
# Descent: 2,034 kg dry + 8,200 kg fuel = 10,234 kg total
# Ascent: 2,445 kg dry + 2,353 kg fuel = 4,798 kg total
REAL_DESCENT_MASS_KG = 10234.0  # Full mass with fuel
REAL_ASCENT_MASS_KG = 4798.0    # Full mass with fuel
MASS_SCALE = 100.0  # 100 kg → 1 Box2D mass unit

# Real Apollo LM RCS: 445 N per thruster (real specification)
# Ascent mass: 4,798 kg → 445 / 4,798 = 0.0927 m/s² acceleration per thruster
REAL_LM_RCS_THRUST_N = 445.0  # Newtons per thruster (real Apollo spec)
RCS_THRUST_FACTOR = REAL_LM_RCS_THRUST_N / REAL_ASCENT_MASS_KG  # 0.0927 m/s²

# RCS pod geometry (local offsets, scaled by ascent scale)
RCS_OFFSET_X = 2.1
RCS_OFFSET_Y = 0.1
RCS_OFFSET_UP = 0.7

# Max gimbal angle for descent engine
MAX_GIMBAL_DEG = 6.0


# -------------------------------------------------
# Apollo Lander Class
# -------------------------------------------------
class ApolloLander:
    """
    Two-body Apollo-style Lunar Module (side view):
      - descent_stage: base with landing legs + engine bell
      - ascent_stage: octagonal cabin with RCS pods + recessed engine nozzle
    Connected with a weld joint. Call separate() to detach.
    """

    def __init__(
        self,
        world: b2World,
        position: b2Vec2,
        scale: float = 1.0,
        descent_density: float = 0.9,
        ascent_density: float = 0.6,
        leg_density: float = 0.4,
        friction: float = 0.9,
        restitution: float = 0.1,
        gravity: float = 1.62,
    ):
        self.world = world
        self.scale = scale

        self.descent_stage = self._create_descent_stage(
            world, position, scale,
            descent_density, leg_density,
            friction, restitution, gravity,
        )

        # Place ascent stage so hull touches descent module
        ascent_pos = b2Vec2(position.x, position.y + 1.85 * scale)

        self.ascent_stage = self._create_ascent_stage(
            world, ascent_pos, scale,
            ascent_density, friction, restitution,
        )

        # Apply mass scaling
        self._apply_mass_scaling(self.descent_stage, REAL_DESCENT_MASS_KG)
        self._apply_mass_scaling(self.ascent_stage, REAL_ASCENT_MASS_KG)

        # Weld joint connecting stages
        jd = b2WeldJointDef()
        anchor = (self.descent_stage.worldCenter + self.ascent_stage.worldCenter) * 0.5
        jd.Initialize(self.descent_stage, self.ascent_stage, anchor)
        jd.collideConnected = False
        self.connection_joint = world.CreateJoint(jd)

    def _apply_mass_scaling(self, body, real_mass_kg: float):
        """Rescale body mass to match real Apollo LM masses."""
        md = body.massData
        if md.mass <= 0:
            return
        target_mass = real_mass_kg / MASS_SCALE
        factor = target_mass / md.mass
        md.mass *= factor
        md.I *= factor
        body.massData = md

    def separate(self):
        """Destroy the weld joint so stages can fly independently."""
        if self.connection_joint is not None:
            self.world.DestroyJoint(self.connection_joint)
            self.connection_joint = None

            # Small upward impulse
            impulse_mag = self.ascent_stage.mass * 2.0
            up = self.ascent_stage.GetWorldVector((0, 1))
            self.ascent_stage.ApplyLinearImpulse(
                impulse_mag * up,
                self.ascent_stage.worldCenter,
                True,
            )

    def _create_descent_stage(
        self, world, position, s, body_density, leg_density, friction, restitution, gravity=1.62
    ):
        """Create descent stage with legs and engine bell."""
        body = world.CreateDynamicBody(
            position=position,
            angle=0.0,
            linearDamping=0.2,
            angularDamping=0.4,
            bullet=True,  # Enable continuous collision detection to prevent tunneling
        )

        data = {
            "type": "descent",
            "scale": s,
            "leg_segments": [],
            "foot_bars": [],
            "contact_probes": [],  # Lunar surface sensing probes below foot pads
        }
        body.userData = data

        # Main box
        main_half_w = (4.2 * 0.5) * s
        main_half_h = (1.6 * 0.5) * s

        main_shape = b2PolygonShape(box=(main_half_w, main_half_h))
        body.CreateFixture(
            shape=main_shape,
            density=body_density,
            friction=friction,
            restitution=restitution,
        )

        # Engine bell positioning
        leg_dir_tmp = b2Vec2(0.7, -1.0)
        leg_dir_tmp.Normalize()
        leg_length_tmp = 2.4 * s
        leg_top_y_tmp = main_half_h * 0.25
        leg_mid_y = leg_top_y_tmp + leg_dir_tmp.y * (2.0 / 3.0 * leg_length_tmp)
        foot_y_tmp = leg_top_y_tmp + leg_dir_tmp.y * leg_length_tmp

        bell_top_y = -main_half_h
        bell_bottom_y = 0.5 * (leg_mid_y + foot_y_tmp)

        bell_top_half_w = 0.45 * s
        bell_bottom_half_w = 0.9 * s

        bell_vertices = [
            b2Vec2(-bell_top_half_w, bell_top_y),
            b2Vec2(bell_top_half_w, bell_top_y),
            b2Vec2(bell_bottom_half_w, bell_bottom_y),
            b2Vec2(-bell_bottom_half_w, bell_bottom_y),
        ]
        bell_shape = b2PolygonShape(vertices=bell_vertices)
        body.CreateFixture(
            shape=bell_shape,
            density=body_density * 0.5,
            friction=friction,
            restitution=restitution,
        )
        data["bell_vertices"] = bell_vertices
        data["bell_base_local"] = b2Vec2(0.0, bell_bottom_y)

        # Helper for struts
        def add_strut(p1: b2Vec2, p2: b2Vec2, thickness: float, dens: float):
            mid = 0.5 * (p1 + p2)
            vec = p2 - p1
            length = vec.length
            if length <= 1e-5:
                return
            angle = math.atan2(vec.y, vec.x)
            half_len = 0.5 * length
            shape = b2PolygonShape(box=(half_len, thickness, mid, angle))
            body.CreateFixture(
                shape=shape,
                density=dens,
                friction=friction,
                restitution=restitution,
            )
            data["leg_segments"].append((p1, p2))

        STRUT_THICK = 0.09 * s

        # Landing legs (both sides)
        for side in (-1, 1):
            top_corner = b2Vec2(side * main_half_w,  main_half_h)
            bottom_corner = b2Vec2(side * main_half_w, -main_half_h)

            leg_top = b2Vec2(
                side * (main_half_w + 0.9 * s),
                main_half_h * 0.25,
            )

            leg_dir = b2Vec2(side * 0.7, -1.0)
            leg_dir.Normalize()
            leg_length = 2.4 * s
            foot = leg_top + leg_dir * leg_length

            leg_mid = leg_top + leg_dir * (leg_length * (2.0 / 3.0))

            # Braces and leg
            add_strut(top_corner, leg_top, STRUT_THICK, leg_density)
            add_strut(bottom_corner, leg_top, STRUT_THICK, leg_density)
            add_strut(leg_top, foot, STRUT_THICK, leg_density)
            add_strut(bottom_corner, leg_mid, STRUT_THICK, leg_density)

            # Foot pad
            bar_half_w = 0.45 * s
            bar_half_h = 0.12 * s
            foot_shape = b2PolygonShape(box=(bar_half_w, bar_half_h, foot, 0.0))
            body.CreateFixture(
                shape=foot_shape,
                density=leg_density * 0.7,
                friction=1.4,
                restitution=0.0,
            )
            data["foot_bars"].append((foot, bar_half_w, bar_half_h))

            # Contact probe - extends below foot pad (like real Apollo 67-inch probes)
            # Length is scaled by lunar gravity ratio so post-cutoff free-fall acceleration
            # is gravity-normalized: shorter probes on high-g bodies prevent over-speed at touchdown.
            # Capped at 0.3*s minimum so probes remain visible and functional on very high-g worlds.
            probe_length = max(1.7 * s * (1.62 / gravity), 0.3 * s)
            probe_attach = b2Vec2(foot.x, foot.y - bar_half_h)  # Bottom of foot pad
            data["contact_probes"].append((probe_attach, probe_length, side))

        data["main_half_w"] = main_half_w
        data["main_half_h"] = main_half_h

        return body

    def _create_ascent_stage(
        self, world, position, s, density, friction, restitution
    ):
        """Create ascent stage with RCS pods."""
        body = world.CreateDynamicBody(
            position=position,
            angle=0.0,
            linearDamping=0.1,
            angularDamping=0.25,
            bullet=True,  # Enable continuous collision detection to prevent tunneling
        )

        h = 1.5 * s
        w = 2.0 * s

        # Octagonal hull
        hull_vertices = [
            b2Vec2(-w,       -h * 0.3),
            b2Vec2(-w * 0.4, -h),
            b2Vec2(w * 0.4,  -h),
            b2Vec2(w,        -h * 0.3),
            b2Vec2(w,         h * 0.3),
            b2Vec2(w * 0.4,   h),
            b2Vec2(-w * 0.4,  h),
            b2Vec2(-w,        h * 0.3),
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

        # RCS pods
        rcs_half_w = 0.35 * s
        rcs_half_h = 0.25 * s
        rcs_y = RCS_OFFSET_Y * s
        rcs_center_offset_x = w + rcs_half_w

        for side in (-1, 1):
            center = b2Vec2(side * rcs_center_offset_x, rcs_y)
            rcs_shape = b2PolygonShape(
                box=(rcs_half_w, rcs_half_h, center, 0.0)
            )
            body.CreateFixture(
                shape=rcs_shape,
                density=density * 0.4,
                friction=friction,
                restitution=restitution,
            )

        body.userData = {
            "type": "ascent",
            "scale": s,
            "rcs_half_w": rcs_half_w,
            "rcs_half_h": rcs_half_h,
            "rcs_center_y": rcs_y,
            "rcs_center_offset_x": rcs_center_offset_x,
            "bell_vertices": ascent_bell_vertices,
            "bell_base_local": bell_base_local,
            "hull_w": w,
            "hull_h": h,
        }

        return body


# -------------------------------------------------
# Rendering Functions
# -------------------------------------------------
def world_to_screen(pos: b2Vec2, cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Convert world coordinates to screen coordinates with camera support."""
    x = screen_width / 2 + (pos.x - cam_x) * PPM
    # Support vertical scrolling: center screen on cam_y
    y = screen_height / 2 - (pos.y - cam_y) * PPM
    return int(x), int(y)


def draw_ascent_front_details(surface, body, data, cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Draw visual details on ascent module."""
    s = data["scale"]
    w = data["hull_w"]
    h = data["hull_h"]

    line_color = (140, 140, 140)
    hatch_color = (110, 110, 110)
    window_color = (20, 20, 20)
    mast_color = (240, 240, 240)

    # Diverging lines
    top_y = h * 0.95
    bot_y = -h * 0.75
    left_top = b2Vec2(-0.4 * s, top_y)
    left_bot = b2Vec2(-0.9 * s, bot_y)
    right_top = b2Vec2(0.4 * s, top_y)
    right_bot = b2Vec2(0.9 * s, bot_y)

    lt_w = world_to_screen(body.transform * left_top, cam_x, screen_width, screen_height, cam_y)
    lb_w = world_to_screen(body.transform * left_bot, cam_x, screen_width, screen_height, cam_y)
    rt_w = world_to_screen(body.transform * right_top, cam_x, screen_width, screen_height, cam_y)
    rb_w = world_to_screen(body.transform * right_bot, cam_x, screen_width, screen_height, cam_y)

    pygame.draw.line(surface, line_color, lt_w, lb_w, 2)
    pygame.draw.line(surface, line_color, rt_w, rb_w, 2)

    # Hatch
    hatch_center = b2Vec2(0.0, -h * 0.25)
    hatch_half_w = 0.55 * s
    hatch_half_h = 0.35 * s
    hatch_verts = [
        hatch_center + b2Vec2(-hatch_half_w, -hatch_half_h),
        hatch_center + b2Vec2(hatch_half_w, -hatch_half_h),
        hatch_center + b2Vec2(hatch_half_w, hatch_half_h),
        hatch_center + b2Vec2(-hatch_half_w, hatch_half_h),
    ]
    hatch_pts = [world_to_screen(body.transform * v, cam_x, screen_width, screen_height, cam_y) for v in hatch_verts]
    pygame.draw.polygon(surface, hatch_color, hatch_pts, 0)

    # Windows
    win_y_mid = h * 0.25
    win_height = 0.6 * s
    left_outer = b2Vec2(-w * 0.9, win_y_mid)
    left_inner_top = b2Vec2(-0.55 * s, win_y_mid + win_height / 2)
    left_inner_bot = b2Vec2(-0.55 * s, win_y_mid - win_height / 2)
    left_tri = [left_outer, left_inner_top, left_inner_bot]
    left_tri_pts = [world_to_screen(body.transform * v, cam_x, screen_width, screen_height, cam_y) for v in left_tri]
    pygame.draw.polygon(surface, window_color, left_tri_pts, 0)

    right_outer = b2Vec2(w * 0.9, win_y_mid)
    right_inner_top = b2Vec2(0.55 * s, win_y_mid + win_height / 2)
    right_inner_bot = b2Vec2(0.55 * s, win_y_mid - win_height / 2)
    right_tri = [right_outer, right_inner_bot, right_inner_top]
    right_tri_pts = [world_to_screen(body.transform * v, cam_x, screen_width, screen_height, cam_y) for v in right_tri]
    pygame.draw.polygon(surface, window_color, right_tri_pts, 0)

    # Dish antenna
    dish_base = b2Vec2(0.25 * s, h * 1.05)
    dish_tip = b2Vec2(0.25 * s, h * 1.35)
    base_world = world_to_screen(body.transform * dish_base, cam_x, screen_width, screen_height, cam_y)
    tip_world = world_to_screen(body.transform * dish_tip, cam_x, screen_width, screen_height, cam_y)

    pygame.draw.line(surface, mast_color, base_world, tip_world, 2)

    dish_center_local = dish_tip + b2Vec2(0.0, 0.35 * s)
    dish_center_screen = world_to_screen(body.transform * dish_center_local, cam_x, screen_width, screen_height, cam_y)

    radius = int(8 * s)
    pygame.draw.circle(surface, (255, 255, 255), dish_center_screen, radius, 0)

    cx, cy = dish_center_screen
    arm = int(radius * 0.5)
    pygame.draw.line(surface, (0, 0, 0), (cx - arm, cy), (cx + arm, cy), 1)
    pygame.draw.line(surface, (0, 0, 0), (cx, cy - arm), (cx, cy + arm), 1)


def draw_descent_front_details(surface, body, data, cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Draw ladder and leg details on descent module."""
    s = data["scale"]
    main_half_w = data["main_half_w"]
    main_half_h = data["main_half_h"]

    ladder_color = (0, 0, 0)

    ladder_half_w = 0.35 * s
    ladder_top_y = main_half_h * 0.9
    ladder_bottom_y = -main_half_h + 0.15 * s
    ladder_height = ladder_top_y - ladder_bottom_y

    left_local = b2Vec2(-ladder_half_w, ladder_bottom_y)
    right_local = b2Vec2(ladder_half_w, ladder_bottom_y)
    left_top_local = b2Vec2(-ladder_half_w, ladder_top_y)
    right_top_local = b2Vec2(ladder_half_w, ladder_top_y)

    # Rails
    pygame.draw.line(
        surface,
        ladder_color,
        world_to_screen(body.transform * left_local, cam_x, screen_width, screen_height),
        world_to_screen(body.transform * left_top_local, cam_x, screen_width, screen_height),
        2,
    )
    pygame.draw.line(
        surface,
        ladder_color,
        world_to_screen(body.transform * right_local, cam_x, screen_width, screen_height),
        world_to_screen(body.transform * right_top_local, cam_x, screen_width, screen_height),
        2,
    )

    # Rungs
    rung_count = 6
    step = ladder_height / (rung_count - 1)
    for i in range(rung_count):
        y = ladder_bottom_y + i * step
        l = world_to_screen(body.transform * b2Vec2(-ladder_half_w, y), cam_x, screen_width, screen_height)
        r = world_to_screen(body.transform * b2Vec2(ladder_half_w, y), cam_x, screen_width, screen_height)
        pygame.draw.line(surface, ladder_color, l, r, 2)

    # Leg segments
    for p1_local, p2_local in data.get("leg_segments", []):
        p1 = world_to_screen(body.transform * p1_local, cam_x, screen_width, screen_height)
        p2 = world_to_screen(body.transform * p2_local, cam_x, screen_width, screen_height)
        pygame.draw.line(surface, (0, 0, 0), p1, p2, 1)

    # Contact probes (lunar surface sensing probes)
    # Probes always hang straight down regardless of lander orientation
    probe_color = (180, 180, 180)  # Light gray - more visible
    for probe_attach, probe_length, side in data.get("contact_probes", []):
        # Transform attachment point to world coordinates
        attach_world = body.transform * probe_attach
        # Probe hangs straight down from attachment point
        probe_tip_world = b2Vec2(attach_world.x, attach_world.y - probe_length)
        p1 = world_to_screen(attach_world, cam_x, screen_width, screen_height, cam_y)
        p2 = world_to_screen(probe_tip_world, cam_x, screen_width, screen_height, cam_y)
        pygame.draw.line(surface, probe_color, p1, p2, 2)  # Thicker line for visibility


def _vec_to_tuple(v):
    """Convert Box2D vector to tuple."""
    if isinstance(v, tuple):
        return v
    return (v.x, v.y)


def _verts_match(shape_verts, bell_verts, eps=1e-6):
    """Check if vertices match."""
    if len(shape_verts) != len(bell_verts):
        return False
    for sv, bv in zip(shape_verts, bell_verts):
        sx, sy = _vec_to_tuple(sv)
        bx, by = _vec_to_tuple(bv)
        dx = sx - bx
        dy = sy - by
        if dx * dx + dy * dy > eps:
            return False
    return True


def draw_body(surface, body, color, cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Draw a lander body with all fixtures."""
    data = getattr(body, "userData", None)
    btype = data.get("type") if data else None

    # Ascent: draw nozzle behind hull
    if btype == "ascent" and data and "bell_vertices" in data:
        verts = [body.transform * v for v in data["bell_vertices"]]
        pts = [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in verts]
        pygame.draw.polygon(surface, (50, 50, 60), pts, 0)

        bell_verts = data["bell_vertices"]

        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2PolygonShape):
                if _verts_match(shape.vertices, bell_verts):
                    continue
                vertices = [body.transform * v for v in shape.vertices]
                pts = [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in vertices]
                pygame.draw.polygon(surface, color, pts, 0)
            elif isinstance(shape, b2CircleShape):
                world_pos = body.transform * shape.pos
                pygame.draw.circle(
                    surface,
                    color,
                    world_to_screen(world_pos, cam_x, screen_width, screen_height),
                    int(shape.radius * PPM),
                    0,
                )

        draw_ascent_front_details(surface, body, data, cam_x, screen_width, screen_height, cam_y)
        return

    # Generic drawing
    for fixture in body.fixtures:
        shape = fixture.shape
        if isinstance(shape, b2CircleShape):
            world_pos = body.transform * shape.pos
            pygame.draw.circle(
                surface,
                color,
                world_to_screen(world_pos, cam_x, screen_width, screen_height),
                int(shape.radius * PPM),
                0,
            )
        elif isinstance(shape, b2PolygonShape):
            vertices = [body.transform * v for v in shape.vertices]
            pts = [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in vertices]
            pygame.draw.polygon(surface, color, pts, 0)

    # Descent overlays
    if data and btype == "descent":
        if "bell_vertices" in data:
            verts = [body.transform * v for v in data["bell_vertices"]]
            pts = [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in verts]
            pygame.draw.polygon(surface, (70, 70, 80), pts, 0)
        draw_descent_front_details(surface, body, data, cam_x, screen_width, screen_height, cam_y)

    # Ascent overlays
    if data and btype == "ascent":
        draw_ascent_front_details(surface, body, data, cam_x, screen_width, screen_height, cam_y)


def draw_rcs_pods(surface, body, cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Draw RCS pods with nozzles."""
    data = getattr(body, "userData", None)
    if not data or data.get("type") != "ascent":
        return

    s = data["scale"]
    rcs_half_w = data["rcs_half_w"]
    rcs_half_h = data["rcs_half_h"]
    rcs_center_y = data["rcs_center_y"]
    rcs_center_offset_x = data["rcs_center_offset_x"]

    pod_color = (110, 110, 110)
    nozzle_color = (180, 180, 180)

    noz_len = 0.35 * s
    noz_base_half = 0.18 * s

    for side in (-1, 1):
        center_local = b2Vec2(side * rcs_center_offset_x, rcs_center_y)

        # Pod rectangle
        corners_local = [
            center_local + b2Vec2(-rcs_half_w, -rcs_half_h),
            center_local + b2Vec2(rcs_half_w, -rcs_half_h),
            center_local + b2Vec2(rcs_half_w,  rcs_half_h),
            center_local + b2Vec2(-rcs_half_w, rcs_half_h),
        ]
        corners_world = [body.transform * v for v in corners_local]
        pygame.draw.polygon(
            surface, pod_color, [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in corners_world], 0
        )

        # Nozzles
        tip_up = center_local + b2Vec2(0.0, rcs_half_h)
        base_mid_up = center_local + b2Vec2(0.0, rcs_half_h + noz_len)
        up_top = base_mid_up + b2Vec2(-noz_base_half, 0.0)
        up_bottom = base_mid_up + b2Vec2(noz_base_half, 0.0)
        up_tri = [tip_up, up_top, up_bottom]

        tip_down = center_local + b2Vec2(0.0, -rcs_half_h)
        base_mid_down = center_local + b2Vec2(0.0, -rcs_half_h - noz_len)
        down_top = base_mid_down + b2Vec2(-noz_base_half, 0.0)
        down_bottom = base_mid_down + b2Vec2(noz_base_half, 0.0)
        down_tri = [tip_down, down_bottom, down_top]

        if side == -1:
            tip_side = center_local + b2Vec2(-rcs_half_w, 0.0)
            base_mid_side = center_local + b2Vec2(-rcs_half_w - noz_len, 0.0)
        else:
            tip_side = center_local + b2Vec2(rcs_half_w, 0.0)
            base_mid_side = center_local + b2Vec2(rcs_half_w + noz_len, 0.0)

        side_top = base_mid_side + b2Vec2(0.0, noz_base_half)
        side_bottom = base_mid_side + b2Vec2(0.0, -noz_base_half)
        side_tri = [tip_side, side_bottom, side_top]

        for tri in (up_tri, down_tri, side_tri):
            tri_world = [body.transform * v for v in tri]
            pygame.draw.polygon(
                surface, nozzle_color,
                [world_to_screen(v, cam_x, screen_width, screen_height, cam_y) for v in tri_world],
                0,
            )


def _draw_flame_triangle(surface, body, base_local: b2Vec2, dir_local: b2Vec2,
                         length=0.8, width=0.25, color=(255, 180, 80),
                         cam_x=0.0, screen_width=1024, screen_height=768, cam_y=0.0):
    """Draw a flame triangle."""
    base_world = body.GetWorldPoint(base_local)
    dir_world = body.GetWorldVector(dir_local)
    tip = base_world + dir_world * length
    perp = b2Vec2(-dir_world.y, dir_world.x)
    side1 = base_world + perp * width
    side2 = base_world - perp * width
    pts = [world_to_screen(p, cam_x, screen_width, screen_height, cam_y) for p in (tip, side1, side2)]
    pygame.draw.polygon(surface, color, pts, 0)


def draw_thrusters(
    surface,
    body,
    is_ascent,
    main_on,
    tl, bl, tr, br,
    sl, sr,
    gimbal_angle_deg=0.0,
    cam_x=0.0,
    screen_width=1024,
    screen_height=768,
    cam_y=0.0,
    throttle=1.0,
):
    """Draw flames for main engine and RCS thrusters.

    Args:
        throttle: Throttle level (0.0 to 1.0) that scales flame length
    """
    user_data = getattr(body, "userData", {}) or {}
    scale = user_data.get("scale", 1.0)

    # Main engine flame
    if main_on:
        if is_ascent and "bell_base_local" in user_data:
            base_local = user_data["bell_base_local"]
            _draw_flame_triangle(
                surface, body,
                base_local=base_local,
                dir_local=b2Vec2(0.0, -1.0),
                length=0.9 * throttle,
                width=0.5 * throttle,
                cam_x=cam_x,
                screen_width=screen_width,
                screen_height=screen_height,
                cam_y=cam_y,
            )
        else:
            if user_data.get("type") == "descent" and "bell_base_local" in user_data:
                base_local = user_data["bell_base_local"]
                length = 1.2 * throttle
                width = 0.6 * throttle

                rad = math.radians(gimbal_angle_deg)
                dir_local = b2Vec2(-math.sin(rad), -math.cos(rad))
            else:
                offset = 2.0 if user_data.get("type") == "descent" else 1.3
                base_local = b2Vec2(0.0, -offset)
                length = (1.2 if not is_ascent else 0.9) * throttle
                width = 0.6 * throttle
                dir_local = b2Vec2(0.0, -1.0)

            _draw_flame_triangle(
                surface, body,
                base_local=base_local,
                dir_local=dir_local,
                length=length,
                width=width,
                cam_x=cam_x,
                screen_width=screen_width,
                screen_height=screen_height,
                cam_y=cam_y,
            )

    if not is_ascent:
        return

    # RCS thrusters
    rcs_half_w = user_data["rcs_half_w"]
    rcs_half_h = user_data["rcs_half_h"]
    rcs_center_y = user_data["rcs_center_y"]
    rcs_center_offset_x = user_data["rcs_center_offset_x"]

    noz_len = 0.35 * scale

    def base_and_dir(thruster_name: str):
        if thruster_name == "tl":
            center = b2Vec2(-rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(0.0, rcs_half_h + noz_len)
            direction = b2Vec2(0.0, 1.0)
        elif thruster_name == "bl":
            center = b2Vec2(-rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(0.0, -rcs_half_h - noz_len)
            direction = b2Vec2(0.0, -1.0)
        elif thruster_name == "tr":
            center = b2Vec2(rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(0.0, rcs_half_h + noz_len)
            direction = b2Vec2(0.0, 1.0)
        elif thruster_name == "br":
            center = b2Vec2(rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(0.0, -rcs_half_h - noz_len)
            direction = b2Vec2(0.0, -1.0)
        elif thruster_name == "sl":
            # Left pod side thruster: flame goes LEFT (outward from spacecraft)
            # Nozzle on the LEFT side of the left pod, flame shoots LEFT
            center = b2Vec2(-rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(-rcs_half_w - noz_len, 0.0)  # Nozzle on LEFT side of left pod
            direction = b2Vec2(-1.0, 0.0)  # Flame goes LEFT (outward)
        elif thruster_name == "sr":
            # Right pod side thruster: flame goes RIGHT (outward from spacecraft)
            # Nozzle on the RIGHT side of the right pod, flame shoots RIGHT
            center = b2Vec2(rcs_center_offset_x, rcs_center_y)
            base = center + b2Vec2(rcs_half_w + noz_len, 0.0)  # Nozzle on RIGHT side of right pod
            direction = b2Vec2(1.0, 0.0)  # Flame goes RIGHT (outward)
        else:
            base = b2Vec2(0.0, 0.0)
            direction = b2Vec2(0.0, 1.0)
        return base, direction

    if tl:
        base, direction = base_and_dir("tl")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)
    if bl:
        base, direction = base_and_dir("bl")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)
    if tr:
        base, direction = base_and_dir("tr")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)
    if br:
        base, direction = base_and_dir("br")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)
    if sl:
        base, direction = base_and_dir("sl")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)
    if sr:
        base, direction = base_and_dir("sr")
        _draw_flame_triangle(surface, body, base, direction, 0.8, 0.22,
                           cam_x=cam_x, screen_width=screen_width, screen_height=screen_height, cam_y=cam_y)


def check_contact_probes(descent_body, terrain_height_func, world_width=None):
    """
    Check if any contact probe is touching or below the terrain.
    Probes always hang straight down regardless of lander orientation.

    Args:
        descent_body: Box2D body of the descent stage
        terrain_height_func: Function that takes x coordinate and returns terrain height
        world_width: World width for cylindrical wrapping (optional)

    Returns:
        bool: True if any probe has made contact with terrain
    """
    data = getattr(descent_body, "userData", None)
    if not data or data.get("type") != "descent":
        return False

    for probe_attach, probe_length, side in data.get("contact_probes", []):
        # Transform attachment point to world coordinates
        attach_world = descent_body.transform * probe_attach
        # Probe tip is straight down from attachment point
        probe_tip_y = attach_world.y - probe_length

        # Get terrain height at probe x position (with wrapping if needed)
        probe_x = attach_world.x
        if world_width is not None:
            probe_x = probe_x % world_width

        terrain_height = terrain_height_func(probe_x)

        # Check if probe tip is at or below terrain
        if probe_tip_y <= terrain_height:
            return True

    return False
