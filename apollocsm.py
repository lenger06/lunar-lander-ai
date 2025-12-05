"""
Apollo Command and Service Module (CSM) - Physics-enabled version

Based on the CSM test code, adapted for integration with apollolandergame.py
Includes full physics body, RCS control, and gimballed main engine.
"""

import math
import pygame
from Box2D import (
    b2PolygonShape,
    b2Vec2,
)


# Constants from apollolander.py
PPM = 22.0  # pixels per meter (matches apollolander.py)

# CSM-specific thrust values (real Apollo specifications)
CSM_MAIN_THRUST = 91190.0  # Service Propulsion System (SPS) thrust in Newtons (91.19 kN)
CSM_RCS_THRUST = 445.0  # RCS thruster force in Newtons (100 lbf, same as LM)


def rotate_vec(vec, angle):
    """Rotate a 2D vector by angle (radians)."""
    ca, sa = math.cos(angle), math.sin(angle)
    return b2Vec2(vec.x * ca - vec.y * sa, vec.x * sa + vec.y * ca)


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class ApolloCSM:
    """
    Apollo Command and Service Module with full physics.

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

        self.body = world.CreateDynamicBody(
            position=position,
            angle=math.pi / 2.0,  # Rotated 90 degrees counter-clockwise (pointing left)
            linearDamping=0.0,  # No damping for constant velocity
            angularDamping=0.5,
        )

        # -------------------------
        # Service Module (scaled)
        # -------------------------
        self.sm_half_w = 1.2 * scale
        self.sm_half_h = 2.0 * scale
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

        # -------------------------
        # Command Module (scaled)
        # -------------------------
        cm_base_y = sm_top_y
        cm_tip_y = cm_base_y + 1.6 * scale
        self.cm_base_y = cm_base_y
        self.cm_tip_y = cm_tip_y
        self.cm_base_half_w = 1.0 * scale

        cm_vertices = [
            b2Vec2(-self.cm_base_half_w, cm_base_y),
            b2Vec2(+self.cm_base_half_w, cm_base_y),
            b2Vec2(0.0, cm_tip_y),
        ]

        # Mass settings via density (scaled by area)
        # Real Apollo CSM: ~30,080 kg total (SM: 24,520 kg + CM: 5,560 kg)
        sm_area = (2.0 * self.sm_half_w) * (2.0 * self.sm_half_h)
        cm_area = 2.0 * self.cm_base_half_w * (cm_tip_y - cm_base_y) * 0.5  # triangular

        # Real Apollo masses
        REAL_SM_MASS_KG = 24520.0  # Service Module fully fueled
        REAL_CM_MASS_KG = 5560.0   # Command Module at reentry

        sm_density = REAL_SM_MASS_KG / sm_area
        cm_density = REAL_CM_MASS_KG / cm_area

        self.service_module_fixture = self.body.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=sm_vertices),
            density=sm_density,
            friction=0.3,
        )
        self.command_module_fixture = self.body.CreatePolygonFixture(
            shape=b2PolygonShape(vertices=cm_vertices),
            density=cm_density,
            friction=0.3,
        )

        # -------------------------
        # Main engine nozzle (below SM) – gimballed SPS
        # -------------------------
        nozzle_attach_y = sm_bottom_y
        attach_half_w = 0.6 * scale
        exit_half_w = 1.1 * scale
        nozzle_length = 1.6 * scale

        # Pivot point (throat region), in body-local coordinates
        self.nozzle_attach_center = b2Vec2(0.0, nozzle_attach_y)

        # Canonical nozzle shape, built pointing straight down in LOCAL frame
        self.nozzle_shape_local = [
            b2Vec2(-attach_half_w, 0.0),
            b2Vec2(+attach_half_w, 0.0),
            b2Vec2(+exit_half_w, -nozzle_length),
            b2Vec2(-exit_half_w, -nozzle_length),
        ]

        # Gimbal state
        self.nozzle_gimbal_angle = 0.0  # radians, relative to body frame
        self.max_gimbal = math.radians(6.0)  # ±6°

        # -------------------------
        # Docking adapter (at top of CM)
        # -------------------------
        dock_width = 0.6 * scale
        dock_height = 0.3 * scale
        dock_center = b2Vec2(0.0, cm_tip_y + dock_height * 0.7)
        self.docking_port_center = dock_center

        hw = dock_width / 2.0
        hh = dock_height / 2.0

        offsets = [
            b2Vec2(-hw, -hh),
            b2Vec2(+hw, -hh),
            b2Vec2(+hw, +hh),
            b2Vec2(-hw, +hh),
        ]
        angle_90 = math.pi / 2.0
        self.docking_port_vertices_local = [
            dock_center + rotate_vec(off, angle_90) for off in offsets
        ]

        # -------------------------
        # CM hatch
        # -------------------------
        self.cm_hatch_rect = {
            "center": b2Vec2(0.0, cm_base_y + 0.4 * scale),
            "half_w": 0.35 * scale,
            "half_h": 0.25 * scale,
        }

        # -------------------------
        # SM white bands
        # -------------------------
        narrow_h = 0.25 * scale
        wide_h = 0.50 * scale
        band_margin = 0.1 * scale

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
            "center": b2Vec2(self.sm_center.x, self.sm_center.y + 0.4 * scale),
            "half_w": 0.6 * scale,
            "half_h": 0.6 * scale,
        }

        # -------------------------
        # RCS pods LEFT and RIGHT (1/3 down from top of SM)
        # -------------------------
        pod_half_w = 0.15 * scale
        pod_half_h = 0.35 * scale
        gap = 0.15 * scale

        pod_center_y = sm_top_y - (sm_height / 3.0)
        left_center_x = self.sm_center.x - (self.sm_half_w + pod_half_w + gap)
        right_center_x = self.sm_center.x + (self.sm_half_w + pod_half_w + gap)

        self.rcs_pods = {
            "LEFT": {
                "center": b2Vec2(left_center_x, pod_center_y),
                "half_w": pod_half_w,
                "half_h": pod_half_h,
                "thrusters": {
                    "UP": {
                        "name": "L_UP",
                        "pos": b2Vec2(left_center_x, pod_center_y + pod_half_h),
                        "dir": b2Vec2(0.0, -1.0),
                    },
                    "DOWN": {
                        "name": "L_DOWN",
                        "pos": b2Vec2(left_center_x, pod_center_y - pod_half_h),
                        "dir": b2Vec2(0.0, +1.0),
                    },
                    "SIDE": {
                        "name": "L_SIDE",
                        "pos": b2Vec2(left_center_x - pod_half_w, pod_center_y),
                        "dir": b2Vec2(+1.0, 0.0),
                    },
                },
            },
            "RIGHT": {
                "center": b2Vec2(right_center_x, pod_center_y),
                "half_w": pod_half_w,
                "half_h": pod_half_h,
                "thrusters": {
                    "UP": {
                        "name": "R_UP",
                        "pos": b2Vec2(right_center_x, pod_center_y + pod_half_h),
                        "dir": b2Vec2(0.0, -1.0),
                    },
                    "DOWN": {
                        "name": "R_DOWN",
                        "pos": b2Vec2(right_center_x, pod_center_y - pod_half_h),
                        "dir": b2Vec2(0.0, +1.0),
                    },
                    "SIDE": {
                        "name": "R_SIDE",
                        "pos": b2Vec2(right_center_x + pod_half_w, pod_center_y),
                        "dir": b2Vec2(-1.0, 0.0),
                    },
                },
            },
        }

        # Flatten thrusters for easier access
        self.rcs_thrusters = {}
        for pod in self.rcs_pods.values():
            for t in pod["thrusters"].values():
                self.rcs_thrusters[t["name"]] = {
                    "pos": t["pos"],
                    "dir": t["dir"],
                }

        self.active_thrusters = set()
        self.thrusting = False

    # -------------------------------------------------
    # Gimbal helpers
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Physics / controls
    # -------------------------------------------------
    def apply_main_thrust(self, dt):
        """Apply main SPS thrust with gimbal angle."""
        # Thrust along +Y in engine frame, rotated by gimbal relative to body
        local_engine_dir = rotate_vec(b2Vec2(0.0, 1.0), self.nozzle_gimbal_angle)
        world_dir = rotate_vec(local_engine_dir, self.body.angle)
        force = CSM_MAIN_THRUST * world_dir

        # Apply at nozzle attach point to get realistic torque
        attach_world = local_to_world(self.body, self.nozzle_attach_center)
        self.body.ApplyForce(force, attach_world, wake=True)

    def clear_rcs_activity(self):
        """Clear all active RCS thrusters (call once per frame)."""
        self.active_thrusters.clear()

    def _fire_thruster(self, name):
        """Fire a specific RCS thruster by name."""
        t = self.rcs_thrusters[name]
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
            tolerance: Docking distance tolerance in meters (default 5.0m = overlap of two 2.5m indicators)
            angle_tolerance: Angular alignment tolerance in degrees (None = no angle check)

        Returns:
            True if docking port indicators overlap (distance between ports <= 5.0m)
        """
        import math

        # Check distance between docking ports
        docking_port_world = local_to_world(self.body, self.docking_port_center)

        # Calculate lander's docking port position (accounting for rotation)
        # Lander docking port is at (0, h*1.1) in local coords, but we need world coords
        lander_h = 1.5 * 0.5625  # lander height * scale (reduced by 25% from 0.75)
        lander_dock_local = b2Vec2(0.0, lander_h * 1.1)
        lander_dock_world = local_to_world(lander_body, lander_dock_local)

        distance = (lander_dock_world - docking_port_world).length

        # Docking succeeds when the two indicators overlap
        # Each indicator has 2.5m radius, so overlapping means distance <= 5.0m
        return distance <= tolerance

    # -------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------
    def _draw_panel_rect(self, surface, panel, fill, outline, line_width, cam_x, cam_y, screen_width, screen_height):
        """Draw a rectangular panel on the CSM."""
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
        verts_screen = [self._world_to_screen(v, cam_x, cam_y, screen_width, screen_height) for v in verts_world]
        pygame.draw.polygon(surface, fill, verts_screen)
        pygame.draw.polygon(surface, outline, verts_screen, line_width)

    def _world_to_screen(self, pos, cam_x, cam_y, screen_width, screen_height):
        """Convert world coordinates to screen coordinates with camera."""
        x = screen_width / 2 + (pos.x - cam_x) * PPM
        y = screen_height / 2 - (pos.y - cam_y) * PPM
        return int(x), int(y)

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
        # Colors
        SM_COLOR = (200, 200, 230)
        CM_COLOR = (220, 220, 220)
        NOZZLE_COLOR = (40, 40, 55)
        RCS_POD_FILL = (60, 60, 60)
        NOZZLE_TRI_FILL = (190, 190, 190)
        OUTLINE = (0, 0, 0)
        DOCK_FILL = (70, 70, 80)      # dark gray docking adapter
        DOCK_OUTLINE = (20, 20, 25)
        BAND_FILL = (240, 240, 250)
        PANEL_FILL = (210, 210, 220)
        FLAME_COLOR = (255, 190, 60)

        # Shared nozzle triangle geometry for RCS
        NOZZLE_TRI_LENGTH = 0.4 * self.scale
        NOZZLE_TRI_WIDTH = 0.35 * self.scale

        # Helper function for world to screen conversion
        def w2s(pos):
            return self._world_to_screen(pos, cam_x, cam_y, screen_width, screen_height)

        # Engine direction vectors (for flame)
        local_engine_dir = rotate_vec(b2Vec2(0.0, 1.0), self.nozzle_gimbal_angle)
        world_engine_dir = rotate_vec(local_engine_dir, self.body.angle)

        # Draw SM
        sm_world = [
            self.body.transform * v
            for v in self.service_module_fixture.shape.vertices
        ]
        pygame.draw.polygon(surface, SM_COLOR, [w2s(p) for p in sm_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in sm_world], 1)

        # Draw CM
        cm_world = [
            self.body.transform * v
            for v in self.command_module_fixture.shape.vertices
        ]
        pygame.draw.polygon(surface, CM_COLOR, [w2s(p) for p in cm_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in cm_world], 1)

        # Draw gimballed nozzle
        nozzle_world = []
        for v in self.nozzle_shape_local:
            v_rot = rotate_vec(v, self.nozzle_gimbal_angle)
            p_local = self.nozzle_attach_center + v_rot
            nozzle_world.append(local_to_world(self.body, p_local))

        pygame.draw.polygon(surface, NOZZLE_COLOR, [w2s(p) for p in nozzle_world])
        pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in nozzle_world], 1)

        # Main engine flame
        if self.thrusting:
            exit_world = nozzle_world[2:4]  # far edge of bell
            exit_mid = (exit_world[0] + exit_world[1]) * 0.5
            backward = -world_engine_dir
            flame_len = 2.0 * self.scale
            flame_width = 1.0 * self.scale
            tip = exit_mid + backward * flame_len
            perp = b2Vec2(-backward.y, backward.x)
            p1 = exit_mid + perp * (flame_width / 2)
            p2 = exit_mid - perp * (flame_width / 2)
            pygame.draw.polygon(
                surface,
                FLAME_COLOR,
                [w2s(p1), w2s(p2), w2s(tip)],
            )

        # Docking adapter
        dock_world = [
            local_to_world(self.body, v) for v in self.docking_port_vertices_local
        ]
        pygame.draw.polygon(surface, DOCK_FILL, [w2s(p) for p in dock_world])
        pygame.draw.polygon(surface, DOCK_OUTLINE, [w2s(p) for p in dock_world], 2)

        # SM white bands
        self._draw_panel_rect(surface, self.sm_band_top,
                              fill=BAND_FILL, outline=OUTLINE, line_width=1,
                              cam_x=cam_x, cam_y=cam_y, screen_width=screen_width, screen_height=screen_height)
        self._draw_panel_rect(surface, self.sm_band_bottom,
                              fill=BAND_FILL, outline=OUTLINE, line_width=1,
                              cam_x=cam_x, cam_y=cam_y, screen_width=screen_width, screen_height=screen_height)

        # Forward SM gray panel
        self._draw_panel_rect(surface, self.sm_forward_panel,
                              fill=PANEL_FILL, outline=OUTLINE, line_width=1,
                              cam_x=cam_x, cam_y=cam_y, screen_width=screen_width, screen_height=screen_height)

        # CM hatch
        self._draw_panel_rect(surface, self.cm_hatch_rect,
                              fill=PANEL_FILL, outline=OUTLINE, line_width=1,
                              cam_x=cam_x, cam_y=cam_y, screen_width=screen_width, screen_height=screen_height)

        # RCS pods (left & right)
        def draw_pod(pod):
            c = pod["center"]
            hw, hh = pod["half_w"], pod["half_h"]
            verts = [
                b2Vec2(c.x - hw, c.y - hh),
                b2Vec2(c.x + hw, c.y - hh),
                b2Vec2(c.x + hw, c.y + hh),
                b2Vec2(c.x - hw, c.y + hh),
            ]
            world_verts = [local_to_world(self.body, v) for v in verts]
            pygame.draw.polygon(surface, RCS_POD_FILL, [w2s(p) for p in world_verts])
            pygame.draw.polygon(surface, OUTLINE, [w2s(p) for p in world_verts], 1)

        draw_pod(self.rcs_pods["LEFT"])
        draw_pod(self.rcs_pods["RIGHT"])

        # Nozzle triangles on RCS
        def draw_nozzle(name):
            t = self.rcs_thrusters[name]
            tip = local_to_world(self.body, t["pos"])          # point touching pod
            force = rotate_vec(t["dir"], self.body.angle)
            nozzle_dir = -force                                # away from pod
            base_mid = tip + nozzle_dir * NOZZLE_TRI_LENGTH
            perp = b2Vec2(-nozzle_dir.y, nozzle_dir.x)
            p1 = base_mid + perp * (NOZZLE_TRI_WIDTH / 2)
            p2 = base_mid - perp * (NOZZLE_TRI_WIDTH / 2)
            pygame.draw.polygon(surface, NOZZLE_TRI_FILL, [w2s(tip), w2s(p1), w2s(p2)])

        for nm in ["L_UP", "L_DOWN", "L_SIDE", "R_UP", "R_DOWN", "R_SIDE"]:
            draw_nozzle(nm)

        # Flames on active RCS thrusters
        # Map thruster flags to thruster names
        active_thrusters = []
        if tl:
            active_thrusters.append("L_UP")
        if tr:
            active_thrusters.append("R_UP")
        if bl:
            active_thrusters.append("L_DOWN")
        if br:
            active_thrusters.append("R_DOWN")
        if sl:
            active_thrusters.append("L_SIDE")
        if sr:
            active_thrusters.append("R_SIDE")

        for name in active_thrusters:
            t = self.rcs_thrusters[name]
            tip_world = local_to_world(self.body, t["pos"])
            force = rotate_vec(t["dir"], self.body.angle)
            nozzle_dir = -force  # same direction as triangle

            base_mid = tip_world + nozzle_dir * NOZZLE_TRI_LENGTH
            flame_len = 0.8 * self.scale
            flame_w = 0.3 * self.scale
            tip_flame = base_mid + nozzle_dir * flame_len
            perp = b2Vec2(-nozzle_dir.y, nozzle_dir.x)
            p1 = base_mid + perp * (flame_w / 2)
            p2 = base_mid - perp * (flame_w / 2)
            pygame.draw.polygon(
                surface,
                FLAME_COLOR,
                [w2s(p1), w2s(p2), w2s(tip_flame)],
            )

        # Draw main SPS engine flame if firing
        if main_on:
            # Calculate nozzle exit point (tip of the bell)
            # Nozzle exit vertices are at indices 2 and 3 in nozzle_shape_local
            nozzle_exit_left = self.nozzle_shape_local[3]  # Left exit vertex
            nozzle_exit_right = self.nozzle_shape_local[2]  # Right exit vertex

            # Center of the exit
            nozzle_exit_center = b2Vec2(
                (nozzle_exit_left.x + nozzle_exit_right.x) / 2,
                (nozzle_exit_left.y + nozzle_exit_right.y) / 2
            )

            # Apply gimbal rotation to the exit point
            cos_g = math.cos(self.nozzle_gimbal_angle)
            sin_g = math.sin(self.nozzle_gimbal_angle)
            nozzle_exit_gimbaled = b2Vec2(
                nozzle_exit_center.x * cos_g - nozzle_exit_center.y * sin_g,
                nozzle_exit_center.x * sin_g + nozzle_exit_center.y * cos_g
            )

            # Translate to body frame and then to world coordinates
            nozzle_exit_body = self.nozzle_attach_center + nozzle_exit_gimbaled
            nozzle_exit_world = local_to_world(self.body, nozzle_exit_body)

            # Flame direction is opposite of engine thrust direction (gimbal-corrected)
            # Engine thrust points in +Y (up in body frame), flame goes -Y (down)
            thrust_dir_gimbaled = b2Vec2(-sin_g, cos_g)  # Gimbal rotates the thrust vector
            flame_dir = rotate_vec(-thrust_dir_gimbaled, self.body.angle)

            flame_len = 1.5 * self.scale
            flame_w = 0.8 * self.scale  # Wider to match nozzle exit width
            tip_flame = nozzle_exit_world + flame_dir * flame_len
            perp = b2Vec2(-flame_dir.y, flame_dir.x)
            p1 = nozzle_exit_world + perp * (flame_w / 2)
            p2 = nozzle_exit_world - perp * (flame_w / 2)
            pygame.draw.polygon(
                surface,
                FLAME_COLOR,
                [w2s(p1), w2s(p2), w2s(tip_flame)],
            )
