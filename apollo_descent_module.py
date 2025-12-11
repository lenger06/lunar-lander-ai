"""
Apollo Descent Module - WorldObject Implementation

The descent stage of the Apollo Lunar Module as a standalone WorldObject.
Includes landing legs, engine bell, and all physics/rendering.
"""

import math
from Box2D import b2PolygonShape, b2Vec2
from world import WorldObject
from apollolander import (
    REAL_DESCENT_MASS_KG,
    MASS_SCALE,
    DESCENT_MAIN_THRUST_FACTOR,
    draw_body,
    draw_descent_front_details,
    draw_thrusters,
)


def rotate_vec(vec, angle):
    """Rotate a 2D vector by angle (radians)."""
    import math
    ca, sa = math.cos(angle), math.sin(angle)
    return b2Vec2(vec.x * ca - vec.y * sa, vec.x * sa + vec.y * ca)


def local_to_world(body, local_point):
    """Convert body-local point to world coordinates."""
    return body.transform * local_point


class ApolloDescent(WorldObject):
    """
    Apollo Descent Module as a WorldObject.

    Features:
    - Landing legs with foot pads
    - Engine bell (gimballed)
    - Realistic mass and physics
    """

    def __init__(
        self,
        world,
        position,
        scale=1.0,
        descent_density=0.5,
        leg_density=0.4,
        friction=0.9,
        restitution=0.1,
    ):
        """
        Initialize descent module.

        Args:
            world: ApolloWorld instance
            position: b2Vec2 position in meters
            scale: Scale factor
            descent_density: Density for main body
            leg_density: Density for legs
            friction: Friction coefficient
            restitution: Bounce coefficient
        """
        super().__init__(world)
        self.scale = scale
        self.color = (220, 180, 100)  # Gold color

        # Create Box2D body in the world's physics world
        self.body = self._create_body(
            world.b2world,
            position,
            scale,
            descent_density,
            leg_density,
            friction,
            restitution,
        )

        # Apply mass scaling to match real Apollo descent mass
        self._apply_mass_scaling(self.body, REAL_DESCENT_MASS_KG)

        # Fuel system (real Apollo descent stage: 8,200 kg propellant)
        self.max_fuel_kg = 8200.0  # Maximum fuel capacity in kg
        self.fuel_kg = 8200.0  # Current fuel in kg (starts full)

        # Thrust state (for rendering)
        self.main_thrusting = False
        self.active_thrusters = set()  # Empty for descent (no RCS)

    def _create_body(
        self, b2world, position, s, body_density, leg_density, friction, restitution
    ):
        """Create descent stage Box2D body with legs and engine bell."""
        body = b2world.CreateDynamicBody(
            position=position,
            angle=0.0,
            linearDamping=0.2,
            angularDamping=0.4,
            bullet=True,
        )

        data = {
            "type": "descent",
            "scale": s,
            "leg_segments": [],
            "foot_bars": [],
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
        def add_strut(p1, p2, thickness, dens):
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
            top_corner = b2Vec2(side * main_half_w, main_half_h)
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

        data["main_half_w"] = main_half_w
        data["main_half_h"] = main_half_h

        return body

    def _apply_mass_scaling(self, body, real_mass_kg):
        """Rescale body mass to match real Apollo descent mass."""
        md = body.massData
        if md.mass <= 0:
            return
        target_mass = real_mass_kg / MASS_SCALE
        factor = target_mass / md.mass
        md.mass *= factor
        md.I *= factor
        body.massData = md

    def update(self, dt):
        """Update descent module physics."""
        # Handle cylindrical world wrapping
        self.handle_cylindrical_wrapping(self.body)

    def apply_main_thrust(self, dt):
        """
        Apply descent engine main thrust (45,040 N real Apollo specs).

        Thrust direction is opposite to nozzle direction (upward in body frame).
        """
        # Check if fuel is available
        if self.fuel_kg <= 0:
            self.main_thrusting = False
            return

        # Thrust along +Y in body frame (upward, opposite to downward nozzle)
        world_dir = rotate_vec(b2Vec2(0.0, 1.0), self.body.angle)

        # Force = mass * acceleration (thrust factor is acceleration in m/s²)
        # Real Apollo Descent Engine: 45,040 N / 10,246 kg = 4.40 m/s²
        force_magnitude = self.body.mass * DESCENT_MAIN_THRUST_FACTOR
        force = force_magnitude * world_dir

        # Apply at engine bell base for realistic torque
        bell_base_local = self.body.userData["bell_base_local"]
        attach_world = local_to_world(self.body, bell_base_local)
        self.body.ApplyForce(force, attach_world, wake=True)
        self.main_thrusting = True

        # Consume fuel based on thrust and time
        # Fuel flow rate = Thrust / (Specific Impulse * g0)
        # Apollo Descent Engine: Isp = 311s, g0 = 9.81 m/s²
        # Fuel flow = 45,040 N / (311 * 9.81) = 14.76 kg/s
        fuel_flow_rate = 45040.0 / (311.0 * 9.81)  # kg/s
        fuel_consumed = fuel_flow_rate * dt
        self.fuel_kg = max(0.0, self.fuel_kg - fuel_consumed)

        # Update body mass to reflect fuel consumption
        self._update_body_mass()

    def _update_body_mass(self):
        """Update Box2D body mass based on current fuel level."""
        # Calculate current total mass: dry mass + fuel mass
        # REAL_DESCENT_MASS_KG includes full fuel (10,246 kg total, 8,200 kg fuel)
        dry_mass_kg = REAL_DESCENT_MASS_KG - self.max_fuel_kg  # 2,046 kg dry mass
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

    def apply_rcs_thrust(self, thruster_list, dt):
        """
        Descent module has no RCS thrusters. This is a no-op for compatibility.
        """
        pass

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """Draw descent module to screen."""
        # Draw body using existing rendering function
        draw_body(surface, self.body, self.color, cam_x, screen_width, screen_height, cam_y)

    def draw_engine(self, surface, main_on, gimbal_angle_deg, cam_x, cam_y, screen_width, screen_height):
        """Draw engine flame if active."""
        draw_thrusters(
            surface,
            self.body,
            is_ascent=False,
            main_on=main_on,
            tl=False, bl=False, tr=False, br=False, sl=False, sr=False,
            gimbal_angle_deg=gimbal_angle_deg,
            cam_x=cam_x,
            screen_width=screen_width,
            screen_height=screen_height,
            cam_y=cam_y,
        )
