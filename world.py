"""
Apollo World - Cylindrical World with Physics

Contains the cylindrical world based on planet properties with:
- Accurate size and physics based on planet definition
- Terrain generation and management
- World object management (stars, CSM, lander, etc.)
- Cylindrical wrapping behavior
"""

from Box2D import b2World, b2Vec2
from apollo_terrain import ApolloTerrain
from apollo_sky import ApolloSky


# Planet properties table
PLANET_PROPERTIES = {
    # Format: gravity (m/s²), damping, diameter (km), low orbit altitude (km)
    # Orbital altitude chosen to maintain similar gameplay (1.05-1.06x radius ratio)
    "mercury":  {"gravity": 3.7,   "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 4879,  "orbit_alt": 130},
    "venus":    {"gravity": 8.87,  "linear_damping": 0.35, "angular_damping": 0.4,  "diameter": 12104, "orbit_alt": 300},
    "earth":    {"gravity": 9.81,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 12742, "orbit_alt": 350},
    "luna":     {"gravity": 1.62,  "linear_damping": 0.0,  "angular_damping": 0.0,  "diameter": 3475,  "orbit_alt": 100},
    "mars":     {"gravity": 3.71,  "linear_damping": 0.05, "angular_damping": 0.15, "diameter": 6779,  "orbit_alt": 180},
    "jupiter":  {"gravity": 24.79, "linear_damping": 0.4,  "angular_damping": 0.5,  "diameter": 139820, "orbit_alt": 4200},
    "saturn":   {"gravity": 10.44, "linear_damping": 0.3,  "angular_damping": 0.4,  "diameter": 116460, "orbit_alt": 3500},
    "uranus":   {"gravity": 8.69,  "linear_damping": 0.25, "angular_damping": 0.3,  "diameter": 50724, "orbit_alt": 1500},
    "neptune":  {"gravity": 11.15, "linear_damping": 0.3,  "angular_damping": 0.35, "diameter": 49244, "orbit_alt": 1500},
}


class WorldObject:
    """Base class for objects in the world (CSM, lander, stars, etc.)"""

    def __init__(self, world):
        """
        Initialize world object.

        Args:
            world: ApolloWorld instance
        """
        self.world = world

    def update(self, dt):
        """
        Update object state.

        Args:
            dt: Time step in seconds
        """
        pass

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        """
        Draw object to screen.

        Args:
            surface: Pygame surface
            cam_x: Camera X position
            cam_y: Camera Y position
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        pass

    def handle_cylindrical_wrapping(self, body):
        """
        Handle cylindrical world wrapping for a Box2D body.

        Args:
            body: Box2D body to wrap
        """
        if body is None:
            return

        # Wrap if beyond world boundaries
        if body.position.x > self.world.width:
            new_pos = b2Vec2(body.position.x - self.world.width, body.position.y)
            body.position = new_pos
        elif body.position.x < 0:
            new_pos = b2Vec2(body.position.x + self.world.width, body.position.y)
            body.position = new_pos


class ApolloWorld:
    """
    Cylindrical world with physics based on planet properties.

    The world wraps horizontally (cylindrical) to simulate orbiting around a planet.
    """

    def __init__(self, planet_name="luna", screen_width_pixels=1600, ppm=22.0, difficulty=1):
        """
        Initialize world based on planet properties.

        Args:
            planet_name: Name of planet from PLANET_PROPERTIES
            screen_width_pixels: Screen width in pixels (for calculating world size)
            ppm: Pixels per meter (for scaling)
            difficulty: Terrain difficulty level (1-3)
        """
        if planet_name not in PLANET_PROPERTIES:
            raise ValueError(f"Unknown planet '{planet_name}'. Available: {list(PLANET_PROPERTIES.keys())}")

        self.planet_name = planet_name
        self.planet_props = PLANET_PROPERTIES[planet_name]
        self.ppm = ppm
        self.difficulty = difficulty

        # Calculate world dimensions
        # Scale factor: convert real km to game meters (1:1000 ratio for playability)
        self.scale_factor = 1000.0
        self.diameter_km = self.planet_props["diameter"]
        self.orbit_alt_km = self.planet_props["orbit_alt"]

        # World width = planet diameter in game meters
        self.width = self.diameter_km * 1000.0 / self.scale_factor

        # Orbital altitude in game meters
        self.orbit_altitude = self.orbit_alt_km * 1000.0 / self.scale_factor

        # Calculate how many screens wide the world is
        screen_width_meters = screen_width_pixels / ppm
        self.world_screens = self.width / screen_width_meters

        # Physics properties
        self.gravity = -self.planet_props["gravity"]  # Negative for downward
        self.linear_damping = self.planet_props["linear_damping"]
        self.angular_damping = self.planet_props["angular_damping"]

        # Create Box2D world
        self.b2world = b2World(gravity=(0, self.gravity), doSleep=True)

        # Terrain
        self.terrain_body = None
        self.terrain_points = []
        self.pads_info = []

        # World objects (CSM, lander, stars, etc.)
        self.objects = []

        # Generate terrain
        self._generate_terrain()

        # Generate stars
        self._generate_stars()

    def _generate_terrain(self):
        """Generate terrain using ApolloTerrain."""
        terrain_gen = ApolloTerrain(world_width_meters=self.width, difficulty=self.difficulty)
        self.terrain_body, self.terrain_points, self.pads_info = terrain_gen.generate_terrain(self.b2world)

    def _generate_stars(self):
        """Generate background stars using ApolloSky."""
        sky_gen = ApolloSky(
            world_width_meters=self.width,
            orbit_altitude=self.orbit_altitude,
            star_cylinder_ratio=0.1,
            num_stars=300
        )

        # Store sky properties for rendering
        self.stars = sky_gen.get_stars()
        self.star_cylinder_ratio = sky_gen.star_cylinder_ratio
        self.star_cylinder_width = sky_gen.star_cylinder_width

    def add_object(self, obj):
        """
        Add an object to the world.

        Args:
            obj: WorldObject instance
        """
        self.objects.append(obj)

    def remove_object(self, obj):
        """
        Remove an object from the world.

        Args:
            obj: WorldObject instance
        """
        if obj in self.objects:
            self.objects.remove(obj)

    def update(self, dt):
        """
        Update world physics and all objects.

        Args:
            dt: Time step in seconds
        """
        # Apply altitude-based gravity using inverse square law
        # Real gravitational formula: g(h) = g₀ * (R / (R + h))²
        # where:
        #   g₀ = surface gravity
        #   R = planet radius
        #   h = altitude above surface
        #
        # For our cylindrical world:
        #   R = diameter / 2 (planet radius in game meters)
        #   h = body.position.y (altitude above surface)

        planet_radius = self.width / 2.0  # Use world width as diameter → radius
        gravity_surface = abs(self.gravity)

        for body in self.b2world.bodies:
            if body.type == 2:  # Dynamic body (2 = b2_dynamicBody)
                altitude = body.position.y

                if altitude < 0:
                    # Below surface: clamp to surface gravity
                    body.gravityScale = 1.0
                else:
                    # Apply inverse square law: g(h) = g₀ * (R / (R + h))²
                    # gravityScale = (R / (R + h))²
                    distance_from_center = planet_radius + altitude
                    gravity_scale = (planet_radius / distance_from_center) ** 2
                    body.gravityScale = gravity_scale

                    # For bodies in orbital regime (above 50m altitude), apply centrifugal force
                    # to counteract gravity and enable stable circular orbits in 2D side-scrolling world
                    if altitude >= 50.0:
                        # Calculate horizontal velocity
                        v_x = body.linearVelocity.x

                        # Calculate gravity at this altitude
                        gravity_at_altitude = gravity_surface * gravity_scale

                        # For a circular orbit in 2D side-scrolling (gravity always points down),
                        # we need centrifugal acceleration = v²/r pointing upward
                        # This counteracts the downward gravity
                        orbital_radius = distance_from_center
                        centrifugal_accel = (v_x * v_x) / orbital_radius

                        # Apply upward centrifugal force to counteract gravity
                        # Force = mass * acceleration
                        mass = body.mass
                        centrifugal_force_y = mass * centrifugal_accel

                        # Apply the force upward (+Y direction)
                        body.ApplyForceToCenter(b2Vec2(0.0, centrifugal_force_y), wake=True)

        # Step Box2D physics
        self.b2world.Step(dt, velocityIterations=6, positionIterations=2)

        # Update all objects
        for obj in self.objects:
            obj.update(dt)

    def draw_stars(self, surface, cam_x, cam_y, screen_width, screen_height, ppm):
        """
        Draw background stars with parallax effect from inner cylinder.

        Args:
            surface: Pygame surface
            cam_x: Camera X position in world coordinates
            cam_y: Camera Y position in world coordinates
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            ppm: Pixels per meter
        """
        from apollo_sky import draw_sky_pygame
        draw_sky_pygame(
            surface=surface,
            stars=self.stars,
            star_cylinder_ratio=self.star_cylinder_ratio,
            star_cylinder_width=self.star_cylinder_width,
            world_width=self.width,
            terrain_points=self.terrain_points,
            cam_x=cam_x,
            cam_y=cam_y,
            screen_width=screen_width,
            screen_height=screen_height,
            ppm=ppm
        )

    def draw_terrain(self, surface, cam_x, cam_y, screen_width, screen_height, ppm):
        """
        Draw terrain with landing pads with cylindrical wrapping.

        Args:
            surface: Pygame surface
            cam_x: Camera X position in world coordinates
            cam_y: Camera Y position in world coordinates
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            ppm: Pixels per meter
        """
        from apollo_terrain import draw_terrain_pygame
        draw_terrain_pygame(surface, self.terrain_points, self.pads_info, cam_x, ppm, screen_width, screen_height, cam_y, world_width=self.width)

    def draw_objects(self, surface, cam_x, cam_y, screen_width, screen_height):
        """
        Draw all world objects.

        Args:
            surface: Pygame surface
            cam_x: Camera X position
            cam_y: Camera Y position
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        for obj in self.objects:
            obj.draw(surface, cam_x, cam_y, screen_width, screen_height)

    def get_info(self):
        """
        Get world information for display.

        Returns:
            Dictionary with world info
        """
        return {
            "planet": self.planet_name.upper(),
            "gravity": self.planet_props["gravity"],
            "diameter_km": self.diameter_km,
            "diameter_m": self.width,
            "orbit_alt_km": self.orbit_alt_km,
            "orbit_alt_m": self.orbit_altitude,
            "world_screens": self.world_screens,
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
        }
