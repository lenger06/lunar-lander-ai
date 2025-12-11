"""
Mini-map visualization for Apollo World.

Displays a top-down view of the entire cylindrical world showing:
- Terrain outline
- Landing pads
- All spacecraft (Descent, Ascent, CSM, CM)
- Satellite
- Camera viewport indicator
"""

import pygame


class MiniMap:
    """
    Mini-map visualization showing the entire world.

    The mini-map displays a scaled-down view of the cylindrical world,
    showing terrain, objects, and the current camera viewport.
    """

    def __init__(self, x, y, width, height, world):
        """
        Initialize mini-map.

        Args:
            x: Screen X position (top-left corner)
            y: Screen Y position (top-left corner)
            width: Mini-map width in pixels
            height: Mini-map height in pixels
            world: ApolloWorld instance
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.world = world

        # Colors
        self.bg_color = (20, 20, 30, 180)  # Semi-transparent dark blue
        self.border_color = (100, 100, 120)
        self.terrain_color = (150, 150, 150)
        self.pad_color = (0, 255, 0)
        self.camera_color = (255, 255, 0, 100)  # Semi-transparent yellow
        self.object_colors = {
            'descent': (255, 200, 100),
            'ascent': (200, 200, 255),
            'sm': (100, 200, 255),
            'cm': (220, 220, 220),
            'satellite': (255, 150, 0),
        }

    def world_to_minimap(self, world_x, world_y):
        """
        Convert world coordinates to mini-map pixel coordinates.

        Args:
            world_x: World X coordinate (meters)
            world_y: World Y coordinate (meters)

        Returns:
            Tuple of (screen_x, screen_y) in mini-map coordinates
        """
        # Normalize world coordinates to [0, 1] range
        norm_x = world_x / self.world.width  # Wrap horizontally
        norm_y = world_y / self.world.orbit_altitude  # Vertical range (0 to orbit altitude)

        # Map to mini-map coordinates
        map_x = self.x + norm_x * self.width
        map_y = self.y + self.height - (norm_y * self.height)  # Flip Y axis

        return int(map_x), int(map_y)

    def draw(self, surface, camera_x, camera_y, screen_width, screen_height, ppm, objects):
        """
        Draw the mini-map.

        Args:
            surface: Pygame surface to draw on
            camera_x: Main camera X position (world coordinates)
            camera_y: Main camera Y position (world coordinates)
            screen_width: Main screen width
            screen_height: Main screen height
            ppm: Pixels per meter (for calculating camera viewport)
            objects: Dictionary of objects to draw (descent, ascent, sm, cm, satellite)
        """
        # Create semi-transparent surface for mini-map background
        minimap_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        minimap_surface.fill(self.bg_color)

        # Draw border
        pygame.draw.rect(minimap_surface, self.border_color,
                        (0, 0, self.width, self.height), 2)

        # Draw terrain outline (simplified)
        self._draw_terrain(minimap_surface)

        # Draw landing pads
        self._draw_pads(minimap_surface)

        # Draw camera viewport indicator
        self._draw_camera_viewport(minimap_surface, camera_x, camera_y,
                                   screen_width, screen_height, ppm)

        # Draw objects
        self._draw_objects(minimap_surface, objects)

        # Blit mini-map to main surface
        surface.blit(minimap_surface, (self.x, self.y))

    def _draw_terrain(self, surface):
        """Draw simplified terrain outline."""
        if not self.world.terrain_points:
            return

        # Draw terrain as a line along the bottom
        points = []
        for i, (wx, wy) in enumerate(self.world.terrain_points):
            mx, my = self.world_to_minimap(wx, wy)
            # Adjust coordinates relative to mini-map surface (not screen)
            points.append((mx - self.x, my - self.y))

        if len(points) > 1:
            pygame.draw.lines(surface, self.terrain_color, False, points, 1)

    def _draw_pads(self, surface):
        """Draw landing pads."""
        for pad in self.world.pads_info:
            x1, y = pad['x1'], pad['y']
            x2 = pad['x2']

            # Convert to mini-map coordinates
            mx1, my = self.world_to_minimap(x1, y)
            mx2, _ = self.world_to_minimap(x2, y)

            # Adjust relative to mini-map surface
            mx1 -= self.x
            mx2 -= self.x
            my -= self.y

            # Draw pad as a thicker line
            pygame.draw.line(surface, self.pad_color, (mx1, my), (mx2, my), 2)

    def _draw_camera_viewport(self, surface, camera_x, camera_y,
                              screen_width, screen_height, ppm):
        """Draw camera viewport indicator (shows what's visible on main screen)."""
        # Calculate viewport size in world coordinates
        viewport_width = screen_width / ppm
        viewport_height = screen_height / ppm

        # Camera is centered, so calculate corners
        left = camera_x - viewport_width / 2
        right = camera_x + viewport_width / 2
        top = camera_y + viewport_height / 2
        bottom = camera_y - viewport_height / 2

        # Convert to mini-map coordinates
        tl_x, tl_y = self.world_to_minimap(left, top)
        br_x, br_y = self.world_to_minimap(right, bottom)

        # Adjust relative to mini-map surface
        tl_x -= self.x
        tl_y -= self.y
        br_x -= self.x
        br_y -= self.y

        width = br_x - tl_x
        height = br_y - tl_y

        # Draw semi-transparent rectangle for viewport
        viewport_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        viewport_surface.fill(self.camera_color)
        surface.blit(viewport_surface, (tl_x, tl_y))

        # Draw viewport border
        pygame.draw.rect(surface, (255, 255, 0), (tl_x, tl_y, width, height), 1)

    def _draw_objects(self, surface, objects):
        """Draw all objects (spacecraft, satellite) on mini-map."""
        for obj_type, obj in objects.items():
            if obj is None or not hasattr(obj, 'body'):
                continue

            # Get object world position
            world_x = obj.body.position.x
            world_y = obj.body.position.y

            # Convert to mini-map coordinates
            mx, my = self.world_to_minimap(world_x, world_y)

            # Adjust relative to mini-map surface
            mx -= self.x
            my -= self.y

            # Get color for this object type
            color = self.object_colors.get(obj_type, (255, 255, 255))

            # Draw as a small circle
            pygame.draw.circle(surface, color, (mx, my), 3, 0)
