"""
Apollo Lander HUD - Heads-Up Display Rendering

Manages all on-screen information display including:
- Telemetry (altitude, velocity, angle)
- Fuel gauges (descent and ascent stages)
- Navigation info (distance to target)
- Scoring
- Game state messages
"""

import pygame
import math


class ApolloHUD:
    """Renders game HUD with telemetry, fuel, navigation, and scoring."""

    def __init__(self, screen_width=1024, screen_height=768):
        """Initialize HUD renderer."""
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts
        self.font_large = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)

        # Colors
        self.color_white = (255, 255, 255)
        self.color_yellow = (255, 255, 100)
        self.color_green = (100, 255, 100)
        self.color_orange = (255, 165, 0)
        self.color_red = (255, 100, 100)
        self.color_gray = (150, 150, 150)
        self.color_cyan = (100, 255, 255)

    def draw_telemetry(self, surface, lander_body, ground_y, mode="DESCENT"):
        """
        Draw telemetry panel (top-left).

        Args:
            surface: Pygame surface
            lander_body: Box2D body to get data from
            ground_y: Ground level for altitude calculation
            mode: "DESCENT" or "ASCENT"
        """
        alt = lander_body.worldCenter.y - ground_y
        vel = lander_body.linearVelocity
        vert_speed = vel.y
        horz_speed = vel.x
        angle_deg = math.degrees(lander_body.angle)

        # Altitude color coding
        if alt < 5.0:
            alt_color = self.color_red
        elif alt < 15.0:
            alt_color = self.color_orange
        else:
            alt_color = self.color_green

        # Vertical speed color coding
        if vert_speed < -5.0:
            vv_color = self.color_red
        elif vert_speed < -2.0:
            vv_color = self.color_orange
        else:
            vv_color = self.color_white

        # Angle color coding
        if abs(angle_deg) > 45:
            angle_color = self.color_red
        elif abs(angle_deg) > 20:
            angle_color = self.color_orange
        else:
            angle_color = self.color_green

        x, y = 10, 10
        line_height = 25

        # Mode
        text = self.font_medium.render(f"MODE: {mode}", True, self.color_yellow)
        surface.blit(text, (x, y))
        y += line_height

        # Altitude
        text = self.font_medium.render(f"ALT:  {alt:6.1f} m", True, alt_color)
        surface.blit(text, (x, y))
        y += line_height

        # Vertical speed
        text = self.font_medium.render(f"Vv:   {vert_speed:+6.2f} m/s", True, vv_color)
        surface.blit(text, (x, y))
        y += line_height

        # Horizontal speed
        text = self.font_medium.render(f"Vh:   {horz_speed:+6.2f} m/s", True, self.color_white)
        surface.blit(text, (x, y))
        y += line_height

        # Angle
        text = self.font_medium.render(f"ANG:  {angle_deg:+6.1f}°", True, angle_color)
        surface.blit(text, (x, y))

    def draw_fuel_gauge(self, surface, x, y, fuel, max_fuel, label, width=30, height=150):
        """
        Draw vertical fuel gauge.

        Args:
            surface: Pygame surface
            x, y: Top-left position
            fuel: Current fuel amount
            max_fuel: Maximum fuel capacity
            label: Label text (e.g., "DESC FUEL")
            width, height: Gauge dimensions
        """
        # Background
        pygame.draw.rect(surface, (40, 40, 40), (x, y, width, height), 0)
        pygame.draw.rect(surface, self.color_white, (x, y, width, height), 2)

        # Fuel level
        fuel_ratio = max(0.0, min(1.0, fuel / max_fuel))
        fuel_height = int(height * fuel_ratio)
        fuel_y = y + height - fuel_height

        # Color based on fuel level
        if fuel_ratio > 0.5:
            fuel_color = self.color_green
        elif fuel_ratio > 0.25:
            fuel_color = self.color_orange
        else:
            fuel_color = self.color_red

        if fuel_height > 0:
            pygame.draw.rect(surface, fuel_color, (x + 2, fuel_y, width - 4, fuel_height), 0)

        # Label
        label_text = self.font_small.render(label, True, self.color_white)
        label_rect = label_text.get_rect(centerx=x + width // 2, top=y + height + 5)
        surface.blit(label_text, label_rect)

        # Percentage
        percent_text = self.font_small.render(f"{int(fuel_ratio * 100)}%", True, self.color_white)
        percent_rect = percent_text.get_rect(centerx=x + width // 2, top=y + height + 20)
        surface.blit(percent_text, percent_rect)

    def draw_throttle_gauge(self, surface, x, y, throttle, width=50, height=170):
        """
        Draw vertical Apollo-style throttle indicator with metallic bezel.

        Matches the propellant gauge style — single vertical column with
        scale markings and a green needle indicator.

        Args:
            surface: Pygame surface
            x, y: Top-left position of the instrument
            throttle: Throttle value (0.0-1.0)
            width, height: Overall instrument dimensions
        """
        # --- Bezel (3D metallic housing) ---
        bezel_margin = 4
        pygame.draw.rect(surface, (50, 50, 50),
                         (x - 1, y - 1, width + 2, height + 2), 0, 3)
        pygame.draw.rect(surface, (120, 120, 125),
                         (x, y, width, height), 0, 3)
        pygame.draw.rect(surface, (160, 160, 165),
                         (x + 1, y + 1, width - 2, height - 2), 1, 3)
        pygame.draw.line(surface, (80, 80, 85),
                         (x + 3, y + height - 2), (x + width - 2, y + height - 2))
        pygame.draw.line(surface, (80, 80, 85),
                         (x + width - 2, y + 3), (x + width - 2, y + height - 2))

        # --- Dark face ---
        face_x = x + bezel_margin
        face_y = y + bezel_margin
        face_w = width - bezel_margin * 2
        face_h = height - bezel_margin * 2
        pygame.draw.rect(surface, (10, 10, 12), (face_x, face_y, face_w, face_h), 0)
        pygame.draw.rect(surface, (60, 60, 65), (face_x, face_y, face_w, face_h), 1)

        # --- Layout ---
        label_zone = 16
        scale_top = face_y + label_zone + 2
        scale_bottom = face_y + face_h - 6
        scale_height = scale_bottom - scale_top
        col_center = face_x + face_w // 2

        # --- Label ---
        font_label = pygame.font.SysFont("consolas", 10, bold=True)
        label = font_label.render("THROTTLE", True, (220, 220, 220))
        surface.blit(label, label.get_rect(centerx=col_center, top=face_y + 3))

        # --- Scale markings ---
        font_scale = pygame.font.SysFont("consolas", 9)
        for pct in range(0, 101, 10):
            frac = pct / 100.0
            tick_y = int(scale_bottom - frac * scale_height)

            is_major = (pct % 20 == 0)
            tick_len = 5 if is_major else 3
            tick_color = (180, 180, 180) if is_major else (110, 110, 110)

            # Ticks on left side
            pygame.draw.line(surface, tick_color,
                             (face_x + 3, tick_y),
                             (face_x + 3 + tick_len, tick_y), 1)
            # Ticks on right side
            pygame.draw.line(surface, tick_color,
                             (face_x + face_w - 4, tick_y),
                             (face_x + face_w - 4 - tick_len, tick_y), 1)

            # Numbers at major ticks (right of center)
            if is_major:
                num_text = font_scale.render(str(pct), True, (200, 200, 200))
                num_rect = num_text.get_rect(centerx=col_center,
                                             centery=tick_y)
                if tick_y > scale_top + 4 and tick_y < scale_bottom - 2:
                    surface.blit(num_text, num_rect)

        # --- Needle indicator ---
        ratio = max(0.0, min(1.0, throttle))
        needle_y = int(scale_bottom - ratio * scale_height)

        if ratio < 0.8:
            needle_color = (0, 230, 0)
        elif ratio < 0.95:
            needle_color = (255, 200, 0)
        else:
            needle_color = (255, 100, 50)

        # Horizontal needle bar spanning the column
        needle_left = face_x + 4
        needle_right = face_x + face_w - 5
        pygame.draw.line(surface, needle_color,
                         (needle_left, needle_y),
                         (needle_right, needle_y), 3)
        # Triangular pointer on left edge
        pygame.draw.polygon(surface, needle_color, [
            (needle_left - 2, needle_y - 3),
            (needle_left - 2, needle_y + 3),
            (needle_left + 2, needle_y),
        ])

    def draw_propellant_gauges(self, surface, x, y, fuel_kg, max_fuel_kg,
                                oxidizer_kg, max_oxidizer_kg, width=100, height=170):
        """
        Draw Apollo LM-style Propellant Quantity Indicator.

        Renders a realistic instrument gauge with metallic bezel, dual gauge
        columns for FUEL and OXID, scale markings, and green needle indicators.

        Args:
            surface: Pygame surface
            x, y: Top-left position of the instrument
            fuel_kg: Current fuel mass (Aerozine 50)
            max_fuel_kg: Maximum fuel capacity
            oxidizer_kg: Current oxidizer mass (N2O4)
            max_oxidizer_kg: Maximum oxidizer capacity
            width, height: Overall instrument dimensions
        """
        # --- Bezel (3D metallic housing) ---
        bezel_margin = 4
        # Outer shadow
        pygame.draw.rect(surface, (50, 50, 50),
                         (x - 1, y - 1, width + 2, height + 2), 0, 3)
        # Mid bezel
        pygame.draw.rect(surface, (120, 120, 125),
                         (x, y, width, height), 0, 3)
        # Inner highlight (top-left lighter for 3D effect)
        pygame.draw.rect(surface, (160, 160, 165),
                         (x + 1, y + 1, width - 2, height - 2), 1, 3)
        # Inner shadow (bottom-right darker for 3D)
        pygame.draw.line(surface, (80, 80, 85),
                         (x + 3, y + height - 2), (x + width - 2, y + height - 2))
        pygame.draw.line(surface, (80, 80, 85),
                         (x + width - 2, y + 3), (x + width - 2, y + height - 2))

        # --- Gauge face (dark interior) ---
        face_x = x + bezel_margin
        face_y = y + bezel_margin
        face_w = width - bezel_margin * 2
        face_h = height - bezel_margin * 2
        pygame.draw.rect(surface, (10, 10, 12), (face_x, face_y, face_w, face_h), 0)
        # Subtle inner border
        pygame.draw.rect(surface, (60, 60, 65), (face_x, face_y, face_w, face_h), 1)

        # --- Layout ---
        label_zone = 16       # top area for FUEL / OXID labels
        scale_top = face_y + label_zone + 2
        scale_bottom = face_y + face_h - 6
        scale_height = scale_bottom - scale_top

        col_width = (face_w - 2) // 2  # each gauge column
        fuel_col_x = face_x + 1
        oxid_col_x = face_x + 1 + col_width
        center_x = face_x + face_w // 2

        # --- Column labels ---
        font_tiny = pygame.font.SysFont("consolas", 10, bold=True)
        fuel_label = font_tiny.render("FUEL", True, (220, 220, 220))
        oxid_label = font_tiny.render("OXID", True, (220, 220, 220))
        surface.blit(fuel_label,
                     fuel_label.get_rect(centerx=fuel_col_x + col_width // 2,
                                         top=face_y + 3))
        surface.blit(oxid_label,
                     oxid_label.get_rect(centerx=oxid_col_x + col_width // 2,
                                         top=face_y + 3))

        # --- Center divider ---
        pygame.draw.line(surface, (80, 80, 85),
                         (center_x, scale_top - 2), (center_x, scale_bottom + 2), 1)

        # --- Scale markings ---
        font_scale = pygame.font.SysFont("consolas", 9)
        for pct in range(0, 101, 10):
            frac = pct / 100.0
            tick_y = int(scale_bottom - frac * scale_height)

            is_major = (pct % 20 == 0)
            tick_len = 5 if is_major else 3
            tick_color = (180, 180, 180) if is_major else (110, 110, 110)

            # Left column ticks (right side of fuel column, toward center)
            pygame.draw.line(surface, tick_color,
                             (center_x - 2, tick_y),
                             (center_x - 2 - tick_len, tick_y), 1)
            # Right column ticks (left side of oxid column, toward center)
            pygame.draw.line(surface, tick_color,
                             (center_x + 2, tick_y),
                             (center_x + 2 + tick_len, tick_y), 1)

            # Outer ticks (far edges)
            pygame.draw.line(surface, tick_color,
                             (fuel_col_x + 2, tick_y),
                             (fuel_col_x + 2 + tick_len, tick_y), 1)
            pygame.draw.line(surface, tick_color,
                             (oxid_col_x + col_width - 3, tick_y),
                             (oxid_col_x + col_width - 3 - tick_len, tick_y), 1)

            # Numbers at major ticks (drawn in the center gap area)
            if is_major:
                num_text = font_scale.render(str(pct), True, (200, 200, 200))
                num_rect = num_text.get_rect(centerx=center_x,
                                             centery=tick_y)
                # Only draw number if it fits and doesn't overlap labels
                if tick_y > scale_top + 4 and tick_y < scale_bottom - 2:
                    surface.blit(num_text, num_rect)

        # --- Needle indicators ---
        fuel_ratio = max(0.0, min(1.0, fuel_kg / max_fuel_kg)) if max_fuel_kg > 0 else 0.0
        oxid_ratio = max(0.0, min(1.0, oxidizer_kg / max_oxidizer_kg)) if max_oxidizer_kg > 0 else 0.0

        for ratio, col_x in [(fuel_ratio, fuel_col_x), (oxid_ratio, oxid_col_x)]:
            needle_y = int(scale_bottom - ratio * scale_height)

            # Needle color based on level
            if ratio > 0.25:
                needle_color = (0, 230, 0)       # bright green
            elif ratio > 0.10:
                needle_color = (255, 165, 0)      # orange warning
            else:
                needle_color = (255, 60, 60)       # red critical

            # Draw needle — thick green horizontal bar spanning the column
            needle_left = col_x + 3
            needle_right = col_x + col_width - 4
            pygame.draw.line(surface, needle_color,
                             (needle_left, needle_y),
                             (needle_right, needle_y), 3)
            # Small triangular pointer on outer edge for visibility
            pygame.draw.polygon(surface, needle_color, [
                (needle_left - 2, needle_y - 3),
                (needle_left - 2, needle_y + 3),
                (needle_left + 2, needle_y),
            ])

    def draw_range_rate_gauge(self, surface, x, y, altitude, vertical_velocity,
                              max_alt=500.0, max_vel=30.0, width=100, height=170):
        """
        Draw Apollo LM-style Range/Alt and Alt Rate dual gauge.

        Left column: Altitude (0 to max_alt, bottom to top).
        Right column: Vertical velocity (-max_vel to +max_vel, center zero).

        Args:
            surface: Pygame surface
            x, y: Top-left position of the instrument
            altitude: Current altitude in meters
            vertical_velocity: Current vertical velocity in m/s (positive = ascending)
            max_alt: Maximum altitude for full scale
            max_vel: Maximum velocity magnitude (symmetric +/-)
            width, height: Overall instrument dimensions
        """
        # --- Bezel (3D metallic housing) ---
        bezel_margin = 4
        pygame.draw.rect(surface, (50, 50, 50),
                         (x - 1, y - 1, width + 2, height + 2), 0, 3)
        pygame.draw.rect(surface, (120, 120, 125),
                         (x, y, width, height), 0, 3)
        pygame.draw.rect(surface, (160, 160, 165),
                         (x + 1, y + 1, width - 2, height - 2), 1, 3)
        pygame.draw.line(surface, (80, 80, 85),
                         (x + 3, y + height - 2), (x + width - 2, y + height - 2))
        pygame.draw.line(surface, (80, 80, 85),
                         (x + width - 2, y + 3), (x + width - 2, y + height - 2))

        # --- Dark face ---
        face_x = x + bezel_margin
        face_y = y + bezel_margin
        face_w = width - bezel_margin * 2
        face_h = height - bezel_margin * 2
        pygame.draw.rect(surface, (10, 10, 12), (face_x, face_y, face_w, face_h), 0)
        pygame.draw.rect(surface, (60, 60, 65), (face_x, face_y, face_w, face_h), 1)

        # --- Layout ---
        label_zone = 16
        scale_top = face_y + label_zone + 2
        scale_bottom = face_y + face_h - 18  # Leave room for value readouts
        scale_height = scale_bottom - scale_top

        col_width = (face_w - 2) // 2
        alt_col_x = face_x + 1
        vel_col_x = face_x + 1 + col_width
        center_x_pos = face_x + face_w // 2

        # --- Column labels ---
        font_tiny = pygame.font.SysFont("consolas", 10, bold=True)
        alt_label = font_tiny.render("ALT", True, (220, 220, 220))
        vel_label = font_tiny.render("RATE", True, (220, 220, 220))
        surface.blit(alt_label,
                     alt_label.get_rect(centerx=alt_col_x + col_width // 2,
                                        top=face_y + 3))
        surface.blit(vel_label,
                     vel_label.get_rect(centerx=vel_col_x + col_width // 2,
                                        top=face_y + 3))

        # --- Center divider ---
        pygame.draw.line(surface, (80, 80, 85),
                         (center_x_pos, scale_top - 2),
                         (center_x_pos, scale_bottom + 2), 1)

        # --- ALT scale markings (left column, 0 at bottom, max_alt at top) ---
        font_scale = pygame.font.SysFont("consolas", 8)

        alt_major_step = 100
        alt_minor_step = 50
        for val in range(0, int(max_alt) + 1, alt_minor_step):
            frac = val / max_alt
            tick_y = int(scale_bottom - frac * scale_height)

            is_major = (val % alt_major_step == 0)
            tick_len = 5 if is_major else 3
            tick_color = (180, 180, 180) if is_major else (110, 110, 110)

            # Outer ticks (far left of alt column)
            pygame.draw.line(surface, tick_color,
                             (alt_col_x + 2, tick_y),
                             (alt_col_x + 2 + tick_len, tick_y), 1)
            # Inner ticks (toward center)
            pygame.draw.line(surface, tick_color,
                             (center_x_pos - 2, tick_y),
                             (center_x_pos - 2 - tick_len, tick_y), 1)

            # Numbers at major ticks (centered in alt column)
            if is_major:
                num_text = font_scale.render(str(int(val)), True, (200, 200, 200))
                num_rect = num_text.get_rect(centerx=alt_col_x + col_width // 2,
                                             centery=tick_y)
                if tick_y > scale_top + 4 and tick_y < scale_bottom - 2:
                    surface.blit(num_text, num_rect)

        # --- VEL scale markings (right column, -max_vel at bottom, +max_vel at top) ---
        vel_major_step = 10
        vel_minor_step = 5

        for val_i in range(int(-max_vel), int(max_vel) + 1, vel_minor_step):
            frac = (val_i + max_vel) / (2 * max_vel)
            tick_y = int(scale_bottom - frac * scale_height)

            is_major = (val_i % vel_major_step == 0)
            tick_len = 5 if is_major else 3
            tick_color = (180, 180, 180) if is_major else (110, 110, 110)

            # Inner ticks (toward center)
            pygame.draw.line(surface, tick_color,
                             (center_x_pos + 2, tick_y),
                             (center_x_pos + 2 + tick_len, tick_y), 1)
            # Outer ticks (far right)
            pygame.draw.line(surface, tick_color,
                             (vel_col_x + col_width - 3, tick_y),
                             (vel_col_x + col_width - 3 - tick_len, tick_y), 1)

            # Numbers at major ticks (centered in vel column)
            if is_major:
                num_str = str(int(val_i))
                num_text = font_scale.render(num_str, True, (200, 200, 200))
                num_rect = num_text.get_rect(centerx=vel_col_x + col_width // 2,
                                             centery=tick_y)
                if tick_y > scale_top + 4 and tick_y < scale_bottom - 2:
                    surface.blit(num_text, num_rect)

        # Prominent zero line on velocity column
        zero_frac = max_vel / (2 * max_vel)
        zero_y = int(scale_bottom - zero_frac * scale_height)
        pygame.draw.line(surface, (150, 150, 150),
                         (vel_col_x + 3, zero_y),
                         (vel_col_x + col_width - 4, zero_y), 1)

        # --- Needle indicators ---
        # ALT needle
        alt_ratio = max(0.0, min(1.0, altitude / max_alt)) if max_alt > 0 else 0.0
        alt_needle_y = int(scale_bottom - alt_ratio * scale_height)

        if altitude > 50:
            alt_needle_color = (0, 230, 0)
        elif altitude > 20:
            alt_needle_color = (255, 165, 0)
        else:
            alt_needle_color = (255, 60, 60)

        needle_left = alt_col_x + 3
        needle_right = alt_col_x + col_width - 4
        pygame.draw.line(surface, alt_needle_color,
                         (needle_left, alt_needle_y),
                         (needle_right, alt_needle_y), 3)
        pygame.draw.polygon(surface, alt_needle_color, [
            (needle_left - 2, alt_needle_y - 3),
            (needle_left - 2, alt_needle_y + 3),
            (needle_left + 2, alt_needle_y),
        ])

        # VEL needle
        vel_ratio = (vertical_velocity + max_vel) / (2 * max_vel)
        vel_ratio = max(0.0, min(1.0, vel_ratio))
        vel_needle_y = int(scale_bottom - vel_ratio * scale_height)

        if vertical_velocity > -3:
            vel_needle_color = (0, 230, 0)
        elif vertical_velocity > -8:
            vel_needle_color = (255, 165, 0)
        else:
            vel_needle_color = (255, 60, 60)

        needle_left = vel_col_x + 3
        needle_right = vel_col_x + col_width - 4
        pygame.draw.line(surface, vel_needle_color,
                         (needle_left, vel_needle_y),
                         (needle_right, vel_needle_y), 3)
        pygame.draw.polygon(surface, vel_needle_color, [
            (needle_right + 2, vel_needle_y - 3),
            (needle_right + 2, vel_needle_y + 3),
            (needle_right - 2, vel_needle_y),
        ])

        # --- Value readouts at bottom of face ---
        font_val = pygame.font.SysFont("consolas", 8)
        readout_y = scale_bottom + 5

        alt_str = f"{altitude:.0f}m"
        alt_readout = font_val.render(alt_str, True, alt_needle_color)
        surface.blit(alt_readout,
                     alt_readout.get_rect(centerx=alt_col_x + col_width // 2,
                                          top=readout_y))

        vel_str = f"{vertical_velocity:+.1f}"
        vel_readout = font_val.render(vel_str, True, vel_needle_color)
        surface.blit(vel_readout,
                     vel_readout.get_rect(centerx=vel_col_x + col_width // 2,
                                          top=readout_y))

    def draw_horizontal_velocity_gauge(self, surface, x, y, horizontal_velocity,
                                       max_vel=10.0, width=160, height=50):
        """
        Draw Apollo-style horizontal velocity gauge with metallic bezel.

        Horizontal bar with center zero, needle slides left/right.

        Args:
            surface: Pygame surface
            x, y: Top-left position of the instrument
            horizontal_velocity: Current horizontal velocity in m/s (positive = right)
            max_vel: Maximum velocity magnitude (symmetric +/-)
            width, height: Overall instrument dimensions
        """
        # --- Bezel (3D metallic housing) ---
        bezel_margin = 4
        pygame.draw.rect(surface, (50, 50, 50),
                         (x - 1, y - 1, width + 2, height + 2), 0, 3)
        pygame.draw.rect(surface, (120, 120, 125),
                         (x, y, width, height), 0, 3)
        pygame.draw.rect(surface, (160, 160, 165),
                         (x + 1, y + 1, width - 2, height - 2), 1, 3)
        pygame.draw.line(surface, (80, 80, 85),
                         (x + 3, y + height - 2), (x + width - 2, y + height - 2))
        pygame.draw.line(surface, (80, 80, 85),
                         (x + width - 2, y + 3), (x + width - 2, y + height - 2))

        # --- Dark face ---
        face_x = x + bezel_margin
        face_y = y + bezel_margin
        face_w = width - bezel_margin * 2
        face_h = height - bezel_margin * 2
        pygame.draw.rect(surface, (10, 10, 12), (face_x, face_y, face_w, face_h), 0)
        pygame.draw.rect(surface, (60, 60, 65), (face_x, face_y, face_w, face_h), 1)

        # --- Layout ---
        font_label = pygame.font.SysFont("consolas", 10, bold=True)
        font_scale = pygame.font.SysFont("consolas", 8)
        font_val = pygame.font.SysFont("consolas", 8)

        bar_margin = 8
        bar_left = face_x + bar_margin
        bar_width = face_w - bar_margin * 2
        bar_center = bar_left + bar_width // 2
        bar_height = 10
        bar_y = face_y + 18

        # --- Label ---
        label = font_label.render("HORZ VEL", True, (220, 220, 220))
        surface.blit(label, label.get_rect(centerx=face_x + face_w // 2,
                                           top=face_y + 3))

        # --- Bar background ---
        pygame.draw.rect(surface, (30, 30, 32),
                         (bar_left, bar_y, bar_width, bar_height), 0)
        pygame.draw.rect(surface, (70, 70, 75),
                         (bar_left, bar_y, bar_width, bar_height), 1)

        # --- Center line ---
        pygame.draw.line(surface, (150, 150, 150),
                         (bar_center, bar_y - 1),
                         (bar_center, bar_y + bar_height + 1), 1)

        # --- Scale tick marks ---
        for val_i in range(int(-max_vel), int(max_vel) + 1):
            frac = (val_i + max_vel) / (2 * max_vel)
            tx = bar_left + int(frac * bar_width)

            is_major = (val_i % 5 == 0)
            tick_h = 4 if is_major else 2
            tick_color = (180, 180, 180) if is_major else (110, 110, 110)

            # Ticks above bar
            pygame.draw.line(surface, tick_color,
                             (tx, bar_y - 1), (tx, bar_y - 1 - tick_h), 1)

            # Numbers at major ticks (below bar)
            if is_major and val_i != 0:
                num_str = str(int(val_i))
                num_text = font_scale.render(num_str, True, (200, 200, 200))
                num_rect = num_text.get_rect(centerx=tx,
                                             top=bar_y + bar_height + 2)
                surface.blit(num_text, num_rect)

        # Zero label below bar
        zero_text = font_scale.render("0", True, (200, 200, 200))
        surface.blit(zero_text, zero_text.get_rect(centerx=bar_center,
                                                    top=bar_y + bar_height + 2))

        # --- Needle indicator ---
        vel_norm = max(-1.0, min(1.0, horizontal_velocity / max_vel))
        needle_x = bar_center + int(vel_norm * (bar_width // 2))

        if abs(horizontal_velocity) < 2.0:
            needle_color = (0, 230, 0)
        elif abs(horizontal_velocity) < 5.0:
            needle_color = (255, 165, 0)
        else:
            needle_color = (255, 60, 60)

        # Vertical needle bar
        pygame.draw.rect(surface, needle_color,
                         (needle_x - 2, bar_y + 1, 5, bar_height - 2), 0)
        # Triangular pointer above bar
        pygame.draw.polygon(surface, needle_color, [
            (needle_x, bar_y - 2),
            (needle_x - 3, bar_y - 6),
            (needle_x + 3, bar_y - 6),
        ])

        # --- Value readout (right side of label area) ---
        vel_str = f"{horizontal_velocity:+.1f} m/s"
        vel_readout = font_val.render(vel_str, True, needle_color)
        surface.blit(vel_readout,
                     vel_readout.get_rect(right=face_x + face_w - 3,
                                          top=face_y + 4))

    def draw_cg_gimbal_indicator(self, surface, cg_offset_x, total_gimbal,
                                  auto_gimbal, manual_gimbal,
                                  max_offset=0.5, max_gimbal=6.0,
                                  x=None, y=None, width=160, height=95):
        """
        Draw combined CG offset and gimbal indicator instrument panel.

        Renders a dual-scale horizontal instrument with metallic bezel showing:
        - CG lateral offset (top bar)
        - Gimbal angle with auto/trim breakdown markers (bottom bar)

        Args:
            surface: Pygame surface
            cg_offset_x: Current CG offset (negative=left, positive=right)
            total_gimbal: Combined gimbal angle (degrees)
            auto_gimbal: Auto CG compensation component (degrees)
            manual_gimbal: Manual player trim component (degrees)
            max_offset: Maximum CG offset for scaling
            max_gimbal: Maximum gimbal angle for scaling
            x, y: Top-left position (defaults to bottom-left area)
            width, height: Overall instrument dimensions
        """
        if x is None:
            x = 10
        if y is None:
            y = self.screen_height - 110

        # --- Bezel (3D metallic housing, matching propellant gauge style) ---
        bezel_margin = 4
        pygame.draw.rect(surface, (50, 50, 50),
                         (x - 1, y - 1, width + 2, height + 2), 0, 3)
        pygame.draw.rect(surface, (120, 120, 125),
                         (x, y, width, height), 0, 3)
        pygame.draw.rect(surface, (160, 160, 165),
                         (x + 1, y + 1, width - 2, height - 2), 1, 3)
        pygame.draw.line(surface, (80, 80, 85),
                         (x + 3, y + height - 2), (x + width - 2, y + height - 2))
        pygame.draw.line(surface, (80, 80, 85),
                         (x + width - 2, y + 3), (x + width - 2, y + height - 2))

        # --- Dark face ---
        face_x = x + bezel_margin
        face_y = y + bezel_margin
        face_w = width - bezel_margin * 2
        face_h = height - bezel_margin * 2
        pygame.draw.rect(surface, (10, 10, 12), (face_x, face_y, face_w, face_h), 0)
        pygame.draw.rect(surface, (60, 60, 65), (face_x, face_y, face_w, face_h), 1)

        # --- Layout ---
        font_label = pygame.font.SysFont("consolas", 10, bold=True)
        font_value = pygame.font.SysFont("consolas", 9)
        bar_margin = 8
        bar_width = face_w - bar_margin * 2
        bar_height = 10
        bar_left = face_x + bar_margin
        bar_center = bar_left + bar_width // 2

        # ============================================
        # CG OFFSET bar (top)
        # ============================================
        cg_label_y = face_y + 3
        cg_bar_y = cg_label_y + 14

        # Label + value
        cg_label = font_label.render("CG OFFSET", True, (200, 200, 200))
        surface.blit(cg_label, cg_label.get_rect(left=bar_left, top=cg_label_y))
        cg_val = font_value.render(f"{cg_offset_x:+.2f}m", True, (180, 180, 180))
        surface.blit(cg_val, cg_val.get_rect(right=bar_left + bar_width, top=cg_label_y + 1))

        # Bar background
        pygame.draw.rect(surface, (30, 30, 32),
                         (bar_left, cg_bar_y, bar_width, bar_height), 0)
        pygame.draw.rect(surface, (70, 70, 75),
                         (bar_left, cg_bar_y, bar_width, bar_height), 1)

        # Center line
        pygame.draw.line(surface, (100, 100, 105),
                         (bar_center, cg_bar_y), (bar_center, cg_bar_y + bar_height), 1)

        # Tick marks
        for frac in [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]:
            tx = bar_center + int(frac * (bar_width // 2))
            pygame.draw.line(surface, (70, 70, 75),
                             (tx, cg_bar_y), (tx, cg_bar_y + 3), 1)

        # CG marker
        cg_norm = max(-1.0, min(1.0, cg_offset_x / max_offset)) if max_offset > 0 else 0.0
        cg_mx = bar_center + int(cg_norm * (bar_width // 2 - 3))

        if abs(cg_norm) < 0.3:
            cg_color = (0, 230, 0)
        elif abs(cg_norm) < 0.6:
            cg_color = (255, 255, 100)
        else:
            cg_color = (255, 60, 60)

        # Filled marker with pointer
        pygame.draw.rect(surface, cg_color,
                         (cg_mx - 2, cg_bar_y + 1, 5, bar_height - 2), 0)
        pygame.draw.polygon(surface, cg_color, [
            (cg_mx, cg_bar_y - 3),
            (cg_mx - 3, cg_bar_y),
            (cg_mx + 3, cg_bar_y),
        ])

        # ============================================
        # GIMBAL bar (bottom)
        # ============================================
        gmb_label_y = cg_bar_y + bar_height + 6
        gmb_bar_y = gmb_label_y + 14

        # Label + value
        gmb_label = font_label.render("GIMBAL", True, (200, 200, 200))
        surface.blit(gmb_label, gmb_label.get_rect(left=bar_left, top=gmb_label_y))
        gmb_val = font_value.render(f"{total_gimbal:+5.1f}\u00b0", True, (180, 180, 180))
        surface.blit(gmb_val, gmb_val.get_rect(right=bar_left + bar_width, top=gmb_label_y + 1))

        # Bar background
        pygame.draw.rect(surface, (30, 30, 32),
                         (bar_left, gmb_bar_y, bar_width, bar_height), 0)
        pygame.draw.rect(surface, (70, 70, 75),
                         (bar_left, gmb_bar_y, bar_width, bar_height), 1)

        # Center line
        pygame.draw.line(surface, (100, 100, 105),
                         (bar_center, gmb_bar_y), (bar_center, gmb_bar_y + bar_height), 1)

        # Degree tick marks
        for deg in range(-5, 6):
            if deg == 0:
                continue
            frac = deg / max_gimbal
            tx = bar_center + int(frac * (bar_width // 2))
            tick_h = 4 if deg % 2 == 0 else 2
            pygame.draw.line(surface, (70, 70, 75),
                             (tx, gmb_bar_y), (tx, gmb_bar_y + tick_h), 1)

        # Helper to compute marker x from gimbal degrees
        def gimbal_to_x(deg):
            norm = max(-1.0, min(1.0, deg / max_gimbal)) if max_gimbal > 0 else 0.0
            return bar_center + int(norm * (bar_width // 2 - 3))

        # Auto gimbal marker (cyan triangle below bar)
        if abs(auto_gimbal) > 0.05:
            auto_x = gimbal_to_x(auto_gimbal)
            pygame.draw.polygon(surface, (0, 200, 220), [
                (auto_x, gmb_bar_y + bar_height + 3),
                (auto_x - 3, gmb_bar_y + bar_height + 7),
                (auto_x + 3, gmb_bar_y + bar_height + 7),
            ])

        # Manual trim marker (yellow triangle below bar, offset slightly)
        if abs(manual_gimbal) > 0.05:
            trim_x = gimbal_to_x(manual_gimbal)
            pygame.draw.polygon(surface, (255, 255, 100), [
                (trim_x, gmb_bar_y + bar_height + 3),
                (trim_x - 3, gmb_bar_y + bar_height + 7),
                (trim_x + 3, gmb_bar_y + bar_height + 7),
            ])

        # Total gimbal marker (main — bright bar with top pointer)
        total_x = gimbal_to_x(total_gimbal)
        if abs(total_gimbal) > 0.1:
            gmb_color = (255, 200, 0)
        else:
            gmb_color = (0, 230, 0)

        pygame.draw.rect(surface, gmb_color,
                         (total_x - 2, gmb_bar_y + 1, 5, bar_height - 2), 0)
        pygame.draw.polygon(surface, gmb_color, [
            (total_x, gmb_bar_y - 3),
            (total_x - 3, gmb_bar_y),
            (total_x + 3, gmb_bar_y),
        ])

        # --- Legend (tiny labels for marker colors) ---
        legend_y = gmb_bar_y + bar_height + 9
        legend_font = pygame.font.SysFont("consolas", 8)

        auto_lbl = legend_font.render("AUTO", True, (0, 200, 220))
        surface.blit(auto_lbl, (bar_left, legend_y))
        trim_lbl = legend_font.render("TRIM", True, (255, 255, 100))
        surface.blit(trim_lbl, trim_lbl.get_rect(right=bar_left + bar_width, top=legend_y))

    # Keep old method signatures as wrappers for backward compatibility
    def draw_cg_indicator(self, surface, cg_offset_x, max_offset=0.5, x=None, y=None):
        """Legacy wrapper — calls draw_cg_gimbal_indicator with zero gimbal."""
        self.draw_cg_gimbal_indicator(surface, cg_offset_x, 0.0, 0.0, 0.0,
                                       max_offset=max_offset, x=x, y=y)

    def draw_gimbal_breakdown(self, surface, total_gimbal, auto_gimbal, manual_gimbal, x=None, y=None):
        """Legacy wrapper — no-op when using combined indicator."""
        pass

    def draw_navigation(self, surface, lander_x, target_x, world_pos_x, pos_x=None, pos_y=None):
        """
        Draw navigation panel.

        Args:
            surface: Pygame surface
            lander_x: Lander X position
            target_x: Target pad X position
            world_pos_x: Lander position in world (for display)
            pos_x: Optional X position (defaults to right side)
            pos_y: Optional Y position (defaults to 10)
        """
        dx = target_x - lander_x
        horiz_dist = abs(dx)

        x = pos_x if pos_x is not None else self.screen_width - 250
        y = pos_y if pos_y is not None else 10
        line_height = 25

        # Header
        text = self.font_medium.render("=== NAVIGATION ===", True, self.color_yellow)
        surface.blit(text, (x, y))
        y += line_height

        # Direction and distance
        if horiz_dist < 1.0:
            direction_text = "DIRECTLY ABOVE"
            direction_color = self.color_green
        elif dx < 0:
            direction_text = f"← LEFT {horiz_dist:.1f}m"
            direction_color = self.color_orange
        else:
            direction_text = f"RIGHT {horiz_dist:.1f}m →"
            direction_color = self.color_orange

        text = self.font_medium.render(direction_text, True, direction_color)
        surface.blit(text, (x, y))
        y += line_height

        # World position
        text = self.font_small.render(f"World pos: {world_pos_x:.1f}m", True, self.color_gray)
        surface.blit(text, (x, y))
        y += line_height

        # Target position
        text = self.font_small.render(f"Target at: {target_x:.1f}m", True, self.color_gray)
        surface.blit(text, (x, y))

    def draw_score(self, surface, score):
        """
        Draw score (top-center).

        Args:
            surface: Pygame surface
            score: Current score
        """
        text = self.font_large.render(f"SCORE: {int(score)}", True, self.color_yellow)
        text_rect = text.get_rect(centerx=self.screen_width // 2, top=10)
        surface.blit(text, text_rect)

    def draw_controls_help(self, surface, mode="DESCENT"):
        """
        Draw controls help (bottom).

        Args:
            surface: Pygame surface
            mode: "DESCENT" or "ASCENT"
        """
        y = self.screen_height - 60

        if mode == "DESCENT":
            help_text = (
                "UP/DOWN=throttle  ,/.=gimbal  /=center  "
                "LEFT/RIGHT=RCS X  W/S=RCS Y  Q/E=RCS roll  SPACE=separate  BACKSPACE=reset  ESC=quit"
            )
        else:
            help_text = (
                "UP=ascent pulse  LEFT/RIGHT=RCS X  W/S=RCS Y  Q/E=RCS roll  BACKSPACE=reset  ESC=quit"
            )

        text = self.font_small.render(help_text, True, self.color_gray)
        text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
        surface.blit(text, text_rect)

    def draw_game_over(self, surface, success, score, fuel_bonus=0, pad_multiplier=1):
        """
        Draw game over screen.

        Args:
            surface: Pygame surface
            success: True if landed successfully, False if crashed
            score: Final score
            fuel_bonus: Bonus points from remaining fuel
            pad_multiplier: Landing pad difficulty multiplier
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, 0))

        y = self.screen_height // 2 - 100
        line_height = 50

        # Result message
        if success:
            result_text = "SUCCESSFUL LANDING!"
            result_color = self.color_green
        else:
            result_text = "CRASHED!"
            result_color = self.color_red

        text = self.font_large.render(result_text, True, result_color)
        text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
        surface.blit(text, text_rect)
        y += line_height

        # Score breakdown (if successful)
        if success:
            text = self.font_medium.render(
                f"Pad Multiplier: x{pad_multiplier}", True, self.color_white
            )
            text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
            surface.blit(text, text_rect)
            y += 35

            text = self.font_medium.render(
                f"Fuel Bonus: +{int(fuel_bonus)}", True, self.color_white
            )
            text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
            surface.blit(text, text_rect)
            y += 35

        # Final score
        text = self.font_large.render(
            f"FINAL SCORE: {int(score)}", True, self.color_yellow
        )
        text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
        surface.blit(text, text_rect)
        y += line_height + 20

        # Restart prompt
        text = self.font_medium.render(
            "Press R to restart or ESC to quit", True, self.color_cyan
        )
        text_rect = text.get_rect(centerx=self.screen_width // 2, top=y)
        surface.blit(text, text_rect)

    def draw_throttle_indicator(self, surface, throttle, gimbal_deg=0.0):
        """
        Draw throttle and gimbal indicators.

        Args:
            surface: Pygame surface
            throttle: Throttle value (0.0-1.0)
            gimbal_deg: Gimbal angle in degrees
        """
        x = 10
        y = self.screen_height - 200

        # Throttle bar
        bar_width = 200
        bar_height = 20

        text = self.font_small.render("THROTTLE", True, self.color_white)
        surface.blit(text, (x, y - 20))

        # Background
        pygame.draw.rect(surface, (40, 40, 40), (x, y, bar_width, bar_height), 0)
        pygame.draw.rect(surface, self.color_white, (x, y, bar_width, bar_height), 2)

        # Throttle level
        throttle_width = int(bar_width * throttle)
        if throttle_width > 0:
            throttle_color = self.color_green if throttle < 0.8 else self.color_orange
            pygame.draw.rect(surface, throttle_color, (x + 2, y + 2, throttle_width - 4, bar_height - 4), 0)

        # Percentage
        percent_text = self.font_small.render(f"{int(throttle * 100)}%", True, self.color_white)
        surface.blit(percent_text, (x + bar_width + 10, y))

        # Gimbal indicator
        y += 40
        text = self.font_small.render(f"GIMBAL: {gimbal_deg:+5.1f}°", True, self.color_white)
        surface.blit(text, (x, y))
