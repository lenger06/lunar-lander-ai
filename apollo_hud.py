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

    def draw_propellant_gauges(self, surface, x, y, fuel_kg, max_fuel_kg,
                                oxidizer_kg, max_oxidizer_kg, width=25, height=120):
        """
        Draw side-by-side fuel and oxidizer gauges with labels.

        Shows Aerozine 50 (fuel) on left, N2O4 (oxidizer) on right.

        Args:
            surface: Pygame surface
            x, y: Top-left position
            fuel_kg: Current fuel mass (Aerozine 50)
            max_fuel_kg: Maximum fuel capacity
            oxidizer_kg: Current oxidizer mass (N2O4)
            max_oxidizer_kg: Maximum oxidizer capacity
            width, height: Individual gauge dimensions
        """
        gap = 5

        # Fuel gauge (left, green theme) - Aerozine 50
        self._draw_propellant_bar(surface, x, y, fuel_kg, max_fuel_kg,
                                  "FUEL", width, height, "fuel")

        # Oxidizer gauge (right, cyan theme) - N2O4
        self._draw_propellant_bar(surface, x + width + gap, y,
                                  oxidizer_kg, max_oxidizer_kg,
                                  "OXID", width, height, "oxidizer")

    def _draw_propellant_bar(self, surface, x, y, amount, max_amount,
                              label, width, height, prop_type):
        """Draw a single propellant bar with appropriate coloring."""
        # Background
        pygame.draw.rect(surface, (40, 40, 40), (x, y, width, height), 0)
        pygame.draw.rect(surface, self.color_white, (x, y, width, height), 2)

        # Fill level
        ratio = max(0.0, min(1.0, amount / max_amount)) if max_amount > 0 else 0.0
        fill_height = int(height * ratio)
        fill_y = y + height - fill_height

        # Color based on type and level
        if prop_type == "oxidizer":
            # Cyan theme for oxidizer
            if ratio > 0.5:
                color = self.color_cyan
            elif ratio > 0.25:
                color = self.color_orange
            else:
                color = self.color_red
        else:
            # Green theme for fuel
            if ratio > 0.5:
                color = self.color_green
            elif ratio > 0.25:
                color = self.color_orange
            else:
                color = self.color_red

        if fill_height > 0:
            pygame.draw.rect(surface, color, (x + 2, fill_y, width - 4, fill_height), 0)

        # Label
        label_text = self.font_small.render(label, True, self.color_white)
        label_rect = label_text.get_rect(centerx=x + width // 2, top=y + height + 5)
        surface.blit(label_text, label_rect)

        # Percentage
        percent_text = self.font_small.render(f"{int(ratio * 100)}%", True, self.color_white)
        percent_rect = percent_text.get_rect(centerx=x + width // 2, top=y + height + 20)
        surface.blit(percent_text, percent_rect)

    def draw_cg_indicator(self, surface, cg_offset_x, max_offset=0.5, x=None, y=None):
        """
        Draw horizontal CG indicator showing lateral offset.

        Displays a centered bar with a moving indicator showing
        current CG position relative to geometric center.

        Args:
            surface: Pygame surface
            cg_offset_x: Current CG offset (negative=left, positive=right)
            max_offset: Maximum expected offset for scaling
            x, y: Position (defaults to bottom-left area)
        """
        if x is None:
            x = 10
        if y is None:
            y = self.screen_height - 50

        bar_width = 100
        bar_height = 12

        # Label
        label_text = self.font_small.render("CG OFFSET", True, self.color_white)
        surface.blit(label_text, (x, y - 18))

        # Background bar
        pygame.draw.rect(surface, (40, 40, 40), (x, y, bar_width, bar_height), 0)
        pygame.draw.rect(surface, self.color_white, (x, y, bar_width, bar_height), 2)

        # Center line
        center_x = x + bar_width // 2
        pygame.draw.line(surface, self.color_gray, (center_x, y), (center_x, y + bar_height), 2)

        # CG marker position
        normalized = cg_offset_x / max_offset if max_offset > 0 else 0.0
        normalized = max(-1.0, min(1.0, normalized))
        marker_x = center_x + int(normalized * (bar_width // 2 - 5))

        # Color based on offset magnitude
        if abs(normalized) < 0.3:
            marker_color = self.color_green
        elif abs(normalized) < 0.6:
            marker_color = self.color_yellow
        else:
            marker_color = self.color_red

        # Draw marker
        pygame.draw.rect(surface, marker_color, (marker_x - 3, y + 2, 6, bar_height - 4), 0)

        # Numeric value
        value_text = self.font_small.render(f"{cg_offset_x:+.2f}m", True, self.color_white)
        surface.blit(value_text, (x + bar_width + 10, y - 2))

    def draw_gimbal_breakdown(self, surface, total_gimbal, auto_gimbal, manual_gimbal, x=None, y=None):
        """
        Draw gimbal indicator with auto/manual breakdown.

        Args:
            surface: Pygame surface
            total_gimbal: Combined gimbal angle (degrees)
            auto_gimbal: Auto CG compensation component
            manual_gimbal: Manual player input component
            x, y: Position (defaults to bottom-left area)
        """
        if x is None:
            x = 10
        if y is None:
            y = self.screen_height - 130

        line_height = 16

        # Total gimbal
        total_color = self.color_yellow if abs(total_gimbal) > 0.5 else self.color_gray
        text = self.font_small.render(f"GIMBAL: {total_gimbal:+5.1f}\u00b0", True, total_color)
        surface.blit(text, (x, y))
        y += line_height

        # Auto component (cyan when active)
        auto_color = self.color_cyan if abs(auto_gimbal) > 0.1 else self.color_gray
        text = self.font_small.render(f"  Auto:  {auto_gimbal:+5.1f}\u00b0", True, auto_color)
        surface.blit(text, (x, y))
        y += line_height

        # Manual component (yellow when active)
        manual_color = self.color_yellow if abs(manual_gimbal) > 0.1 else self.color_gray
        text = self.font_small.render(f"  Trim:  {manual_gimbal:+5.1f}\u00b0", True, manual_color)
        surface.blit(text, (x, y))

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
