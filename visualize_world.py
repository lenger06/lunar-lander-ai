"""Visual demo of ApolloWorld with WorldObjects.

This creates a pygame window showing:
- The cylindrical world
- Background stars
- Terrain with landing pads
- Apollo spacecraft modules (Descent, Ascent, CSM)
"""

import sys
import pygame
from Box2D import b2Vec2, b2WeldJointDef
from world import ApolloWorld, PLANET_PROPERTIES
from apollo_descent_module import ApolloDescent
from apollo_ascent_module import ApolloAscent
from apollo_service_module import ServiceModule
from apollo_command_module import CommandModule
from satellite import Satellite
from mini_map import MiniMap


def main():
    # Parse command-line arguments for planet selection
    selected_planet = "luna"  # Default to Moon
    if len(sys.argv) > 1:
        planet_arg = sys.argv[1].lower()
        if planet_arg in PLANET_PROPERTIES:
            selected_planet = planet_arg
        else:
            print(f"Unknown planet '{sys.argv[1]}'. Available planets:")
            for planet in PLANET_PROPERTIES.keys():
                print(f"  - {planet}")
            print(f"Using default: {selected_planet}")

    # Initialize pygame
    pygame.init()

    # Screen settings
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Apollo World Visualization - {selected_planet.upper()}")

    # Physics settings
    PPM = 22.0  # Pixels per meter
    FPS = 60
    dt = 1.0 / FPS
    clock = pygame.time.Clock()

    # Create world with selected planet
    print(f"\nCreating {selected_planet.upper()} world...")
    world = ApolloWorld(planet_name=selected_planet, screen_width_pixels=SCREEN_WIDTH, ppm=PPM, difficulty=1)
    info = world.get_info()
    print(f"World: {info['planet']}")
    print(f"  Size: {info['diameter_m']:.1f}m ({info['world_screens']:.1f} screens wide)")
    print(f"  Gravity: {info['gravity']} m/s²")
    if info['linear_damping'] == 0.0:
        print(f"  Atmosphere: None")
    else:
        print(f"  Atmosphere: Linear damping {info['linear_damping']}, Angular damping {info['angular_damping']}")
    print(f"  Terrain: {len(world.terrain_points)} points, {len(world.pads_info)} landing pads")

    # Find a landing pad to place spacecraft near
    pad = world.pads_info[0]
    pad_center_x = (pad['x1'] + pad['x2']) / 2.0
    pad_y = pad['y']

    # Create descent stage on landing pad
    print("\nCreating spacecraft...")
    descent = ApolloDescent(world, position=b2Vec2(pad_center_x, pad_y + 5), scale=1.0)
    world.add_object(descent)
    print(f"  Descent module at ({descent.body.position.x:.1f}, {descent.body.position.y:.1f})")

    # Create ascent stage stacked on descent
    # Descent top is at: pad_y + 5 + 0.8 (main_half_h)
    # Ascent bottom extends -1.5 from center, so center should be at: (pad_y + 5.8) + 1.5
    ascent = ApolloAscent(world, position=b2Vec2(pad_center_x, pad_y + 7.3), scale=1.0)
    world.add_object(ascent)
    print(f"  Ascent module at ({ascent.body.position.x:.1f}, {ascent.body.position.y:.1f})")

    # Weld ascent and descent together to form lander
    lander_joint_def = b2WeldJointDef()
    anchor = (descent.body.worldCenter + ascent.body.worldCenter) * 0.5
    lander_joint_def.Initialize(descent.body, ascent.body, anchor)
    lander_joint_def.collideConnected = False
    lander_joint = world.b2world.CreateJoint(lander_joint_def)
    print(f"  Lander stages welded together (press SPACE to separate)")

    # Create Service Module in orbit at orbital altitude with orbital velocity
    import math
    sm_x = pad_center_x + 50
    sm_y = world.orbit_altitude  # Place at exact orbital altitude
    sm = ServiceModule(world, position=(sm_x, sm_y), scale=1.0)
    world.add_object(sm)

    # Rotate SM 90 degrees counter-clockwise (π/2 radians)
    sm.body.angle = math.pi / 2

    # Calculate orbital velocity using the same method as apollolandergame.py
    # Step 1: Get surface gravity and calculate orbital radius
    gravity_surface = abs(world.gravity)  # Surface gravity (m/s²)
    planet_radius = world.width / 2.0  # R = diameter / 2
    orbital_radius = planet_radius + world.orbit_altitude

    # Step 2: Calculate gravity at orbital altitude using inverse square law
    # g(h) = g₀ * (R / (R + h))²
    gravity_at_orbit = gravity_surface * (planet_radius / orbital_radius) ** 2

    # Step 3: Calculate orbital velocity for circular orbit: v = sqrt(g * r)
    # where g is the gravity at orbital altitude and r is the orbital radius
    orbital_velocity = math.sqrt(gravity_at_orbit * orbital_radius)

    # Calculate orbital circumference and period (for display/debugging)
    orbital_circumference_game = 2.0 * math.pi * orbital_radius
    orbital_period_seconds = orbital_circumference_game / orbital_velocity
    orbital_period_minutes = orbital_period_seconds / 60.0

    # Set CSM initial velocity (horizontal, moving right across the screen)
    # NO margin applied - use exact orbital velocity as calculated
    sm.body.linearVelocity = b2Vec2(orbital_velocity, 0.0)

    # Note: gravityScale is managed by world.update() using inverse square law
    # Don't override it here - let the physics system handle it dynamically

    print(f"  Service Module at ({sm.body.position.x:.1f}, {sm.body.position.y:.1f})")
    print(f"    Orbital velocity: {orbital_velocity:.2f} m/s (horizontal)")
    print(f"    Orbital period: {orbital_period_minutes:.1f} minutes ({orbital_period_minutes/60:.2f} hours)")
    print(f"    Gravity at orbit: {gravity_at_orbit:.3f} m/s² ({gravity_at_orbit/gravity_surface*100:.1f}% of surface)")
    print(f"    SM angle: {sm.body.angle * 180 / math.pi:.1f}° (90° counter-clockwise)")

    # Create Command Module stacked on SM
    # Both rotated 90° CCW
    # SM is now horizontal: engine pointing right (+X), docking side pointing left (-X)
    # SM half-height is 3.8m, so SM's docking side (top) is at: sm_x - 3.8
    #
    # CM geometry after 90° CCW rotation:
    # - Heat shield (originally at y=0) points LEFT (-X direction)
    # - Docking port (originally at y=4.15) points RIGHT (+X direction)
    # - CM's center is the origin of its local frame
    # - CM's heat shield (leftmost edge) is at: cm_x - 4.15m
    # - CM's docking port (rightmost edge) is at: cm_x + 4.15m
    #
    # WAIT - I had the rotation backwards!
    # After 90° CCW rotation, the original +Y axis becomes +X axis
    # So: heat shield (y=0) → left (x=0-4.15), docking port (y=4.15) → right (x=0+4.15)
    # NO - the CM body is centered at (0,0) in local coords, extends from base (y=0) to tip (y=4.15)
    # After rotation, this becomes: left edge at cm_x + 0 = cm_x, right edge at cm_x + 4.15
    #
    # For exact touching (CM right edge = SM left edge): cm_x + 4.15 = sm_x - 3.8
    # But wait - I need to reconsider where the CM local origin is...
    # Position CM at SM's docking position (align CM cone tip to yellow band LEFT edge)
    # sm_band_top center Y: 3.8 - 0.15 - 0.15 = 3.5 (local coords before rotation)
    # sm_band_top half-height: 0.15
    # After 90° CCW rotation:
    #   - Band LEFT edge (outer, the YELLOW visible edge): sm_x - 3.65
    #
    # CM geometry (before rotation):
    #   - Triangle: base at y=0 (heat shield), tip at y=3.65
    #   - Body center (centroid) at y=1.217
    #   - Cone tip offset from centroid: 3.65 - 1.217 = 2.433
    #
    # After 90° CCW rotation: local (0, y) → world (-y, x)
    #   - Body center at world X = cm_x
    #   - Cone tip at local offset +2.433 from centroid → world X = cm_x - 2.433
    #
    # Simplified: Just position CM to touch SM's left edge
    # SM left edge (after rotation) is at: sm_x - 3.8
    # Try positioning CM close to this point
    cm_x = sm_x - 3.9  # Position CM near SM's left edge (fine-tuned)
    cm_y = sm_y  # Same altitude as SM
    cm = CommandModule(world, position=(cm_x, cm_y), scale=1.0)
    world.add_object(cm)

    # Rotate CM 90 degrees counter-clockwise to match SM orientation
    cm.body.angle = math.pi / 2

    # Set velocities to match SM (both linear and angular)
    cm.body.linearVelocity = b2Vec2(orbital_velocity, 0.0)  # Match SM velocity (horizontal)
    cm.body.angularVelocity = 0.0  # No rotation
    sm.body.angularVelocity = 0.0  # No rotation

    print(f"  Command Module at ({cm.body.position.x:.1f}, {cm.body.position.y:.1f})")
    print(f"    CM angle: {cm.body.angle * 180 / math.pi:.1f}° (90° counter-clockwise)")

    # Weld SM and CM together at their combined center of mass to avoid torque
    csm_joint_def = b2WeldJointDef()
    # Calculate combined center of mass
    total_mass = sm.body.mass + cm.body.mass
    com_x = (sm.body.mass * sm.body.worldCenter.x + cm.body.mass * cm.body.worldCenter.x) / total_mass
    com_y = (sm.body.mass * sm.body.worldCenter.y + cm.body.mass * cm.body.worldCenter.y) / total_mass
    csm_anchor = b2Vec2(com_x, com_y)
    csm_joint_def.Initialize(sm.body, cm.body, csm_anchor)
    csm_joint_def.collideConnected = False
    csm_joint = world.b2world.CreateJoint(csm_joint_def)
    print(f"  SM and CM welded together at combined COM (press Shift+SPACE to separate)")
    print(f"    CSM COM at ({com_x:.1f}, {com_y:.1f})")
    print(f"  DEBUG: SM Box2D mass = {sm.body.mass:.2f}, CM Box2D mass = {cm.body.mass:.2f}")
    print(f"  DEBUG: SM worldCenter = ({sm.body.worldCenter.x:.2f}, {sm.body.worldCenter.y:.2f})")
    print(f"  DEBUG: CM worldCenter = ({cm.body.worldCenter.x:.2f}, {cm.body.worldCenter.y:.2f})")

    # Store CSM COM for visualization
    csm_com_position = b2Vec2(com_x, com_y)

    # Orbital mechanics debugging - track CSM altitude and velocity over time
    orbital_debug_log = []
    debug_interval = 2.0  # Log every 2 seconds
    last_debug_time = 0.0
    elapsed_time = 0.0

    # Create Satellite moving across the sky
    sat_x = pad_center_x - 30  # Start to the left
    sat_y = world.orbit_altitude * 0.5  # Middle altitude
    satellite = Satellite(world, position=(sat_x, sat_y), velocity=10.0, scale=1.0)
    world.add_object(satellite)
    print(f"  Satellite at ({satellite.body.position.x:.1f}, {satellite.body.position.y:.1f}) moving at {satellite.velocity} m/s")

    # Create mini-map (top-right corner) matching apollolandergame.py dimensions
    minimap_width = 600  # Tripled from 200 for better proportions
    minimap_height = 120
    minimap_x = SCREEN_WIDTH - minimap_width - 10
    minimap_y = 10
    minimap = MiniMap(x=minimap_x, y=minimap_y, width=minimap_width, height=minimap_height, world=world)
    print(f"  Mini-map created at ({minimap_x}, {minimap_y}) size {minimap_width}x{minimap_height}")

    # Camera setup (focus on SM initially so we can control it)
    camera_x = sm.body.position.x
    camera_y = sm.body.position.y

    # Font for info display
    font = pygame.font.SysFont("monospace", 14)

    # Main loop
    running = True
    follow_camera = True
    prev_follow_camera = True  # Track previous state to detect mode transitions
    stages_separated = False
    csm_separated = False
    control_target = "csm"  # "csm" or "lander"

    print("\n" + "=" * 60)
    print("Controls:")
    print("  SPACE: Separate current vehicle's modules")
    print("         (CSM mode: separate SM/CM, Lander mode: separate Descent/Ascent)")
    print("  BACKSPACE: Switch control between CSM and Lander")
    print("  Numpad 5: Toggle flight/camera mode")
    print("")
    print("THRUSTER CONTROLS (for active vehicle):")
    print("  UP Arrow: Main engine")
    print("  LEFT/RIGHT Arrows: Translate left/right (combined RCS)")
    print("  W/S: Translate up/down (both up/down RCS thrusters)")
    print("  Q/E: Rotate counter-clockwise/clockwise")
    print("")
    print("CAMERA MODE:")
    print("  Numpad 8/2/4/6: Pan camera up/down/left/right")
    print("  Arrow keys still control thrusters in camera mode")
    print("")
    print("  ESC: Exit")
    print("=" * 60)

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # SPACE: Separate modules based on current control target
                    if control_target == "csm":
                        # Separate CSM (SM and CM)
                        if not csm_separated and csm_joint:
                            world.b2world.DestroyJoint(csm_joint)
                            csm_joint = None
                            csm_separated = True
                            print("CSM separated! (SM and CM are now independent)")
                            print("Note: CM has no thrusters - it will drift freely")
                    else:  # control_target == "lander"
                        # Separate lander stages (descent and ascent)
                        if not stages_separated and lander_joint:
                            world.b2world.DestroyJoint(lander_joint)
                            lander_joint = None
                            stages_separated = True
                            print("Lander stages separated! (Descent and Ascent are now independent)")
                elif event.key == pygame.K_KP5:
                    follow_camera = not follow_camera
                    print(f"{'FLIGHT' if follow_camera else 'CAMERA'} mode")
                elif event.key == pygame.K_BACKSPACE:
                    control_target = "lander" if control_target == "csm" else "csm"
                    print(f"Control switched to: {control_target.upper()}")

        # Get keyboard state once
        keys = pygame.key.get_pressed()

        # Clear thrust states from previous frame for all vehicles
        sm.main_thrusting = False
        sm.active_thrusters.clear()
        ascent.main_thrusting = False
        ascent.active_thrusters.clear()
        descent.main_thrusting = False
        descent.active_thrusters.clear()

        # Camera controls (numpad 8/2/4/6)
        if not follow_camera:
            camera_speed = 2.0
            if keys[pygame.K_KP4]:  # Numpad 4 - pan left
                camera_x -= camera_speed
            if keys[pygame.K_KP6]:  # Numpad 6 - pan right
                camera_x += camera_speed
            if keys[pygame.K_KP8]:  # Numpad 8 - pan up
                camera_y += camera_speed
            if keys[pygame.K_KP2]:  # Numpad 2 - pan down
                camera_y -= camera_speed

            # Normalize camera X position for cylindrical wrapping
            camera_x = camera_x % world.width

        # Thruster controls - apply to the selected control target
        if control_target == "csm":
            # CSM is always controlled with SM thrusters
            active_vehicle = sm

            # UP arrow: Main engine
            if keys[pygame.K_UP]:
                active_vehicle.apply_main_thrust(dt)
                active_vehicle.main_thrusting = True
            else:
                active_vehicle.main_thrusting = False

            # RCS thruster controls
            rcs_active = []

            # LEFT arrow: Right side thruster only (pushes left)
            if keys[pygame.K_LEFT]:
                rcs_active.append("R_SIDE")

            # RIGHT arrow: Left side thruster only (pushes right)
            if keys[pygame.K_RIGHT]:
                rcs_active.append("L_SIDE")
            # W: Both up thrusters (pushes down)
            if keys[pygame.K_w]:
                rcs_active.extend(["L_UP", "R_UP"])

            # S: Both down thrusters (pushes up)
            if keys[pygame.K_s]:
                rcs_active.extend(["L_DOWN", "R_DOWN"])

            # Q: Left up + Right down (rotate counter-clockwise)
            if keys[pygame.K_q]:
                rcs_active.extend(["L_UP", "R_DOWN"])

            # E: Right up + Left down (rotate clockwise)
            if keys[pygame.K_e]:
                rcs_active.extend(["R_UP", "L_DOWN"])

            # Apply RCS thrust to active vehicle
            if rcs_active:
                active_vehicle.apply_rcs_thrust(rcs_active, dt)
                active_vehicle.active_thrusters = set(rcs_active)
            else:
                active_vehicle.active_thrusters.clear()

        else:  # control_target == "lander"
            # Lander control depends on separation state
            if not stages_separated:
                # Stages connected: Use descent engine for main thrust + ascent RCS
                # (Real Apollo: descent engine for thrust, ascent RCS for attitude control)

                # UP arrow: Descent main engine
                if keys[pygame.K_UP]:
                    descent.apply_main_thrust(dt)
                    descent.main_thrusting = True
                else:
                    descent.main_thrusting = False

                # RCS thruster controls (ascent module RCS)
                rcs_active = []

                # LEFT arrow: Right side thruster only (pushes left)
                if keys[pygame.K_LEFT]:
                    rcs_active.append("R_SIDE")

                # RIGHT arrow: Left side thruster only (pushes right)
                if keys[pygame.K_RIGHT]:
                    rcs_active.append("L_SIDE")
                # W: Both up thrusters (pushes down)
                if keys[pygame.K_w]:
                    rcs_active.extend(["L_UP", "R_UP"])

                # S: Both down thrusters (pushes up)
                if keys[pygame.K_s]:
                    rcs_active.extend(["L_DOWN", "R_DOWN"])

                # Q: Left up + Right down (rotate counter-clockwise)
                if keys[pygame.K_q]:
                    rcs_active.extend(["L_UP", "R_DOWN"])

                # E: Right up + Left down (rotate clockwise)
                if keys[pygame.K_e]:
                    rcs_active.extend(["R_UP", "L_DOWN"])

                # Apply RCS thrust to ascent module (even when connected)
                if rcs_active:
                    ascent.apply_rcs_thrust(rcs_active, dt)
                    ascent.active_thrusters = set(rcs_active)
                else:
                    ascent.active_thrusters.clear()

            else:
                # Stages separated: Use ascent engine and RCS
                # (Real Apollo: ascent engine used after liftoff from moon)
                active_vehicle = ascent

                # UP arrow: Main engine
                if keys[pygame.K_UP]:
                    active_vehicle.apply_main_thrust(dt)
                    active_vehicle.main_thrusting = True
                else:
                    active_vehicle.main_thrusting = False

                # RCS thruster controls
                rcs_active = []

                # LEFT arrow: Right side thruster only (pushes left)
                if keys[pygame.K_LEFT]:
                    rcs_active.append("R_SIDE")

                # RIGHT arrow: Left side thruster only (pushes right)
                if keys[pygame.K_RIGHT]:
                    rcs_active.append("L_SIDE")

                # W: Both up thrusters (pushes down)
                if keys[pygame.K_w]:
                    rcs_active.extend(["L_UP", "R_UP"])

                # S: Both down thrusters (pushes up)
                if keys[pygame.K_s]:
                    rcs_active.extend(["L_DOWN", "R_DOWN"])

                # Q: Left up + Right down (rotate counter-clockwise)
                if keys[pygame.K_q]:
                    rcs_active.extend(["L_UP", "R_DOWN"])

                # E: Right up + Left down (rotate clockwise)
                if keys[pygame.K_e]:
                    rcs_active.extend(["R_UP", "L_DOWN"])

                # Apply RCS thrust to active vehicle
                if rcs_active:
                    active_vehicle.apply_rcs_thrust(rcs_active, dt)
                    active_vehicle.active_thrusters = set(rcs_active)
                else:
                    active_vehicle.active_thrusters.clear()

        # Update physics
        world.update(dt)

        # Orbital mechanics debugging - track CSM state periodically
        elapsed_time += dt
        if elapsed_time - last_debug_time >= debug_interval:
            # Calculate current orbital parameters
            sm_alt = sm.body.position.y
            sm_vel_x = sm.body.linearVelocity.x
            sm_vel_y = sm.body.linearVelocity.y
            sm_speed = math.sqrt(sm_vel_x**2 + sm_vel_y**2)

            # Calculate current gravity scale and acceleration
            planet_radius = world.width / 2.0
            if sm_alt < 0:
                grav_scale = 1.0
            else:
                distance_from_center = planet_radius + sm_alt
                grav_scale = (planet_radius / distance_from_center) ** 2

            gravity_surface = abs(world.gravity)
            vert_accel = world.gravity * grav_scale  # Already negative

            # Log data
            debug_entry = {
                "time": elapsed_time,
                "altitude": sm_alt,
                "vel_x": sm_vel_x,
                "vel_y": sm_vel_y,
                "speed": sm_speed,
                "grav_scale": grav_scale,
                "vert_accel": vert_accel,
                "sm_gravityScale": sm.body.gravityScale,  # What Box2D actually has
                "cm_gravityScale": cm.body.gravityScale,
            }
            orbital_debug_log.append(debug_entry)

            # Print to console
            print(f"[T={elapsed_time:6.1f}s] Alt={sm_alt:6.2f}m | Vel=({sm_vel_x:+6.2f}, {sm_vel_y:+6.2f}) m/s | Speed={sm_speed:6.2f} | GravScale={grav_scale:.6f} (SM:{sm.body.gravityScale:.6f}, CM:{cm.body.gravityScale:.6f}) | Accel={vert_accel:+7.4f} m/s²")

            last_debug_time = elapsed_time

        # Camera position is independent - stays where user positioned it
        # No automatic following of descent module
        # (User can manually pan camera in free mode with arrow keys)

        # Update previous state for next frame
        prev_follow_camera = follow_camera

        # Render
        screen.fill((0, 0, 0))  # Black background

        # Draw stars
        world.draw_stars(screen, camera_x, camera_y, SCREEN_WIDTH, SCREEN_HEIGHT, PPM)

        # Draw terrain
        world.draw_terrain(screen, camera_x, camera_y, SCREEN_WIDTH, SCREEN_HEIGHT, PPM)

        # Draw objects
        world.draw_objects(screen, camera_x, camera_y, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Draw CSM center of mass indicator (yellow crosshair)
        # Recalculate COM position dynamically based on current body positions
        def w2s(pos):
            """World to screen conversion."""
            x = SCREEN_WIDTH / 2 + (pos.x - camera_x) * PPM
            y = SCREEN_HEIGHT / 2 - (pos.y - camera_y) * PPM
            return int(x), int(y)

        # Calculate current combined center of mass
        total_mass = sm.body.mass + cm.body.mass
        current_com_x = (sm.body.mass * sm.body.worldCenter.x + cm.body.mass * cm.body.worldCenter.x) / total_mass
        current_com_y = (sm.body.mass * sm.body.worldCenter.y + cm.body.mass * cm.body.worldCenter.y) / total_mass
        current_com = b2Vec2(current_com_x, current_com_y)

        com_screen = w2s(current_com)
        COM_COLOR = (255, 255, 0)  # Yellow
        COM_RADIUS = 8
        COM_LINE_LENGTH = 16

        # Draw circle
        pygame.draw.circle(screen, COM_COLOR, com_screen, COM_RADIUS, 2)

        # Draw vertical line (crosshair)
        pygame.draw.line(screen, COM_COLOR,
                        (com_screen[0], com_screen[1] - COM_LINE_LENGTH),
                        (com_screen[0], com_screen[1] + COM_LINE_LENGTH), 2)

        # Draw horizontal line (crosshair)
        pygame.draw.line(screen, COM_COLOR,
                        (com_screen[0] - COM_LINE_LENGTH, com_screen[1]),
                        (com_screen[0] + COM_LINE_LENGTH, com_screen[1]), 2)

        # Draw engine flames for lander modules
        # Descent engine flame (when stages connected or when descent is firing)
        descent.draw_engine(
            screen,
            main_on=descent.main_thrusting,
            gimbal_angle_deg=0.0,
            cam_x=camera_x,
            cam_y=camera_y,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
        )

        # Ascent engine and RCS flames (always render RCS, main engine only when separated)
        # Convert active_thrusters set to boolean flags for draw_engine
        active_list = list(ascent.active_thrusters)
        ascent.draw_engine(
            screen,
            main_on=ascent.main_thrusting,  # Main engine only fires when separated
            tl="L_UP" in active_list,
            bl="L_DOWN" in active_list,
            tr="R_UP" in active_list,
            br="R_DOWN" in active_list,
            sl="L_SIDE" in active_list,
            sr="R_SIDE" in active_list,
            cam_x=camera_x,
            cam_y=camera_y,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
        )

        # Draw info overlay
        # Calculate fuel percentages
        descent_fuel_pct = (descent.fuel_kg / descent.max_fuel_kg) * 100
        ascent_fuel_pct = (ascent.fuel_kg / ascent.max_fuel_kg) * 100
        sm_fuel_pct = (sm.fuel_kg / sm.max_fuel_kg) * 100

        info_lines = [
            f"World: {info['planet']} ({info['diameter_m']:.0f}m wide)",
            f"Gravity: {info['gravity']} m/s²",
            f"",
            f"Descent:   ({descent.body.position.x:.1f}, {descent.body.position.y:.1f}) mass={descent.body.mass:.1f}",
            f"  Fuel: {descent.fuel_kg:.0f}/{descent.max_fuel_kg:.0f} kg ({descent_fuel_pct:.1f}%)",
            f"Ascent:    ({ascent.body.position.x:.1f}, {ascent.body.position.y:.1f}) mass={ascent.body.mass:.1f}",
            f"  Fuel: {ascent.fuel_kg:.0f}/{ascent.max_fuel_kg:.0f} kg ({ascent_fuel_pct:.1f}%)",
            f"SM:        ({sm.body.position.x:.1f}, {sm.body.position.y:.1f}) mass={sm.body.mass:.1f}",
            f"  Fuel: {sm.fuel_kg:.0f}/{sm.max_fuel_kg:.0f} kg ({sm_fuel_pct:.1f}%)",
            f"CM:        ({cm.body.position.x:.1f}, {cm.body.position.y:.1f}) mass={cm.body.mass:.1f}",
            f"Satellite: ({satellite.body.position.x:.1f}, {satellite.body.position.y:.1f}) vel={satellite.velocity:.1f}",
            f"",
            f"Camera: ({camera_x:.1f}, {camera_y:.1f})",
            f"Lander: {'SEPARATED' if stages_separated else 'WELDED'}",
            f"CSM:    {'SEPARATED' if csm_separated else 'WELDED'}",
            f"",
            f"Controlling: [{'CSM' if control_target == 'csm' else 'LANDER'}]  (BACKSPACE to switch)",
            f"Mode: {'[FLIGHT]' if follow_camera else '[CAMERA] Numpad 8/2/4/6=Pan'}  Arrows=Thrust W/S/Q/E=RCS",
            f"SPACE=Sep.Lander  Shift+SPACE=Sep.CSM  Numpad5=Toggle",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            # Draw shadow for readability
            screen.blit(font.render(line, True, (0, 0, 0)), (11, y_offset + 1))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 18

        # Draw mini-map
        objects_dict = {
            'descent': descent,
            'ascent': ascent,
            'sm': sm,
            'cm': cm,
            'satellite': satellite
        }
        minimap.draw(screen, camera_x, camera_y, SCREEN_WIDTH, SCREEN_HEIGHT, PPM, objects_dict)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("\nVisualization closed.")


if __name__ == "__main__":
    main()
