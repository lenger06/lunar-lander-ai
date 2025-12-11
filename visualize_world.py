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

    # Create Service Module in orbit
    sm_x = pad_center_x + 50
    sm_y = world.orbit_altitude * 0.8
    sm = ServiceModule(world, position=(sm_x, sm_y), scale=1.0)
    world.add_object(sm)
    print(f"  Service Module at ({sm.body.position.x:.1f}, {sm.body.position.y:.1f})")

    # Create Command Module stacked on SM
    # Both are upright (vertical orientation)
    # SM top is at sm_y + 3.8 (sm_half_h in local coords)
    # CM base (heat shield) is at local y = 0
    # So CM position should be: sm_y + 3.8 (to align CM base with SM top)
    cm_x = sm_x
    cm_y = sm_y + 3.8  # SM top edge aligns with CM base (heat shield)
    cm = CommandModule(world, position=(cm_x, cm_y), scale=1.0)
    world.add_object(cm)
    print(f"  Command Module at ({cm.body.position.x:.1f}, {cm.body.position.y:.1f})")

    # Weld SM and CM together at contact point (SM top / CM base)
    csm_joint_def = b2WeldJointDef()
    # Weld anchor at the contact surface: SM's top edge = CM's base edge
    csm_anchor = b2Vec2(sm_x, sm_y + 3.8)
    csm_joint_def.Initialize(sm.body, cm.body, csm_anchor)
    csm_joint_def.collideConnected = False
    csm_joint = world.b2world.CreateJoint(csm_joint_def)
    print(f"  SM and CM welded together (press Shift+SPACE to separate)")

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
