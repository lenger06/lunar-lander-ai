# World Architecture

## Overview

The `world.py` module provides a world-centric architecture for the Apollo Lunar Lander game. Instead of managing world physics, terrain, and objects separately in the game loop, everything is encapsulated in an `ApolloWorld` class that acts as a container.

### Related Files

| File | Role |
|------|------|
| `world.py` | ApolloWorld class, WorldObject base, planet properties |
| `apollo_terrain.py` | Procedural terrain generation (3 difficulty levels, landing pads) |
| `apollo_sky.py` | Star field rendering |
| `apollolander.py` | Lunar Module orchestrator (uses WorldObject pattern) |
| `apollo_descent_module.py` | Descent stage WorldObject |
| `apollo_ascent_module.py` | Ascent stage WorldObject |
| `apollo_csm_module.py` | CSM WorldObject wrapper |
| `apollo_command_module.py` | Command module geometry & rendering |
| `apollo_service_module.py` | Service module (SPS engine, RCS) |
| `apollo_rcs_pod.py` | RCS thruster pod (quad thrusters) |

## Key Design Principles

1. **World as Container**: The world owns and manages all physics, terrain, and objects
2. **Planet-Based Physics**: Physics properties (gravity, damping) are derived from the planet properties table
3. **Cylindrical Wrapping**: The world wraps horizontally to simulate orbiting around a planet
4. **Object-Oriented**: All things in the world (CSM, lander, stars, etc.) inherit from `WorldObject`
5. **Easy Extension**: Adding new objects to the world is straightforward using the `add_object()` method

## Core Classes

### `PLANET_PROPERTIES` (Dictionary)

Defines physics properties for 9 different planets/moons:

```python
PLANET_PROPERTIES = {
    "mercury": {...},
    "venus": {...},
    "earth": {...},
    "luna": {...},     # The Moon
    "mars": {...},
    "jupiter": {...},
    "saturn": {...},
    "uranus": {...},
    "neptune": {...},
}
```

Each planet definition includes:
- `gravity`: Surface gravity in m/s²
- `linear_damping`: Atmospheric drag (0.0 for no atmosphere)
- `angular_damping`: Rotational drag (0.0 for no atmosphere)
- `diameter`: Planet diameter in km
- `orbit_alt`: Low orbit altitude in km

### `WorldObject` (Base Class)

Base class for all objects that exist in the world.

**Methods:**
- `update(dt)`: Update object state (called each frame)
- `draw(surface, cam_x, cam_y, screen_width, screen_height)`: Render object to screen
- `handle_cylindrical_wrapping(body)`: Helper to wrap Box2D bodies around the world

**Usage:**
```python
class MyObject(WorldObject):
    def __init__(self, world):
        super().__init__(world)
        # Create Box2D body in world.b2world

    def update(self, dt):
        # Update physics, handle wrapping, etc.
        self.handle_cylindrical_wrapping(self.body)

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        # Render to pygame surface
        pass
```

### `ApolloWorld` (Main World Class)

The main world container that manages everything.

**Constructor:**
```python
world = ApolloWorld(
    planet_name="luna",      # Which planet from PLANET_PROPERTIES
    screen_width_pixels=1600, # For calculating world size
    ppm=22.0,                # Pixels per meter (for scaling)
    difficulty=1             # Terrain difficulty (1-3)
)
```

**Key Properties:**
- `planet_name`: Name of the planet
- `planet_props`: Dictionary of planet properties
- `width`: World width in meters (= planet diameter / 1000)
- `orbit_altitude`: Orbital altitude in meters
- `gravity`: Gravity value (negative for downward)
- `linear_damping`, `angular_damping`: Physics damping
- `b2world`: Box2D physics world instance
- `terrain_body`: Box2D static body for terrain
- `terrain_points`: List of (x, y) terrain coordinates
- `pads_info`: List of landing pad definitions
- `stars`: List of (x, y, brightness) for background stars
- `objects`: List of WorldObject instances

**Key Methods:**
- `add_object(obj)`: Add a WorldObject to the world
- `remove_object(obj)`: Remove a WorldObject from the world
- `update(dt)`: Step physics and update all objects
- `draw_stars(...)`: Draw background stars with cylindrical wrapping
- `draw_terrain(...)`: Draw terrain and landing pads
- `draw_objects(...)`: Draw all world objects
- `get_info()`: Get world properties as a dictionary

## World Size Calculations

The world uses a 1:1000 scale factor to convert real-world dimensions to game dimensions:

```
Game meters = Real kilometers / 1000

Examples:
- Moon diameter: 3,475 km → 3,475 m in game
- Moon orbit: 100 km → 100 m in game
- Earth diameter: 12,742 km → 12,742 m in game
```

This makes the game playable while maintaining realistic proportions.

## Cylindrical Wrapping

The world wraps horizontally to simulate a cylindrical surface:

1. **Terrain**: Generated in 3 copies (left, center, right) for seamless wrapping
2. **Stars**: Drawn in 3 panels for seamless parallax scrolling
3. **Objects**: Use `handle_cylindrical_wrapping()` to teleport when crossing boundaries

```python
# Object wrapping example
if body.position.x > world.width:
    body.position = b2Vec2(body.position.x - world.width, body.position.y)
elif body.position.x < 0:
    body.position = b2Vec2(body.position.x + world.width, body.position.y)
```

## Usage Example

### Basic World Creation and Update Loop

```python
from world import ApolloWorld

# Create world for the Moon
world = ApolloWorld(planet_name="luna", difficulty=1)

# Get world info
info = world.get_info()
print(f"Playing on {info['planet']}")
print(f"Gravity: {info['gravity']} m/s²")
print(f"World size: {info['diameter_m']}m")

# Game loop
dt = 1.0/60.0  # 60 FPS
while running:
    # Update physics and all objects
    world.update(dt)

    # Render
    screen.fill((0, 0, 0))
    world.draw_stars(screen, cam_x, cam_y, screen_width, screen_height, ppm)
    world.draw_terrain(screen, cam_x, cam_y, screen_width, screen_height, ppm)
    world.draw_objects(screen, cam_x, cam_y, screen_width, screen_height)
```

### Adding Custom Objects

```python
from world import WorldObject

class Asteroid(WorldObject):
    def __init__(self, world, position):
        super().__init__(world)
        # Create Box2D body
        self.body = world.b2world.CreateDynamicBody(position=position)
        self.body.CreateCircleFixture(radius=2.0, density=1.0)

    def update(self, dt):
        # Handle wrapping
        self.handle_cylindrical_wrapping(self.body)

    def draw(self, surface, cam_x, cam_y, screen_width, screen_height):
        # Render asteroid
        import pygame
        # ... drawing code ...

# Add asteroid to world
asteroid = Asteroid(world, position=(100, 50))
world.add_object(asteroid)
```

## Benefits of This Architecture

1. **Separation of Concerns**: World physics, terrain, and objects are cleanly separated
2. **Easy to Extend**: Adding new objects is just creating a WorldObject subclass
3. **Realistic Physics**: Planet properties table ensures accurate physics for each world
4. **Scalable**: Works for any size planet from Mercury to Jupiter
5. **Maintainable**: Clear ownership hierarchy (World → Objects → Box2D bodies)

## Spacecraft WorldObject Modules

The game includes three spacecraft modules that demonstrate the WorldObject pattern:

### `ApolloDescent` ([apollo_descent_module.py](apollo_descent_module.py))

The Apollo Lunar Module Descent Stage with landing legs and engine.

**Features:**
- Landing legs with realistic geometry
- Engine bell (descent propulsion system)
- Mass scaling to realistic Apollo values (~10,246 kg → 102.46 Box2D units)
- Gold metallic color (220, 180, 100)

**Constructor:**
```python
descent = ApolloDescent(
    world,                # ApolloWorld instance
    position=b2Vec2(x, y), # Starting position
    scale=1.0,            # Size multiplier
    descent_density=0.5,  # Body density
    leg_density=0.4,      # Landing leg density
    friction=0.9,         # Surface friction
    restitution=0.1       # Bounciness
)
```

**Properties:**
- `body`: Box2D dynamic body
- `scale`: Size scaling factor
- `color`: RGB tuple for rendering

### `ApolloAscent` ([apollo_ascent_module.py](apollo_ascent_module.py))

The Apollo Lunar Module Ascent Stage with RCS system and docking port.

**Features:**
- RCS thruster pods (4 quad clusters)
- Docking port for CSM rendezvous
- Ascent propulsion system
- Mass scaling (~4,819 kg → 48.19 Box2D units)
- Gray color (180, 180, 180)

**Constructor:**
```python
ascent = ApolloAscent(
    world,                # ApolloWorld instance
    position=b2Vec2(x, y), # Starting position
    scale=1.0,            # Size multiplier
    ascent_density=0.6,   # Body density
    friction=0.9,         # Surface friction
    restitution=0.1       # Bounciness
)
```

**Properties:**
- `body`: Box2D dynamic body
- `scale`: Size scaling factor
- `color`: RGB tuple for rendering
- RCS pod offsets for rendering thrusters

### `ApolloCSMObject` ([apollo_csm_module.py](apollo_csm_module.py))

The Command and Service Module (CSM) wrapper as a WorldObject.

**Features:**
- Wraps existing `ApolloCSM` class
- Service module with SPS engine
- Command module with docking port
- Mass scaling (~30,080 kg → 30080.00 Box2D units)
- Integrated RCS system

**Constructor:**
```python
csm = ApolloCSMObject(
    world,                # ApolloWorld instance
    position=(x, y),      # Starting position (tuple or b2Vec2)
    scale=0.75            # Size multiplier
)
```

**Properties:**
- `body`: Box2D dynamic body (from wrapped ApolloCSM)
- `csm`: Underlying ApolloCSM instance
- `scale`: Size scaling factor
- `docking_port_center`: Docking port location for rendezvous

### Mass Scaling System

All spacecraft modules use a consistent mass scaling approach:

```python
MASS_SCALE = 100.0  # 1 Box2D mass unit = 100 kg

# Real-world masses
REAL_DESCENT_MASS_KG = 10246.0  # Apollo LM descent stage
REAL_ASCENT_MASS_KG = 4819.0    # Apollo LM ascent stage
REAL_CSM_MASS_KG = 30080.0      # Apollo CSM

# Box2D masses (after scaling)
# Descent: 102.46 units
# Ascent:  48.19 units
# CSM:     300.80 units
```

This ensures realistic physics behavior while keeping Box2D simulation stable.

### Usage Example: Creating a Complete Mission

```python
from world import ApolloWorld
from apollo_descent_module import ApolloDescent
from apollo_ascent_module import ApolloAscent
from apollo_csm_module import ApolloCSMObject
from Box2D import b2Vec2

# Create lunar world
world = ApolloWorld(planet_name="luna", difficulty=1)

# Create descent stage on surface
descent = ApolloDescent(world, position=b2Vec2(100, 10), scale=1.0)
world.add_object(descent)

# Create ascent stage (stacked on descent initially)
ascent = ApolloAscent(world, position=b2Vec2(100, 15), scale=1.0)
world.add_object(ascent)

# Create CSM in orbit
csm = ApolloCSMObject(world, position=(200, 100), scale=0.75)
world.add_object(csm)

# Game loop
dt = 1.0/60.0
while running:
    world.update(dt)  # Updates all objects automatically

    # Render
    world.draw_stars(surface, cam_x, cam_y, w, h, ppm)
    world.draw_terrain(surface, cam_x, cam_y, w, h, ppm)
    world.draw_objects(surface, cam_x, cam_y, w, h)
```

## Integration with AI Training

The `ApolloWorld` and its physics are used by the training environment (`CurriculumGameEnv` in `train_in_game.py`) to ensure trained agents experience identical physics to the actual game. Key shared settings:

- Box2D solver: 10 velocity iterations, 10 position iterations
- Timestep: 1/60s
- Gravity, damping: from `PLANET_PROPERTIES` table
- Terrain: same `ApolloTerrain` generator with difficulty levels

The game (`apollolandergame_with_ai.py`) duplicates the planet properties table for CLI parsing but uses the same physics parameters.

## Future Extensions

The WorldObject architecture makes it easy to add:
- Asteroids
- Satellites
- Space stations
- Other spacecraft
- Debris fields
- Weather effects (for planets with atmosphere)

Each new object type just needs to inherit from `WorldObject` and implement `update()` and `draw()`.
