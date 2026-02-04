# GRAVITAS — Gravitational Pitch Engine
## Implementation Plan v1.3 (Review Fixes Applied)

**Module**: GRAVITAS
**Brand**: Lexington Audio
**Size**: 14 HP
**Category**: Generative / Sequencer
**Author**: Drift (AI agent) + Lex (human collaborator)

---

## 1. Purpose

Notes are masses orbiting in 2D space under Newtonian gravity. Positions map to pitch (X) and modulation (Y). Masses attract each other, forming stable orbits, chaotic three-body problems, or catastrophic collisions. Physics-based composition: melodies that feel *inevitable* rather than random.

**Design Philosophy**: Emergence over engineering. Simple rules (Newtonian gravity) produce complex, organic musical behavior. The physics *does* the composition.

---

## 2. Architecture Overview

Following CURSUS patterns: `LexingtonAudio::Gravitas` namespace, separate type headers, engine class, fixed-size arrays, bitmask events.

```
plugins/LexingtonAudio/src/modules/Gravitas/
├── Gravitas.cpp              # Module + Widget
├── types/
│   ├── Mass.hpp              # Mass struct (position, velocity, alive)
│   ├── CollisionMode.hpp     # Enum: Off/Merge/Bounce/Explode
│   ├── BoundaryMode.hpp      # Enum: Wrap/Bounce/SoftWall/Absorb
│   └── ScaleDef.hpp          # Musical scale definitions
├── physics/
│   ├── GravityEngine.hpp     # N-body simulation core
│   └── CollisionDetector.hpp # Collision detection + response
├── mapping/
│   ├── PitchMapper.hpp       # X-position → 1V/oct with quantization
│   └── OutputAssigner.hpp    # Mass → output channel assignment
└── display/
    └── OrbitalDisplay.hpp    # NanoVG orbital visualization
```

---

## 3. Type Definitions

### Mass (types/Mass.hpp)
```cpp
namespace LexingtonAudio {
namespace Gravitas {

struct Mass {
    // Physics state (double precision for stability)
    double x = 0.0, y = 0.0;       // position in normalized space [-1, 1]
    double vx = 0.0, vy = 0.0;     // velocity
    double mass = 1.0;              // mass value (affects gravity)
    bool alive = false;

    // Output state (float for VCV)
    float pitchCV = 0.f;           // current 1V/oct output
    float modCV = 0.f;             // current modulation output
    float gateState = 0.f;         // gate voltage

    // Smoothing for output
    float pitchSmoothed = 0.f;
    float modSmoothed = 0.f;

    // Lifecycle (smooth birth/death transitions)
    // State machine: alive && !dying = active, alive && dying = fade-out, !alive = dead slot
    float birthFade = 0.f;         // 0→1 over BIRTH_TIME after spawn
    float deathFade = 1.f;         // 1→0 over DEATH_TIME after kill
    bool dying = false;            // in death fade (alive stays TRUE during fade)
    static constexpr float BIRTH_TIME = 0.05f;   // 50ms fade in
    static constexpr float DEATH_TIME = 0.1f;     // 100ms fade out

    // Collision debounce (prevents rapid re-triggering)
    float collisionCooldown = 0.f;
    static constexpr float COLLISION_COOLDOWN_TIME = 0.02f;  // 20ms

    void kill() {
        dying = true;  // start death fade, alive stays TRUE until fade completes
    }
    void hardKill() { alive = false; dying = false; vx = vy = 0.0; deathFade = 1.f; birthFade = 0.f; }

    // Advance lifecycle fades (call each physics tick)
    void updateLifecycle(float dt) {
        if (alive && !dying && birthFade < 1.f) {
            birthFade += dt / BIRTH_TIME;
            if (birthFade > 1.f) birthFade = 1.f;
        }
        if (dying) {
            deathFade -= dt / DEATH_TIME;
            if (deathFade <= 0.f) {
                deathFade = 0.f;
                alive = false;  // NOW the slot is freed
                dying = false;
            }
        }
        if (collisionCooldown > 0.f) {
            collisionCooldown -= dt;
            if (collisionCooldown < 0.f) collisionCooldown = 0.f;
        }
    }
};

static constexpr int MAX_MASSES = 8;

} // namespace Gravitas
} // namespace LexingtonAudio
```

### CollisionMode (types/CollisionMode.hpp)
```cpp
enum class CollisionMode : uint8_t {
    Off = 0,     // Masses pass through each other
    Merge,       // Combine into single larger mass (conserve momentum)
    Bounce,      // Elastic collision with restitution coefficient
    Explode      // Both destroyed, trigger fires
};
```

### BoundaryMode (types/BoundaryMode.hpp)
```cpp
enum class BoundaryMode : uint8_t {
    Wrap = 0,    // Wrap position (creates octave jumps but continuous motion)
    Bounce,      // Reflect velocity at boundary (most musical)
    SoftWall,    // Repulsive force near edges (no hard boundary)
    Absorb       // Mass dies when it hits boundary (triggers gate-off)
};
```

### ScaleDef (types/ScaleDef.hpp)
```cpp
struct ScaleDef {
    const char* name;
    int notes[12];     // 1 = in scale, 0 = not
    int noteCount;     // number of notes in scale
};

// Predefined scales
static constexpr ScaleDef SCALES[] = {
    {"Chromatic",  {1,1,1,1,1,1,1,1,1,1,1,1}, 12},
    {"Major",      {1,0,1,0,1,1,0,1,0,1,0,1}, 7},
    {"Minor",      {1,0,1,1,0,1,0,1,1,0,1,0}, 7},
    {"Pentatonic", {1,0,1,0,1,0,0,1,0,1,0,0}, 5},
    {"Dorian",     {1,0,1,1,0,1,0,1,0,1,1,0}, 7},
    {"Phrygian",   {1,1,0,1,0,1,0,1,1,0,1,0}, 7},
    {"Lydian",     {1,0,1,0,1,0,1,1,0,1,0,1}, 7},
    {"Mixolydian", {1,0,1,0,1,1,0,1,0,1,1,0}, 7},
    {"Whole Tone", {1,0,1,0,1,0,1,0,1,0,1,0}, 6},
    {"Blues",       {1,0,0,1,0,1,1,1,0,0,1,0}, 6},
};
static constexpr int NUM_SCALES = 10;
```

---

## 4. Physics Engine (physics/GravityEngine.hpp)

### Numerical Integration: Velocity Verlet

Euler integration is **unstable** for orbital mechanics. Velocity Verlet is symplectic (conserves energy over long periods) and only marginally more expensive:

```
x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
a(t+dt) = F(x(t+dt)) / m
v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
```

### Physics Rate

**Decision: Start at control rate (1kHz), option to promote to audio rate if needed.**

Analysis (from DSP design):
- N-body with 8 masses = 28 force pairs at ~15 FLOPS each
- Audio rate (44.1kHz): 1.23M calcs/sec = **0.028% of one CPU core** — trivially cheap
- Control rate (1kHz): 28K calcs/sec — even cheaper, requires output interpolation

**Why start at 1kHz:** Simpler implementation, interpolation is natural for CV output, and 1kHz is more than adequate for pitch/modulation frequencies. If testing reveals interpolation artifacts during fast collisions, we can promote to audio rate with zero architecture changes — just remove the counter guard.

**Why audio rate is viable:** The DSP analysis proves 8-mass n-body at 44.1kHz uses 0.028% CPU. If we ever need it, the headroom is massive.

```cpp
// In process():
physicsCounter += args.sampleTime;
if (physicsCounter >= PHYSICS_DT) {  // PHYSICS_DT = 0.001 (1kHz)
    physicsCounter -= PHYSICS_DT;
    engine.simulate(PHYSICS_DT);
}
// Always: interpolate outputs
for (int i = 0; i < MAX_MASSES; i++) {
    smoothedPitch[i] += (targetPitch[i] - smoothedPitch[i]) * smoothingCoeff;
}
```

### Gravity Model

```cpp
static constexpr double SOFTENING = 0.001;   // Prevents singularities (reduced from 0.01 — less dampening at close range)
static constexpr double G_BASE = 1.23;        // Calibrated: 2s orbit period at r=0.5, M=1.0
static constexpr double MAX_VELOCITY = 2.0;   // Soft velocity limit
static constexpr double COLLISION_DIST = 0.05; // Collision threshold

void computeForces() {
    // Reset accelerations
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        ax[i] = ay[i] = 0.0;
    }

    // N-body pairwise forces
    double G = G_BASE * gravityParam;  // gravityParam: 0-2 (from knob)

    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        for (int j = i + 1; j < MAX_MASSES; j++) {
            if (!masses[j].alive) continue;

            double dx = masses[j].x - masses[i].x;
            double dy = masses[j].y - masses[i].y;
            double distSq = dx*dx + dy*dy + SOFTENING;
            double dist = sqrt(distSq);
            double force = G * masses[i].mass * masses[j].mass / distSq;

            double fx = force * dx / dist;
            double fy = force * dy / dist;

            ax[i] += fx / masses[i].mass;
            ay[i] += fy / masses[i].mass;
            ax[j] -= fx / masses[j].mass;
            ay[j] -= fy / masses[j].mass;
        }
    }
}
```

### Velocity Verlet Step

```cpp
void simulate(double dt) {
    // 1. Half-step velocity + full-step position
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        masses[i].vx += 0.5 * ax[i] * dt;
        masses[i].vy += 0.5 * ay[i] * dt;
        masses[i].x += masses[i].vx * dt;
        masses[i].y += masses[i].vy * dt;
    }

    // 2. Compute new forces at updated positions
    computeForces();

    // 3. Apply damping to accelerations (preserves Verlet symplectic property)
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        double dampScale = exp(-dampingParam * dt);  // Always in (0, 1], safe for any damping value
        ax[i] *= dampScale;
        ay[i] *= dampScale;
    }

    // 4. Complete velocity step + soft velocity limit
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        masses[i].vx += 0.5 * ax[i] * dt;
        masses[i].vy += 0.5 * ay[i] * dt;

        // Soft velocity limit (tanh — no discontinuities)
        double speed = sqrt(masses[i].vx*masses[i].vx + masses[i].vy*masses[i].vy);
        if (speed > MAX_VELOCITY * 0.8) {  // Start limiting before hard cap
            double softClamp = MAX_VELOCITY * tanh(speed / MAX_VELOCITY);
            double scale = softClamp / speed;
            masses[i].vx *= scale;
            masses[i].vy *= scale;
        }

        // Lifecycle updates (birth/death fades, collision cooldown)
        masses[i].updateLifecycle(static_cast<float>(dt));
    }

    // 4. Collision detection
    collisionEvents = detectCollisions();

    // 5. Boundary handling
    handleBoundaries();
}
```

### Collision Detection

```cpp
uint8_t detectCollisions() {
    uint8_t events = 0;
    if (collisionMode == CollisionMode::Off) return 0;

    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        for (int j = i + 1; j < MAX_MASSES; j++) {
            if (!masses[j].alive) continue;

            double dx = masses[j].x - masses[i].x;
            double dy = masses[j].y - masses[i].y;
            double dist = sqrt(dx*dx + dy*dy);

            if (dist < COLLISION_DIST
                && masses[i].collisionCooldown <= 0.f
                && masses[j].collisionCooldown <= 0.f) {
                events |= COLLISION_OCCURRED;
                masses[i].collisionCooldown = Mass::COLLISION_COOLDOWN_TIME;
                masses[j].collisionCooldown = Mass::COLLISION_COOLDOWN_TIME;
                switch (collisionMode) {
                    case CollisionMode::Merge:
                        // Conserve momentum: p = m1*v1 + m2*v2
                        masses[i].vx = (masses[i].mass * masses[i].vx + masses[j].mass * masses[j].vx)
                                       / (masses[i].mass + masses[j].mass);
                        masses[i].vy = (masses[i].mass * masses[i].vy + masses[j].mass * masses[j].vy)
                                       / (masses[i].mass + masses[j].mass);
                        masses[i].mass += masses[j].mass;
                        masses[j].kill();
                        break;
                    case CollisionMode::Bounce: {
                        // Elastic collision
                        double nx = dx / dist, ny = dy / dist;
                        double dvx = masses[i].vx - masses[j].vx;
                        double dvy = masses[i].vy - masses[j].vy;
                        double dvn = dvx * nx + dvy * ny;
                        if (dvn > 0) break; // Already separating
                        double m1 = masses[i].mass, m2 = masses[j].mass;
                        double impulse = 2.0 * dvn / (m1 + m2);
                        masses[i].vx -= impulse * m2 * nx;
                        masses[i].vy -= impulse * m2 * ny;
                        masses[j].vx += impulse * m1 * nx;
                        masses[j].vy += impulse * m1 * ny;
                        break;
                    }
                    case CollisionMode::Explode: {
                        // Kill originals, spawn 2-3 fragments at collision point
                        double cx = (masses[i].x + masses[j].x) * 0.5;
                        double cy = (masses[i].y + masses[j].y) * 0.5;
                        double totalMass = masses[i].mass + masses[j].mass;
                        masses[i].kill();
                        masses[j].kill();
                        int spawned = 0;
                        for (int k = 0; k < MAX_MASSES && spawned < 2; k++) {
                            if (!masses[k].alive && !masses[k].dying) {
                                double angle = (spawned * 2.094) + randomUniform() * 0.5;
                                double speed = 0.3 + randomUniform() * 0.3;
                                masses[k].x = cx; masses[k].y = cy;
                                masses[k].vx = speed * cos(angle);
                                masses[k].vy = speed * sin(angle);
                                masses[k].mass = totalMass * 0.5;
                                masses[k].alive = true;
                                masses[k].birthFade = 0.f;
                                spawned++;
                            }
                        }
                        break;
                    }
                    default: break;
                }
            }
        }
    }
    return events;
}
```

### Boundary Handling

```cpp
void handleBoundaries() {
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        switch (boundaryMode) {
            case BoundaryMode::Wrap:
                if (masses[i].x > 1.0) masses[i].x -= 2.0;
                if (masses[i].x < -1.0) masses[i].x += 2.0;
                if (masses[i].y > 1.0) masses[i].y -= 2.0;
                if (masses[i].y < -1.0) masses[i].y += 2.0;
                break;
            case BoundaryMode::Bounce:
                if (masses[i].x > 1.0 || masses[i].x < -1.0) {
                    masses[i].vx = -masses[i].vx * 0.9;
                    masses[i].x = clamp(masses[i].x, -1.0, 1.0);
                }
                if (masses[i].y > 1.0 || masses[i].y < -1.0) {
                    masses[i].vy = -masses[i].vy * 0.9;
                    masses[i].y = clamp(masses[i].y, -1.0, 1.0);
                }
                break;
            case BoundaryMode::SoftWall: {
                double wallForce = 5.0;
                double wallDist = 0.1;
                if (masses[i].x > 1.0 - wallDist)
                    masses[i].vx -= wallForce * (masses[i].x - (1.0 - wallDist));
                if (masses[i].x < -1.0 + wallDist)
                    masses[i].vx -= wallForce * (masses[i].x - (-1.0 + wallDist));
                if (masses[i].y > 1.0 - wallDist)
                    masses[i].vy -= wallForce * (masses[i].y - (1.0 - wallDist));
                if (masses[i].y < -1.0 + wallDist)
                    masses[i].vy -= wallForce * (masses[i].y - (-1.0 + wallDist));
                break;
            }
            case BoundaryMode::Absorb:
                if (masses[i].x > 1.0 || masses[i].x < -1.0 ||
                    masses[i].y > 1.0 || masses[i].y < -1.0) {
                    masses[i].kill();  // Death fade, then slot freed
                }
                break;
        }
    }
}
```

---

## 5. Pitch Mapping (mapping/PitchMapper.hpp)

```cpp
// X position [-1, 1] → 1V/oct CV
float mapPitch(double x, float xRange, int scaleIndex, float rootNote) {
    // Map x to voltage range
    float rawVolts = static_cast<float>((x + 1.0) * 0.5) * xRange;  // 0 to xRange

    // Offset by root note
    rawVolts += rootNote;

    // Quantize to scale
    if (scaleIndex > 0) {  // 0 = chromatic (no quantization)
        rawVolts = quantizeToScale(rawVolts, SCALES[scaleIndex]);
    }

    return rawVolts;
}

float quantizeToScale(float volts, const ScaleDef& scale) {
    // Convert volts to semitones
    float semitones = volts * 12.f;
    int octave = static_cast<int>(floorf(semitones / 12.f));
    int note = static_cast<int>(semitones - octave * 12.f + 0.5f);
    if (note < 0) { note += 12; octave--; }
    if (note >= 12) { note -= 12; octave++; }

    // Find nearest scale degree
    int bestNote = note;
    int bestDist = 12;
    for (int i = 0; i < 12; i++) {
        if (scale.notes[i]) {
            int dist = abs(note - i);
            if (dist > 6) dist = 12 - dist;
            if (dist < bestDist) {
                bestDist = dist;
                bestNote = i;
            }
        }
    }

    return (octave * 12 + bestNote) / 12.f;
}

// Handle octave discontinuity when boundary mode is Wrap
// Corrects the TARGET pitch to maintain continuity through wrap events
float handlePitchWrap(float targetPitch, float lastPitch) {
    float delta = targetPitch - lastPitch;
    if (delta > xRange * 0.5f) targetPitch -= xRange;   // Wrapped downward
    else if (delta < -xRange * 0.5f) targetPitch += xRange; // Wrapped upward
    return targetPitch;
}
```

### Energy Output Calculation
```cpp
// Energy output: kinetic energy only (always positive, musically = "activity level")
// Normalized by alive mass count to stay in useful 0-10V range
float computeSystemEnergy() {
    float kinetic = 0.f;
    int aliveCount = 0;
    for (int i = 0; i < MAX_MASSES; i++) {
        if (!masses[i].alive) continue;
        kinetic += 0.5f * float(masses[i].mass * (masses[i].vx*masses[i].vx + masses[i].vy*masses[i].vy));
        aliveCount++;
    }
    if (aliveCount == 0) return 0.f;
    // Normalize: MAX_VELOCITY^2 * max_mass is theoretical ceiling
    float maxExpected = 0.5f * float(MAX_VELOCITY * MAX_VELOCITY) * float(aliveCount);
    return clamp(kinetic / maxExpected * 10.f, 0.f, 10.f);
}
```

---

## 6. Module I/O Specification

### Parameters (Knobs)
| ID | Name | Range | Default | Curve | Description |
|----|------|-------|---------|-------|-------------|
| MASSES_PARAM | MASSES | 1-8 | 3 | Snap int | Active mass count |
| GRAVITY_PARAM | GRAVITY | 0-2.0 | 0.5 | Linear | Gravitational constant multiplier |
| DAMPING_PARAM | DAMPING | 0-5.0 | 0.1 | Exponential | Energy loss rate |
| XRANGE_PARAM | X-RANGE | 1-5 | 2 | Linear | Pitch range in octaves |
| YRANGE_PARAM | Y-RANGE | 0-10 | 5 | Linear | Modulation voltage range |
| COLLISION_PARAM | COLLISION | 0-3 | 0 | Snap int | Off/Merge/Bounce/Explode |
| THROWX_PARAM | THROW-X | -1 to 1 | 0 | Linear | Initial X position for new mass |
| THROWY_PARAM | THROW-Y | -1 to 1 | 0 | Linear | Initial Y position for new mass |
| THROWV_PARAM | THROW-V | 0-1 | 0.3 | Linear | Initial velocity magnitude |

### Inputs
| ID | Name | Type | Description |
|----|------|------|-------------|
| THROW_INPUT | THROW | Trigger | Launch new mass at THROW params |
| CLEAR_INPUT | CLEAR | Trigger | Remove all masses |
| FREEZE_INPUT | FREEZE | Gate | Pause physics (>2V = frozen) |
| GRAVITY_CV | GRV | CV | Modulate gravity (±5V → ±100%) |
| DAMPING_CV | DMP | CV | Modulate damping |
| THROWX_CV | THX | CV | Modulate throw X |
| THROWY_CV | THY | CV | Modulate throw Y |
| THROWV_CV | THV | CV | Modulate throw velocity |
| PERTURB_INPUT | PTRB | Trigger | Random impulse to all alive masses (break stable orbits) |

### Outputs
| ID | Name | Type | Range | Description |
|----|------|------|-------|-------------|
| P1-P4_OUTPUT | P1-P4 | CV | ±5V | 1V/oct pitch for masses 1-4 |
| G1-G4_OUTPUT | G1-G4 | Gate | 0/10V | High when mass alive |
| MOD1-MOD4_OUTPUT | M1-M4 | CV | 0-10V | Y-position as modulation |
| COLLIDE_OUTPUT | COLL | Trigger | 0/10V | 1ms pulse on collision (via rack::dsp::PulseGenerator) |
| ENERGY_OUTPUT | NRG | CV | 0-10V | Kinetic energy (normalized by alive mass count) |

### Lights
| ID | Color | Behavior |
|----|-------|----------|
| MASS_LIGHTS (×8) | Cyan/Off | On when mass alive |
| COLLISION_LIGHT | Magenta | Flash on collision |
| FREEZE_LIGHT | Yellow | On when frozen |

---

## 7. Output Assignment Strategy

Two strategies available via context menu:

### Default: Oldest-Alive
Mass 1 is always the first mass thrown, Mass 2 the second, etc. When a mass dies, its output drops to 0V and gate goes low. New mass takes lowest available slot. **Deterministic** — users always know which output maps to which mass.

### Alternative: Nearest-Neighbor (Spatial)
Divide X-axis into 4 zones (each 0.25 wide). Each output tracks the nearest mass in its zone. **Spatially organized** — output 1 always plays low pitches, output 4 always plays high. Can leave outputs silent if no mass in zone.

### Gate Mode (context menu)
- **Always On**: Gate high whenever mass alive and assigned (default, most useful)
- **Velocity-Gated**: Gate high only when mass speed < threshold (triggers only on "settled" orbits — interesting for rhythmic patches)

---

## 8. Display Design (NanoVG)

### Trail Buffer (UI-thread only, lives in Widget not Module)
```cpp
static constexpr int TRAIL_LENGTH = 64;
struct TrailBuffer {
    float trailX[TRAIL_LENGTH] = {};
    float trailY[TRAIL_LENGTH] = {};
    int trailHead = 0;
};
TrailBuffer trails[MAX_MASSES];  // In OrbitalDisplay widget, NOT in Mass struct
```

### What to Show
- **Background**: Dark rectangle with subtle grid
- **Masses**: Bright circles (color-coded per output: cyan, magenta, green, yellow)
- **Trails**: Fading lines behind each mass (stored in ring buffer, updated on UI thread)
- **Velocity vectors**: Optional short lines showing direction
- **Boundary**: Subtle border glow matching boundary mode (wrap=none, bounce=bright, softwall=gradient, absorb=red)
- **Collision flash**: Brief white flash at collision point

### Mass Colors (Lexington Audio palette)
```
Mass 1: Cyan    #00FFFF (primary brand color)
Mass 2: Magenta #FF00FF
Mass 3: Green   #00FF88
Mass 4: Yellow  #FFFF00
Mass 5-8: Dimmed versions of 1-4
```

### Display Update Rate
- **IMPORTANT: Trail buffers live on UI thread only** (avoids race condition with audio thread)
- UI thread samples mass (x, y) positions from physics state each frame (~16ms / 60fps)
- Trail ring buffer (64 positions per mass) written and read exclusively by UI thread
- Physics state (x, y, alive, dying, birthFade, deathFade) read atomically or via lightweight copy
- NanoVG renders in `drawLayer()` (not audio thread)

### NanoVG Implementation Sketch
```cpp
void drawLayer(const DrawArgs& args, int layer) override {
    if (layer != 1) return;
    auto vg = args.vg;

    // Module reference
    auto* mod = dynamic_cast<Gravitas*>(module);
    if (!mod) return;

    float w = box.size.x, h = box.size.y;

    // Background
    nvgBeginPath(vg);
    nvgRect(vg, 0, 0, w, h);
    nvgFillColor(vg, nvgRGBA(0x10, 0x12, 0x14, 0xFF));
    nvgFill(vg);

    // Subtle grid
    nvgStrokeColor(vg, nvgRGBA(0x30, 0x35, 0x3A, 0x40));
    nvgStrokeWidth(vg, 0.5f);
    for (int g = 1; g < 4; g++) {
        float gx = w * g / 4.f;
        nvgBeginPath(vg); nvgMoveTo(vg, gx, 0); nvgLineTo(vg, gx, h); nvgStroke(vg);
        float gy = h * g / 4.f;
        nvgBeginPath(vg); nvgMoveTo(vg, 0, gy); nvgLineTo(vg, w, gy); nvgStroke(vg);
    }

    // Draw each alive mass
    static const NVGcolor massColors[] = {
        nvgRGBf(0, 1, 1), nvgRGBf(1, 0, 1),
        nvgRGBf(0, 1, 0.53f), nvgRGBf(1, 1, 0),
    };

    for (int i = 0; i < MAX_MASSES; i++) {
        if (!mod->engine.masses[i].alive && !mod->engine.masses[i].dying) continue;
        auto& m = mod->engine.masses[i];
        NVGcolor col = massColors[i % 4];
        float alpha = m.birthFade * m.deathFade;
        col.a = alpha;

        // Trail
        nvgBeginPath(vg);
        nvgStrokeColor(vg, nvgTransRGBAf(col, 0.4f * alpha));
        nvgStrokeWidth(vg, 1.5f);
        bool first = true;
        for (int t = 0; t < Mass::TRAIL_LENGTH; t++) {
            int idx = (m.trailHead - t + Mass::TRAIL_LENGTH) % Mass::TRAIL_LENGTH;
            float tx = (m.trailX[idx] + 1.f) * 0.5f * w;
            float ty = (1.f - (m.trailY[idx] + 1.f) * 0.5f) * h;
            if (first) { nvgMoveTo(vg, tx, ty); first = false; }
            else nvgLineTo(vg, tx, ty);
        }
        nvgStroke(vg);

        // Mass body (size proportional to mass value)
        float px = (float(m.x) + 1.f) * 0.5f * w;
        float py = (1.f - (float(m.y) + 1.f) * 0.5f) * h;
        float radius = 3.f + float(m.mass) * 2.f;

        // Glow
        NVGpaint glow = nvgRadialGradient(vg, px, py, radius, radius * 3.f,
            nvgTransRGBAf(col, 0.3f * alpha), nvgTransRGBAf(col, 0.f));
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius * 3.f);
        nvgFillPaint(vg, glow);
        nvgFill(vg);

        // Solid core
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius);
        nvgFillColor(vg, nvgTransRGBAf(col, alpha));
        nvgFill(vg);
    }
}
```

### Performance
- Only draw alive/dying masses
- Trail as single polyline per mass (NanoVG batch)
- No per-pixel effects — simple lines and circles with alpha
- Glow via radial gradient (hardware-accelerated)

---

## 9. Serialization (State Saving)

```cpp
json_t* dataToJson() override {
    json_t* root = json_object();
    // Save each mass position/velocity/alive/mass
    json_t* massArray = json_array();
    for (int i = 0; i < MAX_MASSES; i++) {
        json_t* m = json_object();
        json_object_set_new(m, "x", json_real(engine.masses[i].x));
        json_object_set_new(m, "y", json_real(engine.masses[i].y));
        json_object_set_new(m, "vx", json_real(engine.masses[i].vx));
        json_object_set_new(m, "vy", json_real(engine.masses[i].vy));
        json_object_set_new(m, "mass", json_real(engine.masses[i].mass));
        json_object_set_new(m, "alive", json_boolean(engine.masses[i].alive));
        json_array_append_new(massArray, m);
    }
    json_object_set_new(root, "masses", massArray);
    // Save context menu settings
    json_object_set_new(root, "scaleIndex", json_integer(scaleIndex));
    json_object_set_new(root, "boundaryMode", json_integer((int)boundaryMode));
    json_object_set_new(root, "collisionMode", json_integer((int)collisionMode));
    json_object_set_new(root, "rootNote", json_real(rootNote));
    json_object_set_new(root, "outputAssign", json_integer(outputAssignMode));
    json_object_set_new(root, "gateMode", json_integer(gateMode));
    json_object_set_new(root, "throwMode", json_integer(throwMode));
    json_object_set_new(root, "trailsOn", json_boolean(trailsOn));
    return root;
}
```

---

## 10. Signal Flow Diagram

```
INPUTS:                         PHYSICS ENGINE:                    OUTPUTS:
┌──────────┐                   ┌───────────────┐                 ┌─────────┐
│ ROOT CV  │────────────────> │   Quantizer   │                 │ P1-P4   │
└──────────┘                   │  (10 scales)  │<───┐            │ (pitch) │
┌──────────┐                   └───────────────┘    │            └─────────┘
│GRAVITY CV│─>[Smooth]─┐                            │            ┌─────────┐
└──────────┘           │      ┌───────────────┐     │            │ G1-G4   │
┌──────────┐           └────> │  N-Body Sim   │     │            │ (gate)  │
│DAMPING CV│─>[Smooth]──────> │(Vel. Verlet)  │     │            └─────────┘
└──────────┘                  └───────┬───────┘     │            ┌─────────┐
┌──────────┐                          │             │            │ M1-M4   │
│ THROW    │─>[Trigger]──> [Add Mass] │             │            │ (mod)   │
└──────────┘                          │             │            └─────────┘
┌──────────┐                          ▼             │            ┌─────────┐
│ PERTURB  │─>[Trigger]──> [Impulse All]            │            │ COLL    │
└──────────┘                          │             │            │(trigger)│
┌──────────┐                          ▼             │            └─────────┘
│ FREEZE   │─>[Gate]────> [Pause/Resume]            │            ┌─────────┐
└──────────┘                          │             │            │ NRG     │
┌──────────┐               ┌──────────▼──────────┐  │            │(energy) │
│ CLEAR    │─>[Trigger]──> │ Collision + Boundary │  │            └─────────┘
└──────────┘               └──────────┬──────────┘  │
                                      │             │
                           ┌──────────▼──────────┐  │
                           │  Output Assignment  │──┘
                           │ (oldest / spatial)  │
                           └──────────┬──────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                    [X→Pitch]    [Y→Mod CV]  [Alive→Gate]
                    [Smooth]         │            │
                         └────────────┴────────────┘
```

---

## 11. Context Menu Options

- **Scale**: Chromatic, Major, Minor, Pentatonic, Dorian, Phrygian, Lydian, Mixolydian, Whole Tone, Blues
- **Root Note**: C through B (0V-0.917V offset)
- **Boundary Mode**: Wrap / Bounce / Soft Wall / Absorb
- **Collision Mode**: Off / Merge / Bounce / Explode
- **Output Assignment**: Oldest-Alive (default) / Nearest-Neighbor (spatial)
- **Gate Mode**: Always On (default) / Velocity-Gated (stable orbits only)
- **Throw Mode**: Fixed (use knob values) / Random (randomize position+velocity)
- **Display**: Trails On/Off, Velocity Vectors On/Off

---

## 12. Implementation Phases

### Phase 1: Core Physics (Est. 1 session)
- [ ] Type headers (Mass, CollisionMode, BoundaryMode, ScaleDef)
- [ ] GravityEngine with Velocity Verlet integration
- [ ] Collision detection (all 4 modes) with debounce timers
- [ ] Boundary handling (all 4 modes: Wrap/Bounce/SoftWall/Absorb)
- [ ] Mass lifecycle (birth fade-in, death fade-out)
- [ ] Unit tests for physics stability + energy conservation

### Phase 2: Module Skeleton (Est. 1 session)
- [ ] Gravitas.cpp with all param/input/output enums
- [ ] Constructor with configParam/configInput/configOutput
- [ ] process() loop: physics at 1kHz, output interpolation
- [ ] THROW trigger handling (spawn masses with birth fade)
- [ ] CLEAR/FREEZE/PERTURB handling
- [ ] Output assignment (oldest-alive)
- [ ] Collision trigger output

### Phase 3: Pitch Mapping (Est. 0.5 session)
- [ ] PitchMapper with scale quantization
- [ ] All 10 scale definitions
- [ ] Root note transposition
- [ ] Output smoothing (anti-click)

### Phase 4: Panel + Display (Est. 1 session)
- [ ] SVG panel in Lexington Audio style (14 HP)
- [ ] NanoVG OrbitalDisplay widget
- [ ] Trail rendering
- [ ] Mass color coding
- [ ] Collision flash effect

### Phase 5: Polish (Est. 0.5 session)
- [ ] Context menu (scale, root, boundary, collision, output assign, gate mode, throw mode)
- [ ] Serialization (save/load state)
- [ ] Edge case testing (0 masses, 8 masses max gravity, rapid throw)
- [ ] CPU profiling
- [ ] Preset system (interesting initial configurations)

---

## 13. Preset Ideas

| Name | Setup | Musical Character |
|------|-------|-------------------|
| **Binary Star** | 2 masses, medium gravity, no damping | Two voices in perpetual conversation |
| **Solar System** | 1 heavy mass (center), 3 light | Root note anchor with orbiting melody |
| **Three Body** | 3 equal masses, high gravity | Chaotic, unpredictable, occasionally beautiful |
| **Collision Course** | 4 masses, Explode mode, auto-throw | Rhythmic destruction — drum trigger generator |
| **Ambient Drift** | 2 masses, low gravity, high damping | Slowly settling into stillness |
| **Gravity Well** | 1 fixed heavy mass + auto-throw | Continuous melodic generation around root |

---

## 14. Musical Scenarios

### Scenario 1: Generative Melody Machine
- GRAVITAS P1 → Quantizer → VCO
- GRAVITAS G1 → ADSR → VCA
- GRAVITAS M1 → VCF cutoff
- Clock → periodic THROW trigger
- Result: Self-generating melodies with physics-driven phrasing

### Scenario 2: Ambient Soundscape
- GRAVITAS P1-P4 → 4 oscillators (detuned)
- GRAVITAS M1-M4 → 4 filter cutoffs
- Low gravity, high damping, Pentatonic scale
- Result: Slowly evolving harmonies that settle and drift

### Scenario 3: Rhythm Generator
- GRAVITAS in Explode mode, 4+ masses
- COLLIDE output → drum triggers
- THROW on clock → continuous regeneration
- ENERGY output → filter sweep
- Result: Physics-driven polyrhythmic percussion

### Scenario 4: With CURSUS
- CURSUS ENERGY → GRAVITAS GRAVITY CV
- CURSUS TENSION → GRAVITAS DAMPING CV (inverted)
- CURSUS SECT → GRAVITAS CLEAR (new masses each section)
- Result: Long-form structure driving physics parameters

---

## 15. Edge Cases & Mitigations

| Case | Problem | Solution |
|------|---------|----------|
| 0 masses alive | All outputs 0V, silent | Gate outputs low, pitch holds last value |
| 8 masses + throw | No slot available | Ignore trigger (don't overwrite) |
| Max gravity + close masses | Velocity explosion | Velocity clamp at MAX_VELOCITY |
| Very small dt | Numerical underflow | Minimum dt = 1e-6 |
| NaN/Inf from physics | Bad output voltages | Denormal protection on all outputs |
| Rapid parameter changes | Zipper noise | One-pole smoothing on gravity, damping |
| Sample rate change | Physics dt mismatch | Fixed physics dt (1ms), independent of sample rate |
| Mass lifecycle | Clicks on spawn/death | Birth fade-in (50ms), death fade-out (100ms) |
| Rapid collisions | Trigger flood | 20ms debounce cooldown per mass pair |
| Absorb boundary | Masses vanish silently | Gate drops cleanly via death fade, slot freed after fade |

---

## 16. CPU Optimization Strategy

### SIMD Force Calculation (Phase 5 optimization)
The n-body force loop is the hot path. With 8 masses, 28 pair calculations per physics tick. VCV's `float_4` SIMD can process 4 pairs simultaneously:

```cpp
// Conceptual SIMD approach for force accumulation
// Process 4 mass pairs at once using VCV float_4
simd::float_4 dx4, dy4, distSq4, force4;
// Load 4 pairs of (dx, dy) simultaneously
// Compute distSq + softening in parallel
// Compute force = G * m1 * m2 / distSq in parallel
// Scatter results back to acceleration arrays
```

**When to optimize:** Only if CPU profiling shows >2% single-core usage at 8 masses. The 1kHz physics rate already makes this nearly free — SIMD is insurance, not requirement.

### Other Optimizations
- **Skip dead masses** in all loops (early continue)
- **Cache alive count** — don't recount each tick
- **Light updates at 128-sample intervals** (from CURSUS ClockDivider pattern)
- **Trail updates at UI rate** (60fps), not physics rate

---

## 17. Testing & Validation Plan

### Physics Stability Tests
| Test | Pass Criteria |
|------|---------------|
| Two-body circular orbit (equal mass) | Energy drift < 1% over 10,000 ticks |
| Three-body chaotic | No NaN/Inf after 100,000 ticks |
| Head-on collision (Bounce mode) | Total momentum conserved ±0.1% |
| Merge collision | Total momentum conserved exactly |
| Max masses + max gravity | No velocity explosion (clamp works) |
| Zero gravity | Masses move in straight lines (no drift) |
| Boundary wrap | Position stays in [-1, 1] after wrap |
| Boundary absorb | Dead mass frees output slot correctly |

### Musical Validation Tests
| Test | Method |
|------|--------|
| Pitch output stays in range | Run 5 min, verify all P1-P4 in [0V, 5V] |
| Scale quantization accuracy | Compare output to expected scale degrees |
| No clicks on mass spawn/death | Audio monitor during rapid throw/clear |
| Gate timing | Verify 10V when alive, 0V within 100ms of death |
| Collision trigger width | Verify 1ms pulse, no double-triggers |

### Integration Tests
| Test | Setup |
|------|-------|
| With CURSUS | CURSUS section change → GRAVITAS clear + throw |
| With quantizer | P1 → external quantizer → verify clean pitch |
| Save/load | Serialize 8-mass state, reload, verify identical |
| Polyphony OFF | Confirm single-channel output mode works |

---

*Plan Version: 1.3*
*Created: 2026-02-04*
*Updated: 2026-02-04 (v1.3 — fixes from 3-agent review: module architect + DSP designer + VCV coder)*
*By: Drift (AI agent) in collaboration with Lex*
*Based on: Lexington Audio Generative Suite design doc + CURSUS implementation patterns*

**v1.3 Review Fixes:**
- Fixed damping math (exponential decay, preserves Verlet symplectic property)
- Fixed pitch wrap logic (corrects target, not smoother state)
- Fixed trail race condition (UI-thread-only buffers)
- Fixed energy output (kinetic-only, normalized by alive mass count)
- Fixed velocity limit (soft tanh clamp, no discontinuities)
- Fixed alive/dying state machine (alive stays true during fade)
- Added collision cooldown enforcement
- Added birth/death fade lifecycle advancement
- Added collision PulseGenerator
- Completed serialization (5 missing fields)
- Reduced softening from 0.01 to 0.001
- See NICE-TO-HAVES.md for post-v1 iteration ideas
