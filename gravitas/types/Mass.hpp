#pragma once

namespace LexingtonAudio {
namespace Gravitas {

struct Mass {
    // Physics state (double precision for stability)
    double x = 0.0, y = 0.0;
    double vx = 0.0, vy = 0.0;
    double mass = 1.0;
    bool alive = false;

    // Output state (float for VCV)
    float pitchCV = 0.f;
    float modCV = 0.f;
    float gateState = 0.f;

    // Smoothing for output
    float pitchSmoothed = 0.f;
    float modSmoothed = 0.f;

    // Lifecycle (smooth birth/death transitions)
    float birthFade = 0.f;
    float deathFade = 1.f;
    bool dying = false;
    static constexpr float BIRTH_TIME = 0.05f;
    static constexpr float DEATH_TIME = 0.1f;

    // Collision debounce
    float collisionCooldown = 0.f;
    static constexpr float COLLISION_COOLDOWN_TIME = 0.02f;

    void kill() {
        dying = true;
    }

    void hardKill() {
        alive = false;
        dying = false;
        vx = vy = 0.0;
        deathFade = 1.f;
        birthFade = 0.f;
    }

    void updateLifecycle(float dt) {
        if (alive && !dying && birthFade < 1.f) {
            birthFade += dt / BIRTH_TIME;
            if (birthFade > 1.f) birthFade = 1.f;
        }
        if (dying) {
            deathFade -= dt / DEATH_TIME;
            if (deathFade <= 0.f) {
                deathFade = 0.f;
                alive = false;
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
