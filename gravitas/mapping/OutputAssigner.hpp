#pragma once
#include <cstdint>
#include "../types/Mass.hpp"

namespace LexingtonAudio {
namespace Gravitas {

static constexpr int NUM_OUTPUTS = 4;

enum class AssignMode : uint8_t {
    OldestAlive = 0,
    NearestNeighbor
};

enum class GateMode : uint8_t {
    AlwaysOn = 0,
    VelocityGated
};

struct OutputSlot {
    int massIndex = -1;
    bool active = false;
};

class OutputAssigner {
public:
    AssignMode assignMode = AssignMode::OldestAlive;
    GateMode gateMode = GateMode::AlwaysOn;
    float velocityThreshold = 0.3f;

    OutputSlot slots[NUM_OUTPUTS];

    void assign(const Mass* masses) {
        switch (assignMode) {
            case AssignMode::OldestAlive:
                assignOldest(masses);
                break;
            case AssignMode::NearestNeighbor:
                assignSpatial(masses);
                break;
        }
    }

    float gateVoltage(const Mass& mass) const {
        if (!mass.alive) return 0.f;
        float lifeAlpha = mass.birthFade * mass.deathFade;
        if (lifeAlpha < 0.01f) return 0.f;

        if (gateMode == GateMode::VelocityGated) {
            float speed = static_cast<float>(
                std::sqrt(mass.vx * mass.vx + mass.vy * mass.vy)
            );
            if (speed > velocityThreshold) return 0.f;
        }

        return 10.f;
    }

private:
    void assignOldest(const Mass* masses) {
        int slot = 0;
        for (int i = 0; i < MAX_MASSES && slot < NUM_OUTPUTS; i++) {
            if (masses[i].alive || masses[i].dying) {
                slots[slot].massIndex = i;
                slots[slot].active = true;
                slot++;
            }
        }
        for (; slot < NUM_OUTPUTS; slot++) {
            slots[slot].massIndex = -1;
            slots[slot].active = false;
        }
    }

    void assignSpatial(const Mass* masses) {
        for (int s = 0; s < NUM_OUTPUTS; s++) {
            slots[s].massIndex = -1;
            slots[s].active = false;

            float zoneLow = -1.f + s * 0.5f;
            float zoneHigh = zoneLow + 0.5f;
            float bestDist = 999.f;

            for (int i = 0; i < MAX_MASSES; i++) {
                if (!masses[i].alive && !masses[i].dying) continue;
                float mx = static_cast<float>(masses[i].x);
                if (mx < zoneLow || mx > zoneHigh) continue;

                float zoneCenter = (zoneLow + zoneHigh) * 0.5f;
                float dist = std::abs(mx - zoneCenter);
                if (dist < bestDist) {
                    bestDist = dist;
                    slots[s].massIndex = i;
                    slots[s].active = true;
                }
            }
        }
    }
};

} // namespace Gravitas
} // namespace LexingtonAudio
