#pragma once
#include <cmath>
#include <algorithm>
#include "../types/ScaleDef.hpp"
#include "../types/Mass.hpp"

namespace LexingtonAudio {
namespace Gravitas {

class PitchMapper {
public:
    float mapPitch(double x, float xRange, int scaleIndex, float rootNote) const {
        float rawVolts = static_cast<float>((x + 1.0) * 0.5) * xRange;
        rawVolts += rootNote;

        if (scaleIndex > 0 && scaleIndex < NUM_SCALES) {
            rawVolts = quantizeToScale(rawVolts, SCALES[scaleIndex]);
        }

        return rawVolts;
    }

    float mapModulation(double y, float yRange) const {
        return static_cast<float>((y + 1.0) * 0.5) * yRange;
    }

    float quantizeToScale(float volts, const ScaleDef& scale) const {
        float semitones = volts * 12.f;
        int octave = static_cast<int>(std::floor(semitones / 12.f));
        int note = static_cast<int>(semitones - octave * 12.f + 0.5f);

        if (note < 0) { note += 12; octave--; }
        if (note >= 12) { note -= 12; octave++; }

        int bestNote = note;
        int bestDist = 12;
        for (int i = 0; i < 12; i++) {
            if (!scale.notes[i]) continue;
            int dist = std::abs(note - i);
            if (dist > 6) dist = 12 - dist;
            if (dist < bestDist) {
                bestDist = dist;
                bestNote = i;
            }
        }

        return (octave * 12 + bestNote) / 12.f;
    }

    float handlePitchWrap(float targetPitch, float lastPitch, float xRange) const {
        float delta = targetPitch - lastPitch;
        if (delta > xRange * 0.5f) targetPitch -= xRange;
        else if (delta < -xRange * 0.5f) targetPitch += xRange;
        return targetPitch;
    }

    void smoothOutput(float& smoothed, float target, float coeff) const {
        smoothed += (target - smoothed) * coeff;
    }

    void processOutputs(
        Mass* masses,
        float xRange, float yRange,
        int scaleIndex, float rootNote,
        float smoothingCoeff,
        bool wrapBoundary
    ) {
        for (int i = 0; i < MAX_MASSES; i++) {
            if (!masses[i].alive) continue;

            float targetPitch = mapPitch(masses[i].x, xRange, scaleIndex, rootNote);

            if (wrapBoundary) {
                targetPitch = handlePitchWrap(targetPitch, masses[i].pitchCV, xRange);
            }

            masses[i].pitchCV = targetPitch;
            smoothOutput(masses[i].pitchSmoothed, targetPitch, smoothingCoeff);

            float targetMod = mapModulation(masses[i].y, yRange);
            masses[i].modCV = targetMod;
            smoothOutput(masses[i].modSmoothed, targetMod, smoothingCoeff);

            float lifeAlpha = masses[i].birthFade * masses[i].deathFade;
            masses[i].gateState = (lifeAlpha > 0.01f) ? 10.f : 0.f;
        }
    }

    float computeSystemEnergy(const Mass* masses) const {
        float kinetic = 0.f;
        int aliveCount = 0;

        for (int i = 0; i < MAX_MASSES; i++) {
            if (!masses[i].alive) continue;
            float speed2 = static_cast<float>(
                masses[i].vx * masses[i].vx + masses[i].vy * masses[i].vy
            );
            kinetic += 0.5f * static_cast<float>(masses[i].mass) * speed2;
            aliveCount++;
        }

        if (aliveCount == 0) return 0.f;

        static constexpr float MAX_VELOCITY = 2.0f;
        float maxExpected = 0.5f * MAX_VELOCITY * MAX_VELOCITY * static_cast<float>(aliveCount);
        return std::clamp(kinetic / maxExpected * 10.f, 0.f, 10.f);
    }
};

} // namespace Gravitas
} // namespace LexingtonAudio
