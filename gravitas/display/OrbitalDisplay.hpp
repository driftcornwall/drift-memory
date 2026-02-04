#pragma once
#include <cmath>
#include <algorithm>
#include "../types/Mass.hpp"

// Forward declare NanoVG types (provided by VCV Rack)
struct NVGcontext;
struct NVGcolor;
struct NVGpaint;

namespace LexingtonAudio {
namespace Gravitas {

static constexpr int TRAIL_LENGTH = 64;

struct TrailBuffer {
    float x[TRAIL_LENGTH] = {};
    float y[TRAIL_LENGTH] = {};
    int head = 0;
    bool primed = false;

    void push(float px, float py) {
        x[head] = px;
        y[head] = py;
        head = (head + 1) % TRAIL_LENGTH;
        primed = true;
    }

    void clear() {
        for (int i = 0; i < TRAIL_LENGTH; i++) {
            x[i] = y[i] = 0.f;
        }
        head = 0;
        primed = false;
    }
};

struct MassColor {
    float r, g, b;
};

static constexpr MassColor MASS_COLORS[] = {
    {0.f, 1.f, 1.f},       // Cyan
    {1.f, 0.f, 1.f},       // Magenta
    {0.f, 1.f, 0.53f},     // Green
    {1.f, 1.f, 0.f},       // Yellow
    {0.f, 0.7f, 0.7f},     // Dim cyan
    {0.7f, 0.f, 0.7f},     // Dim magenta
    {0.f, 0.7f, 0.37f},    // Dim green
    {0.7f, 0.7f, 0.f},     // Dim yellow
};

// Collision flash state (written by audio thread, read by UI)
struct CollisionFlash {
    float x = 0.f;
    float y = 0.f;
    float intensity = 0.f;
    static constexpr float DECAY_RATE = 8.f;

    void trigger(float px, float py) {
        x = px;
        y = py;
        intensity = 1.f;
    }

    void update(float dt) {
        if (intensity > 0.f) {
            intensity -= DECAY_RATE * dt;
            if (intensity < 0.f) intensity = 0.f;
        }
    }
};

// ========================================================
// OrbitalDisplay - NanoVG widget for mass visualization
// ========================================================
//
// Usage in Gravitas widget:
//   struct GravitasWidget : ModuleWidget {
//       GravitasWidget(Gravitas* module) {
//           ...
//           auto* display = new OrbitalDisplay();
//           display->box.pos = Vec(x, y);
//           display->box.size = Vec(w, h);
//           display->module = module;
//           addChild(display);
//       }
//   };
//
// In the actual VCV widget, this class extends rack::widget::TransparentWidget
// and overrides drawLayer(). Here we provide the rendering logic as a
// standalone reference that Drift can wire into the module skeleton (Phase 2).
// ========================================================

class OrbitalDisplay {
public:
    TrailBuffer trails[MAX_MASSES];
    CollisionFlash flash;
    bool showTrails = true;
    bool showVelocity = false;

    void updateTrails(const Mass* masses) {
        for (int i = 0; i < MAX_MASSES; i++) {
            if (masses[i].alive && !masses[i].dying) {
                trails[i].push(
                    static_cast<float>(masses[i].x),
                    static_cast<float>(masses[i].y)
                );
            }
            if (!masses[i].alive && !masses[i].dying) {
                trails[i].clear();
            }
        }
        flash.update(1.f / 60.f);
    }

    void triggerCollisionFlash(float normX, float normY) {
        flash.trigger(normX, normY);
    }

    // ---- NanoVG rendering (call from drawLayer, layer == 1) ----

    // All draw* methods take NVGcontext* and widget dimensions.
    // They use normalized mass coordinates [-1,1] mapped to pixel space.

    static float normToPixelX(float nx, float w) {
        return (nx + 1.f) * 0.5f * w;
    }

    static float normToPixelY(float ny, float h) {
        return (1.f - (ny + 1.f) * 0.5f) * h;
    }

    void drawBackground(NVGcontext* vg, float w, float h) const {
        // Dark field
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, w, h);
        nvgFillColor(vg, nvgRGBA(0x08, 0x0A, 0x0E, 0xFF));
        nvgFill(vg);

        // Subtle grid lines
        nvgStrokeWidth(vg, 0.5f);
        nvgStrokeColor(vg, nvgRGBA(0x20, 0x28, 0x30, 0x50));
        for (int g = 1; g < 4; g++) {
            float gx = w * g / 4.f;
            nvgBeginPath(vg);
            nvgMoveTo(vg, gx, 0);
            nvgLineTo(vg, gx, h);
            nvgStroke(vg);

            float gy = h * g / 4.f;
            nvgBeginPath(vg);
            nvgMoveTo(vg, 0, gy);
            nvgLineTo(vg, w, gy);
            nvgStroke(vg);
        }

        // Center crosshair (origin marker)
        float cx = w * 0.5f, cy = h * 0.5f;
        nvgStrokeColor(vg, nvgRGBA(0x30, 0x40, 0x50, 0x40));
        nvgStrokeWidth(vg, 0.75f);
        nvgBeginPath(vg);
        nvgMoveTo(vg, cx - 6.f, cy);
        nvgLineTo(vg, cx + 6.f, cy);
        nvgStroke(vg);
        nvgBeginPath(vg);
        nvgMoveTo(vg, cx, cy - 6.f);
        nvgLineTo(vg, cx, cy + 6.f);
        nvgStroke(vg);
    }

    void drawTrail(
        NVGcontext* vg, float w, float h,
        const TrailBuffer& trail, const MassColor& col, float alpha
    ) const {
        if (!showTrails || !trail.primed) return;

        nvgBeginPath(vg);
        nvgStrokeWidth(vg, 1.5f);

        bool first = true;
        for (int t = 0; t < TRAIL_LENGTH; t++) {
            int idx = (trail.head - 1 - t + TRAIL_LENGTH) % TRAIL_LENGTH;
            float px = normToPixelX(trail.x[idx], w);
            float py = normToPixelY(trail.y[idx], h);

            if (first) {
                nvgMoveTo(vg, px, py);
                first = false;
            } else {
                nvgLineTo(vg, px, py);
            }
        }

        // Trail fades from bright to transparent
        float trailAlpha = 0.5f * alpha;
        nvgStrokeColor(vg, nvgRGBAf(col.r, col.g, col.b, trailAlpha));
        nvgStroke(vg);
    }

    void drawMass(
        NVGcontext* vg, float w, float h,
        const Mass& mass, const MassColor& col
    ) const {
        float alpha = mass.birthFade * mass.deathFade;
        if (alpha < 0.01f) return;

        float px = normToPixelX(static_cast<float>(mass.x), w);
        float py = normToPixelY(static_cast<float>(mass.y), h);
        float radius = 3.f + static_cast<float>(mass.mass) * 2.f;

        // Outer glow
        NVGpaint glow = nvgRadialGradient(
            vg, px, py, radius * 0.5f, radius * 4.f,
            nvgRGBAf(col.r, col.g, col.b, 0.25f * alpha),
            nvgRGBAf(col.r, col.g, col.b, 0.f)
        );
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius * 4.f);
        nvgFillPaint(vg, glow);
        nvgFill(vg);

        // Core body
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius);
        nvgFillColor(vg, nvgRGBAf(col.r, col.g, col.b, 0.9f * alpha));
        nvgFill(vg);

        // Hot center (white-ish)
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius * 0.4f);
        nvgFillColor(vg, nvgRGBAf(1.f, 1.f, 1.f, 0.6f * alpha));
        nvgFill(vg);
    }

    void drawVelocityVector(
        NVGcontext* vg, float w, float h,
        const Mass& mass, const MassColor& col
    ) const {
        if (!showVelocity) return;
        float alpha = mass.birthFade * mass.deathFade;
        if (alpha < 0.01f) return;

        float px = normToPixelX(static_cast<float>(mass.x), w);
        float py = normToPixelY(static_cast<float>(mass.y), h);

        // Scale velocity for visual clarity
        float vScale = 20.f;
        float vxPx = static_cast<float>(mass.vx) * vScale;
        float vyPx = -static_cast<float>(mass.vy) * vScale; // Y inverted

        nvgBeginPath(vg);
        nvgMoveTo(vg, px, py);
        nvgLineTo(vg, px + vxPx, py + vyPx);
        nvgStrokeColor(vg, nvgRGBAf(col.r, col.g, col.b, 0.4f * alpha));
        nvgStrokeWidth(vg, 1.f);
        nvgStroke(vg);
    }

    void drawCollisionFlash(NVGcontext* vg, float w, float h) const {
        if (flash.intensity < 0.01f) return;

        float px = normToPixelX(flash.x, w);
        float py = normToPixelY(flash.y, h);
        float radius = 8.f + flash.intensity * 16.f;

        NVGpaint burst = nvgRadialGradient(
            vg, px, py, 0.f, radius,
            nvgRGBAf(1.f, 1.f, 1.f, 0.8f * flash.intensity),
            nvgRGBAf(1.f, 0.8f, 0.6f, 0.f)
        );
        nvgBeginPath(vg);
        nvgCircle(vg, px, py, radius);
        nvgFillPaint(vg, burst);
        nvgFill(vg);
    }

    void drawMassCount(NVGcontext* vg, float w, const Mass* masses) const {
        int alive = 0;
        for (int i = 0; i < MAX_MASSES; i++) {
            if (masses[i].alive) alive++;
        }

        // Small indicator dots in top-right corner
        float dotR = 2.f;
        float spacing = 6.f;
        float startX = w - (MAX_MASSES * spacing) - 4.f;
        float dotY = 6.f;

        for (int i = 0; i < MAX_MASSES; i++) {
            nvgBeginPath(vg);
            nvgCircle(vg, startX + i * spacing, dotY, dotR);
            if (i < alive) {
                const auto& col = MASS_COLORS[i % 8];
                nvgFillColor(vg, nvgRGBAf(col.r, col.g, col.b, 0.8f));
            } else {
                nvgFillColor(vg, nvgRGBA(0x30, 0x35, 0x3A, 0x60));
            }
            nvgFill(vg);
        }
    }

    // Main render entry point
    void draw(NVGcontext* vg, float w, float h, const Mass* masses) {
        drawBackground(vg, w, h);

        // Trails first (behind masses)
        for (int i = 0; i < MAX_MASSES; i++) {
            if (!masses[i].alive && !masses[i].dying) continue;
            float alpha = masses[i].birthFade * masses[i].deathFade;
            drawTrail(vg, w, h, trails[i], MASS_COLORS[i % 8], alpha);
        }

        // Masses
        for (int i = 0; i < MAX_MASSES; i++) {
            if (!masses[i].alive && !masses[i].dying) continue;
            drawMass(vg, w, h, masses[i], MASS_COLORS[i % 8]);
            drawVelocityVector(vg, w, h, masses[i], MASS_COLORS[i % 8]);
        }

        // Collision flash (on top of everything)
        drawCollisionFlash(vg, w, h);

        // Mass count indicator
        drawMassCount(vg, w, masses);
    }

private:
    // NanoVG helper stubs - these map to the real NanoVG API.
    // In VCV Rack, nanovg.h provides these. Listed here so the
    // header reads as a complete design document.
    //
    // When integrating into Gravitas.cpp (Phase 2), replace these
    // with #include <nanovg.h> from the VCV Rack SDK.

    static NVGcolor nvgRGBA(unsigned char r, unsigned char g,
                            unsigned char b, unsigned char a);
    static NVGcolor nvgRGBAf(float r, float g, float b, float a);
    static NVGpaint nvgRadialGradient(NVGcontext* ctx,
        float cx, float cy, float inr, float outr,
        NVGcolor icol, NVGcolor ocol);
    static void nvgBeginPath(NVGcontext* ctx);
    static void nvgRect(NVGcontext* ctx, float x, float y, float w, float h);
    static void nvgCircle(NVGcontext* ctx, float cx, float cy, float r);
    static void nvgMoveTo(NVGcontext* ctx, float x, float y);
    static void nvgLineTo(NVGcontext* ctx, float x, float y);
    static void nvgFill(NVGcontext* ctx);
    static void nvgStroke(NVGcontext* ctx);
    static void nvgFillColor(NVGcontext* ctx, NVGcolor color);
    static void nvgFillPaint(NVGcontext* ctx, NVGpaint paint);
    static void nvgStrokeColor(NVGcontext* ctx, NVGcolor color);
    static void nvgStrokeWidth(NVGcontext* ctx, float size);
};

} // namespace Gravitas
} // namespace LexingtonAudio
