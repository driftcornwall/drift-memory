#pragma once
#include <cstdint>

namespace LexingtonAudio {
namespace Gravitas {

enum class BoundaryMode : uint8_t {
    Wrap = 0,
    Bounce,
    SoftWall,
    Absorb
};

} // namespace Gravitas
} // namespace LexingtonAudio
