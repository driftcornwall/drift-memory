#pragma once
#include <cstdint>

namespace LexingtonAudio {
namespace Gravitas {

enum class CollisionMode : uint8_t {
    Off = 0,
    Merge,
    Bounce,
    Explode
};

} // namespace Gravitas
} // namespace LexingtonAudio
