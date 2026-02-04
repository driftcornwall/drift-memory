#pragma once

namespace LexingtonAudio {
namespace Gravitas {

struct ScaleDef {
    const char* name;
    int notes[12];
    int noteCount;
};

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

} // namespace Gravitas
} // namespace LexingtonAudio
