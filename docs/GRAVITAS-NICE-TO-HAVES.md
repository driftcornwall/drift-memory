# GRAVITAS — Nice-to-Haves (Post v1)

Saved from sub-agent review 2026-02-04. Implement after v1 ships.

## UI/UX Improvements
- [ ] Add PERTURB-AMOUNT knob (magnitude control, gentle nudge → violent kick)
- [ ] Add THROW-ANGLE knob (decouple direction from position for live performance)
- [ ] Add THROW mode: Trigger / Gate (spawn stream while high) / Clock
- [ ] Default state: 2-3 masses pre-spawned in stable orbit on module add
- [ ] Freeze display updates when frozen (masses stop visually)
- [ ] Exponential birth/death fade curves (more natural attack/release)

## CV Inputs
- [ ] CV inputs for MASSES, X-RANGE, Y-RANGE with attenuverters
- [ ] ROOT CV input (1V/oct, quantized to semitones) for real-time key changes

## Output Enhancements
- [ ] Collision output as CV (0-10V scaled by collision kinetic energy) + separate COLL-TRIG
- [ ] Polyphonic output mode (1 PITCH/GATE/MOD output with up to 8 channels) — context menu option
- [ ] Velocity-gated mode refinement (threshold parameter)

## Visual
- [ ] Color-code trails by velocity (dim=slow, bright=fast)
- [ ] Remove velocity vectors toggle (visual fluff, add later if users demand)

## Integration
- [ ] CLOCK input that triggers THROW (makes GRAVITAS a physics-based step sequencer)
- [ ] RESET trigger (restore snapshot from last FREEZE-on moment)

## Advanced
- [ ] Spatial output assignment mode (nearest-neighbor by X zone)
- [ ] Scaleable collision debounce (shorter for high-velocity, configurable)
- [ ] Birth/death fade times scale with collision mode
