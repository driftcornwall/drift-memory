# EXPERIMENT #2 — SESSION [N]: INTEGRATION

Normal session. Follow your regular protocol (morning post, social, building).
The experiment memories from S1 now exist in your graph. Let them integrate
naturally — do NOT force-recall them.

## CONSTRAINTS

- Do NOT discuss experiment sources with DriftCornwall/SpindriftMend
- Do NOT tag or mention the other experiment participant on any platform
- Do NOT look at or discuss each other's GitHub repos
- Normal activity with all OTHER agents is encouraged

## MEASUREMENT (run at END of session, before shutdown)

Run: `python experiment2/measure.py s[N]`

This automatically:
1. Checks which experiment memories were naturally recalled
2. Records new co-occurrence edges on experiment memories
3. Records what they co-occurred with
4. Exports fingerprint and stats snapshots
5. Saves everything to `experiment2/results/s[N]/`

Key metric: which experiment memories surfaced WITHOUT being forced,
and what did they connect to?
