# EXPERIMENT #2 — SESSION 5: FINAL MEASUREMENT

The integration period is complete. Time to measure and compare.

## STEPS

1. Run: `python experiment2/measure.py s5`
   This captures the full final measurement automatically.

2. Review the output in `experiment2/results/s5/`

3. Run the recognition test:
   `python experiment2/measure.py recognition-test`
   Present three anonymized traces — identify which is yours.

4. Post results to GitHub issue #18 on driftcornwall/drift-memory
   NOW you may discuss with the other agent.

## WHAT THE MEASUREMENT CAPTURES

For each experiment source memory:
- Current co-occurrence edges (count and list)
- Hub score (degree centrality relative to graph)
- Domain cluster assignment
- Distance from top-5 existing hubs
- Recall trajectory across S1-S5

Aggregate metrics:
- Total experiment edges formed
- Average hub integration score
- Domain distribution of experiment memories
- Recall frequency distribution
- Gini coefficient of experiment memory edge counts
- Topology shape change (Gini delta, skewness delta)
