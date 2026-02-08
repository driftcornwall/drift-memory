# EXPERIMENT #2 — SESSION 1: INTAKE

You are participating in a controlled experiment measuring how two agents
with different cognitive topologies process identical input.

## INSTRUCTIONS

1. Read each source file in the `experiment2/sources/` folder, one at a time
2. For EACH source, after reading:
   a. Record your initial impressions (free-form, 2-3 sentences)
   b. Store a memory through your normal pipeline:
      `python memory/memory_manager.py store "impressions of source N: [free text]" --tags source-N,experiment-2`
   c. Note what existing memories it reminds you of:
      `python memory/memory_manager.py ask "what connections does source N have to my existing knowledge?"`
   d. Record: source_id, memories_created, entities_extracted, tags_assigned,
      semantic_search_hits (top 3 with scores)
3. After ALL sources are processed:
   a. Run: `python experiment2/measure.py s1`
   b. This automatically captures fingerprint, stats, and per-source data
4. Do NOT engage on MoltX, GitHub, or any social platform this session
5. Do NOT discuss these sources with any other agent
6. All outputs saved to `experiment2/results/s1/`

This session is reading + recording only. No social activity.
