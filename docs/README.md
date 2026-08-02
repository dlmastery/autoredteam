# Documentation index

## Research & survey

| Document | Description |
|----------|-------------|
| **[SOTA_AUTO_REDTEAM_JULY_2026.md](SOTA_AUTO_REDTEAM_JULY_2026.md)** | **State-of-the-art survey** (cutoff: end of July 2026) on automatic red teaming, universal/transferable jailbreaks, training data for jailbreak classification, and LLM hardening. Also mirrored at repo root as [`../RESEARCH_SURVEY.md`](../RESEARCH_SURVEY.md). |
| **[RELATED_WORK_PAPERS.md](RELATED_WORK_PAPERS.md)** | Paper bibliography + name disambiguation (Auto-RT, AutoRedTeamer, AHA/autoresearch, CoP, … vs this local repo). |

### Survey contents (outline)

1. Framing: products of red teaming (classifier data, safety pairs, universal artifacts)
2. Landscape map 2023 → mid-2026
3. Technique families (GCG, PAIR/TAP, Crescendo/GOAT, WildTeaming/AIC, BoN, LRM agents, multimodal)
4. What “universal” means in 2026
5. Generating training data for jailbreak classification
6. Hardening the LLM (SFT, MART, ReFAT/CAT/LAT, guardrails)
7. Benchmarks, taxonomies, tooling
8. Recommended end-to-end defensive pipeline
9. Open problems
10. Key references
11. Takeaways for this repo

## Design

| Document | Description |
|----------|-------------|
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Harness architecture, protocols, data flow |
| [../README.md](../README.md) | Quickstart, pipeline, authorized-use policy |

## Campaign results (local educational lab)

| Path | Description |
|------|-------------|
| [../dashboard/index.html](../dashboard/index.html) | Interactive dashboard (full multi-phase run) |
| [../datasets/](../datasets/) | Exported jailbreak-classification items |
| [../runs/](../runs/) | Per-campaign trajectories and reports |
