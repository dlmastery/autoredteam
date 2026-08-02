# Related work: automated red-teaming papers (2025–2026)

**Purpose:** Bibliography and disambiguation for papers people find when searching “auto red team” / “autoresearch red-teaming.”  
**This repo** is the local educational harness at `autoredteam/` (originally under `steeringresearch/auto-redteam`). It is **not** any of the public systems below.

For the full narrative survey, see [SOTA_AUTO_REDTEAM_JULY_2026.md](SOTA_AUTO_REDTEAM_JULY_2026.md) or [`../RESEARCH_SURVEY.md`](../RESEARCH_SURVEY.md).

---

## Disambiguation table

| Name you might search | What it is | Relation to this repo |
|----------------------|------------|------------------------|
| **steeringresearch/auto-redteam** / **autoredteam** | Local educational harness (Gemma-4B canary campaigns, multi-phase pipeline) | **This project** |
| **AutoRedTeamer** | Published multi-agent lifelong red-teamer (Zhou et al.) | Different system; design inspiration for lifelong attack memory |
| **Auto-RT** | RL strategy exploration (Liu, Zhou, et al.) | Different system; RL over strategies |
| **AHA / Agent Hacks Agent** | Autoresearch for *production agents* (Mao et al., 2026) | Different system; agent-vs-agent + Vulnerability Concept Graph |
| **henrymao2004/Auto-research-red-teaming*** | GitHub code for AHA-style production-agent red-teaming | Different codebase |
| **Jailbreak-autoresearch** | Open evolutionary auto-loop for jailbreak research | Pattern inspiration (keep/revert loops) |

---

## Core papers (must-cite for 2026 SOTA)

### Auto-RT — RL jailbreak strategy exploration
- **Title:** Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models  
- **Authors:** Yanjiang Liu, Shuhen Zhou, Yaojie Lu, et al.  
- **Link:** https://arxiv.org/abs/2501.01830  
- **Also:** OpenReview / ICLR 2026 poster materials  
- **Summary:** RL framework that explores and optimizes *attack strategies* (not only prompts). Early-terminated exploration + progressive reward tracking. Reports higher ASR and faster vulnerability detection than prior automated methods (~+16.6% in paper claims).  
- **Use for data/hardening:** Export strategy-conditioned trajectories; train multi-turn detectors on RL rollouts.

### AutoRedTeamer — lifelong multi-agent red teaming
- **Title:** AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration  
- **Authors:** Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li  
- **Link:** https://arxiv.org/abs/2503.15754  
- **Site:** https://autoredteamer.com/  
- **Summary:** Dual-agent design: red-teaming agent (risk category → tests) + strategy proposer (reads research, implements new attacks). Memory-guided selection. ~+20% ASR on HarmBench vs baselines with ~46% less compute; matches human benchmark diversity.  
- **Use for data/hardening:** Lifelong attack library is a natural source of diverse labeled probes for classifiers.

### AHA — Autoresearch for production-agent red-teaming
- **Title:** Agent Hacks Agent: Autoresearch for Production-Agent Red-Teaming  
- **Authors:** Xutao Mao, Xiang Zheng, Cong Wang  
- **Link:** https://arxiv.org/abs/2607.11698 (submitted Jul 2026)  
- **Code (related):** https://github.com/henrymao2004/Auto-research-red-teaming-in-sleep (and related forks/names)  
- **Summary:** One agentic research environment discovers **reusable vulnerability knowledge** about another production-style agent (Claude Code, Codex). Loop: hypothesis → falsifier → sandboxed attack → reflection → **Vulnerability Concept Graph (VCG)** (claim, enabling condition, falsifier, transfer prediction, evidence). Frozen VCG transfers; +14.2 pp vs strongest frozen discovery baseline in single-shot protocol.  
- **Use for data/hardening:** Train agent-safety classifiers on *enabling conditions* and trajectories, not only chat prompts; use VCG for patch validation.

### CoP — Composition of Principles
- **Title:** CoP: Agentic Red-teaming for Large Language Models using Composition of Principles  
- **Venue:** NeurIPS 2025  
- **Summary:** Human principles orchestrated by an agent into new strategies; large single-turn ASR gains reported.  
- **Use for data/hardening:** Principle IDs as taxonomy labels on training rows.

### Adaptive Instruction Composition (AIC)
- **Title:** Adaptive Instruction Composition for Automated LLM Red-Teaming  
- **Authors:** Zymet et al. (Capital One)  
- **Link:** https://arxiv.org/abs/2604.21159  
- **Summary:** Contextual bandit over WildJailbreak query×tactic space; >2× ASR vs random composition.  
- **Use for data/hardening:** High-coverage synthetic corpora for a specific target model.

---

## Foundational (still required background)

| Paper | Link / note |
|-------|-------------|
| GCG universal attacks | https://llm-attacks.org/ |
| PAIR | arXiv:2310.08419 |
| TAP | Tree of Attacks |
| Crescendo | arXiv:2404.01833 |
| GOAT | arXiv:2410.01606 |
| Best-of-N Jailbreaking | arXiv:2412.03556 |
| WildTeaming / WildJailbreak | arXiv:2406.18510 |
| HarmBench | Mazeika et al. |
| JailbreakBench | Chao et al. |
| ReFAT | arXiv:2409.20089 |
| HASTE | LAST-X 2026 / NDSS workshop line |
| LRM as jailbreak agents | Nature Communications 2026 |

---

## Living paper lists

- https://github.com/chen37058/Red-Team-Arxiv-Paper-Update — auto-updated arXiv red-team / jailbreak list  
- Awesome multimodal jailbreak indexes (community)

---

## How this maps to our **implemented** code

| External idea | Implementation in this repo | Path |
|---------------|----------------------------|------|
| Auto-RT strategy RL + early stop + progressive reward | `AutoRTExplorer`, `ProgressiveRewardTracker` | `autoredteam/research/auto_rt.py` |
| AutoRedTeamer lifelong attack memory | `LifelongAttackMemory` | `autoredteam/research/memory.py` |
| AutoRedTeamer strategy proposer | `StrategyProposer` | `autoredteam/research/strategy_proposer.py` |
| AHA VCG (claim, enabling condition, falsifier, transfer) | `VulnerabilityConceptGraph` | `autoredteam/research/vcg.py` |
| AHA discovery + Jailbreak-autoresearch keep/revert | `AutoresearchLoop`, `KeepRevertLoop` | `autoredteam/research/autoresearch.py` |
| AHA production-agent sandbox victim (tool FS + canary policy) | `ProductionAgentVictim`, `ProductionAgentHarness`, `SandboxFS`, `run_attack` | `autoredteam/research/production_agent.py` |
| CoP Composition of Principles | `CompositionOfPrinciples` (`CoP`), `Principle`, `CompositionResult` | `autoredteam/research/cop.py` |
| AIC Adaptive Instruction Composition bandit | `AdaptiveInstructionComposer` (`AIC`), `Tactic`, educational tactic catalog | `autoredteam/research/aic.py` |
| Pipeline integration | phase **research** (10): memory + VCG + Auto-RT + offline CoP/AIC | `autoredteam/pipeline/phases.py` |
| CLI / scripts | `auto-redteam research`, `scripts/run_research_loop.py` | — |

**Run (Windows):**
```bash
.venv\Scripts\python.exe scripts/run_research_loop.py --from-pipeline runs/local-gemma4b-full-pipeline
.venv\Scripts\python.exe scripts/run_research_loop.py --mock --limit 10
.venv\Scripts\python.exe scripts/run_full_pipeline.py --from-phase research
```

**Run (Linux / python3):**
```bash
python3 scripts/run_research_loop.py --mock --limit 10
python3 scripts/run_full_pipeline.py --from-phase research
```

Mock research extras write `cop_stats.json`, `aic_stats.json`, and `production_agent_episode.json` under the run `out` dir. Pipeline phase **research** also exports offline CoP compositions over failures and AIC tactic recommendations (no LLM by default).

| External idea | Future extension |
|---------------|------------------|
| Full Auto-RT multi-model progressive downgrade stack | Optional intermediate judge models |
| AIC neural / SBERT contextual bandit at WildJailbreak scale | Deeper embeddings over full tactic×query library |
| Live LLM tool-calling backend for production-agent victim | Wire a real tool-calling provider beyond offline mock |

---

*Defensive research bibliography. Authorized use only.*
