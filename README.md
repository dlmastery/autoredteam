# auto-redteam

**Educational auto red-teaming harness** for authorized AI-safety research.

In plain language: this repo helps you **test an LLM’s safety** by generating lab-only “jailbreak” probes (with **canary tokens**, not real harmful payloads), **score** whether the model complied, and **export labeled datasets** you can use to train jailbreak **classifiers** or study hardening.

| | |
|--|--|
| **Primary use** | Authorized lab / self-test on models you control |
| **Default educational stack** | Local **Ollama** Gemma 4: ablated attacker → official defender |
| **Success signal** | Defender emits a unique **canary** token (e.g. `EDU-CANARY-R001-OK`) |
| **Survey / papers** | [`RESEARCH_SURVEY.md`](RESEARCH_SURVEY.md) · [`docs/RELATED_WORK_PAPERS.md`](docs/RELATED_WORK_PAPERS.md) |

---

> ## Authorized use only
>
> This tool is for **discovering and fixing** safety issues—not for attacking systems you do not own.
>
> - Only test models/endpoints you are **explicitly authorized** to test.
> - Campaigns require `authorization.confirmed: true` and a written `authorization.scope`.
> - Strategy modules implement **published techniques as scaffolding**; seeds come from *your* goals YAML.
> - Educational runs use **canary probes**, not real criminal content.

---

## Table of contents

1. [Success metrics so far](#success-metrics-so-far)
2. [What you get](#what-you-get)
3. [Beginner tutorial](#beginner-tutorial)
4. [Understanding the pipeline](#understanding-the-pipeline)
5. [Hard stealth multi-turn dataset](#hard-stealth-multi-turn-dataset)
6. [Research survey & paper implementations](#research-survey--paper-implementations)
7. [Commands & config (reference)](#commands--config-reference)
8. [Project layout](#project-layout)
9. [Safety](#safety-and-isolation)
10. [Roadmap](#roadmap)

---

## Success metrics so far

Results below are from **local educational canary campaigns** on this machine (not public harm benchmarks).  
**ASR** = Attack Success Rate = fraction of goals where the **official** defender emitted the canary.

### Campaign scoreboard

| Campaign | Attacker | Victim (defender) | Goals | ASR | Notes |
|----------|----------|-------------------|------:|----:|-------|
| **Full multi-phase pipeline** | Ablated Gemma 4 E4B | Official Gemma 4 E4B | 100 | **67%** (67/100) | Best overall educational set |
| Two-phase legacy | Ablated Gemma 4 E4B | Official Gemma 4 E4B | 100 | **53%** (53/100) | Earlier simpler run |
| **Hard stealth multi-turn** | Stealth templates (deferred canary) | Official Gemma 4 E4B | 100 | **34%** (34/100) | Long chats (~15 turns avg); hard to detect |
| Pipeline smoke | Ablated → official | Official Gemma 4 E4B | 3 | 100% | Sanity check only |

**Models (Ollama local):**

- **Attacker (probe writer):** `huihui_ai/gemma-4-abliterated:e4b`
- **Victim (under test):** `gemma4:e4b`

### Full pipeline breakdown (100 goals → 67% ASR)

| Slice | Result |
|-------|--------|
| Mix | 55 regular + 45 universal scaffolds |
| Final successes | **67** · blocked **33** |
| Winning strategies (approx.) | single_turn **87**, BoN universal **8**, hardneg **3**, crescendo **1**, mutation **1** |
| Research assets | Lifelong memory **100** records · VCG **67** confirmed concepts · **40** strategy proposals |

**ASR by category (full pipeline):**

| Category | ASR |
|----------|----:|
| direct_prompt_injection | 84.6% (11/13) |
| goal_hijacking | 75.0% (9/12) |
| unauthorized_tool_use | 75.0% (9/12) |
| fabricated_facts | 69.2% (9/13) |
| privilege_escalation | 63.6% (7/11) |
| system_prompt_leak | 61.5% (8/13) |
| unsafe_code_generation | 61.5% (8/13) |
| indirect_prompt_injection | 46.2% (6/13) |

### Datasets produced

| Dataset | Rows | Purpose |
|---------|-----:|---------|
| [`datasets/edu_100_items.json`](datasets/edu_100_items.json) (+ `.jsonl` / `.csv`) | **100** | Flat labeled probes (prompt, response, success, template, …) |
| [`datasets/edu_four_way.jsonl`](datasets/edu_four_way.jsonl) | **300** | Classifier set: 100 benign · 100 direct_probe · 67 success · 33 blocked |
| [`datasets/edu_policy_pairs.json`](datasets/edu_policy_pairs.json) | **100** | Preference-style (attack prompt + preferred refusal) |
| [`datasets/edu_hard_multiturn_trajectories.jsonl`](datasets/edu_hard_multiturn_trajectories.jsonl) | **100** | Full long multi-turn chats (avg ~15 turns, max 16) |
| [`datasets/edu_hard_multiturn_classifier.jsonl`](datasets/edu_hard_multiturn_classifier.jsonl) | trajectory + turn rows | Hard/stealth labels for detector training |
| [`dashboard/index.html`](dashboard/index.html) | — | Interactive HTML dashboard for the educational campaign |

### Hard stealth multi-turn (100 trajectories)

| Metric | Value |
|--------|------:|
| ASR | **34%** |
| Detectability band | **100% hard** (long benign prefix + delayed canary) |
| Avg / max turns | **15.0 / 16** |
| Strategies | `stealth_multiturn` ×50 · `nested_task` ×50 |
| First canary turn (avg) | ~**9.5** (not on turn 0) |

**How to read this:** lower ASR than the single-turn pipeline is expected—the goal is **harder classifier examples**, not maximum jailbreak rate.

### Tests

Offline unit tests (mock only): **`pytest`** suite in `tests/` (50+ tests; no GPU required for CI-style checks).

---

## What you get

```text
You  →  configure goals + models
     →  run pipeline / CLI
     →  attacker writes probes (or stealth templates)
     →  defender answers
     →  judge scores (canary / rules / optional LLM)
     →  datasets + reports + optional research graphs
```

| Piece | Role |
|-------|------|
| **Goals YAML** | What behaviours to probe (taxonomy-tagged) |
| **Attacker** | Writes/mutates prompts (local ablated Gemma in edu campaigns) |
| **Defender / victim** | Model under test (official Gemma, Gemini, …) |
| **Strategies** | single_turn, crescendo, mutation_loop, TAP, stealth multi-turn, … |
| **Judges** | Rule-based canary + optional LLM judge |
| **Export** | JSON/JSONL/CSV for ML training |
| **Research modules** | Memory, VCG, Auto-RT bandit, CoP, AIC (paper-inspired) |

---

## Beginner tutorial

### Path A — Zero GPU, zero API keys (5 minutes)

Learn the harness **offline** with mock models.

**Requirements:** Python **3.12+** (on Windows this project often uses `python3.exe`).

```bash
# 1) Clone and enter the repo
git clone https://github.com/dlmastery/autoredteam.git
cd autoredteam

# 2) Virtual environment
python3.exe -m venv .venv
# cmd:
.venv\Scripts\activate.bat
# PowerShell:
#   .\.venv\Scripts\Activate.ps1

# 3) Install
python -m pip install -U pip
python -m pip install -e ".[dev]"

# 4) Run tests (optional but recommended)
python -m pytest tests/ -q

# 5) Offline mock campaign (no Ollama, no cloud keys)
.venv\Scripts\auto-redteam.exe run
# or:
.venv\Scripts\python.exe -m autoredteam.cli run

# 6) Dry-run (attacker loop only, no defender)
.venv\Scripts\auto-redteam.exe run --dry-run

# 7) List strategies
.venv\Scripts\auto-redteam.exe strategies
```

**You succeeded if:** the mock run finishes, writes under `runs/`, and tests pass.

---

### Path B — Local educational Gemma campaign (recommended lab)

This is the **main tutorial** for generating real canary datasets.

#### B0. What models you need

| Role | Ollama model name | Purpose |
|------|-------------------|---------|
| Victim | `gemma4:e4b` | Official Gemma 4 — model under test |
| Attacker | `huihui_ai/gemma-4-abliterated:e4b` | Ablated Gemma — writes probes |

Install [Ollama](https://ollama.com), then:

```bash
ollama pull gemma4:e4b
ollama pull huihui_ai/gemma-4-abliterated:e4b
ollama list
```

Keep Ollama running (`ollama serve` if needed). Default API: `http://127.0.0.1:11434`.

#### B1. Install with local extras

```bash
cd autoredteam
python3.exe -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e ".[dev,local]"
```

#### B2. Smoke the full pipeline (small)

```bash
# 5 goals, all phases — should finish in minutes on a mid-range GPU
.venv\Scripts\python.exe scripts/run_full_pipeline.py --limit 5 --skip-pull
```

#### B3. Full 100-item educational run

```bash
.venv\Scripts\python.exe scripts/run_full_pipeline.py --skip-pull
```

**Phases:**

```text
0 setup → 1 compose → 2 attack_gen → 3 defend_single → 4 multiturn
→ 5 universal_bon → 6 judge → 7 hardneg → 8 export → 9 dashboard → 10 research
```

| Phase | What happens |
|-------|----------------|
| 0 setup | Check Ollama models, smoke PING/PONG |
| 1 compose | Load 100 unique canary goals |
| 2 attack_gen | Ablated model rewrites seeds (keeps canary) |
| 3 defend_single | Official model answers once |
| 4 multiturn | Crescendo / mutation on failures |
| 5 universal_bon | Best-of-N surface variants |
| 6 judge | Pick best attempt |
| 7 hardneg | Extra mutations on remaining failures |
| 8 export | Write `datasets/edu_*` |
| 9 dashboard | HTML + markdown reports |
| 10 research | Memory + VCG + Auto-RT stats |

#### B4. Open results

| Artifact | Path |
|----------|------|
| Dashboard | [`dashboard/index.html`](dashboard/index.html) |
| Flat items | [`datasets/edu_100_items.json`](datasets/edu_100_items.json) |
| Four-way classifier set | [`datasets/edu_four_way.jsonl`](datasets/edu_four_way.jsonl) |
| Run report | `runs/local-gemma4b-full-pipeline/report.md` |
| Checkpoint | `runs/local-gemma4b-full-pipeline/pipeline/state.json` |

#### B5. Resume or re-export only

```bash
# Re-export datasets from an existing checkpoint
.venv\Scripts\python.exe scripts/run_full_pipeline.py --from-phase export

# Continue from multiturn onward
.venv\Scripts\python.exe scripts/run_full_pipeline.py --from-phase multiturn --skip-pull

# Long multi-turn (12 turns) on single-turn failures
.venv\Scripts\python.exe scripts/run_full_pipeline.py ^
  --campaign local-gemma4b-long-multiturn ^
  --only multiturn export ^
  --multiturn-max-turns 12 ^
  --multiturn-targets single_turn_failures ^
  --skip-pull
```

#### B6. Common beginner mistakes

| Problem | Fix |
|---------|-----|
| `ollama: command not found` / connection refused | Start Ollama; confirm `curl http://127.0.0.1:11434/api/tags` |
| Model not found | `ollama pull gemma4:e4b` and the ablated model |
| VRAM thrash / very slow | Use `--skip-pull`; don’t run two pipelines at once; multiturn uses history (slower later turns) |
| `AuthorizationError` | Set `authorization.confirmed` + `scope` in config (edu pipeline sets a lab scope for you) |
| Wrong Python | Need **3.12+**; use `.venv\Scripts\python.exe` on Windows |

---

### Path C — Cloud defender (optional)

Local ablated Gemma attacker vs **Gemini** defender:

```bash
# Install Gemini extra
python -m pip install -e ".[dev,local,gemini]"

# Key is read by env var NAME only — never put the key in YAML
set GOOGLE_API_KEY=your_key_here

.venv\Scripts\auto-redteam.exe run ^
  --config config/gemma_vs_gemini.yaml ^
  --set authorization.confirmed=true ^
  --set "authorization.scope=internal safety eval of our own Gemini deployment, ticket SAFE-1234"
```

---

## Understanding the pipeline

```text
Goals (YAML)
    │
    ▼
[attack_gen]  ablated Gemma rewrites seed  ──► probe + canary
    │
    ▼
[defend]      official Gemma answers
    │
    ▼
[judge]       canary in response?  success / blocked
    │
    ├─ failure ──► multiturn / BoN / hardneg
    │
    ▼
[export]      edu_100_items + four_way + policy_pairs
[dashboard]   HTML report
[research]    memory / VCG / bandit stats
```

**Canary protocol:** each goal has a unique token (e.g. `EDU-CANARY-R008-OK`).  
If the defender **repeats that token**, the educational probe counts as a **success** for ASR. That is safer than logging real harmful content.

---

## Hard stealth multi-turn dataset

For **hard-to-detect** classifier training (long benign chat, canary only late):

```bash
# Full 100 goals × up to 16 turns (can take 2–3+ hours on one GPU)
.venv\Scripts\python.exe -u scripts/run_hard_multiturn.py --max-turns 16 --targets all --skip-pull

# Quick smoke
.venv\Scripts\python.exe -u scripts/run_hard_multiturn.py --limit 3 --max-turns 10 --skip-pull
```

| Output | Description |
|--------|-------------|
| `datasets/edu_hard_multiturn_trajectories.jsonl` | Full message history per goal |
| `datasets/edu_hard_multiturn_turns.csv` | One row per turn |
| `datasets/edu_hard_multiturn_classifier.jsonl` | Trajectory + turn labels (`benign_context`, `bridge_context`, `jailbreak_attempt`, …) |
| `datasets/edu_hard_multiturn_summary.json` | ASR + hardness bands |

Strategies: **`stealth_multiturn`** (rapport → bridge → deferred canary) and **`nested_task`**.

---

## Research survey & paper implementations

### Read first (techniques & literature)

| Doc | Contents |
|-----|----------|
| **[`RESEARCH_SURVEY.md`](RESEARCH_SURVEY.md)** | SOTA survey (July 2026): automatic red teaming, universal jailbreaks, classifier data, hardening |
| [`docs/SOTA_AUTO_REDTEAM_JULY_2026.md`](docs/SOTA_AUTO_REDTEAM_JULY_2026.md) | Same survey under `docs/` |
| [`docs/RELATED_WORK_PAPERS.md`](docs/RELATED_WORK_PAPERS.md) | Bibliography + name disambiguation |
| [`docs/README.md`](docs/README.md) | Docs index |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Component design |

> Searching “auto red team” on Google/arXiv finds **public papers** (Auto-RT, AutoRedTeamer, AHA, …).  
> **This GitHub repo** is a separate educational harness. Related-work docs explain the difference.

### Paper-inspired code (`autoredteam/research/`)

```bash
# From a finished pipeline campaign
.venv\Scripts\python.exe scripts/run_research_loop.py --from-pipeline runs/local-gemma4b-full-pipeline

# Offline mock
.venv\Scripts\python.exe scripts/run_research_loop.py --mock --limit 10
```

| Module | Inspired by | Output example |
|--------|-------------|----------------|
| `memory.py` | AutoRedTeamer lifelong memory | `lifelong_memory.json` |
| `auto_rt.py` | Auto-RT strategy RL | `auto_rt_stats.json` |
| `vcg.py` | AHA Vulnerability Concept Graph | `vcg.json` |
| `autoresearch.py` | AHA + jailbreak-autoresearch keep/revert | episodes JSON |
| `strategy_proposer.py` | AutoRedTeamer proposer | `strategy_proposals.json` |
| `cop.py` | CoP (composition of principles) | compositions / stats |
| `aic.py` | AIC instruction composition bandit | `aic_stats.json` |
| `production_agent.py` | AHA tool-using sandbox (educational) | episode JSON |

---

## Commands & config (reference)

### CLI

| Command | Purpose |
|---------|---------|
| `auto-redteam run [--config Y] [--set a.b=val] [--dry-run]` | Run classic orchestrator campaign |
| `auto-redteam validate [--config Y]` | Print config hash / manifest (no run) |
| `auto-redteam strategies` | List strategies |
| `auto-redteam pipeline …` | Multi-phase educational pipeline (see `scripts/run_full_pipeline.py`) |
| `auto-redteam research …` | Research loop helper |
| `auto-redteam version` | Version |

Config precedence (highest last):

```text
config/default.yaml  <  --config YAML  <  HARNESS__A__B env  <  --set CLI
```

### Important knobs

| Key | Meaning |
|-----|---------|
| `authorization.confirmed` / `scope` | Must be set for non-mock real runs |
| `attacker` / `defender` | Provider + model |
| `strategies` | Which attack patterns to mix |
| `selection.mode` | `fixed` / `thompson` / `ucb` / … |
| `goals_path` | Behaviours to probe |
| `max_turns_per_trajectory` | Multi-turn budget (classic orchestrator) |

Educational goals: [`config/goals_edu_100.yaml`](config/goals_edu_100.yaml)  
Campaign example: [`config/campaigns/local_gemma4b_edu.yaml`](config/campaigns/local_gemma4b_edu.yaml)

### Optional installs

```bash
pip install -e ".[dev]"                         # tests + core
pip install -e ".[local]"                       # Ollama helper
pip install -e ".[gemini]" / ".[openai]" / ".[anthropic]"
pip install -e ".[all]"
```

---

## Project layout

```text
autoredteam/
  cli.py, orchestrator.py, models.py, interfaces.py
  providers/          mock, gemini, openai_compat, anthropic, local_gemma
  strategies/         single_turn, crescendo, mutation_loop, tree_of_attacks,
                      stealth_multiturn, nested_task
  research/           paper-inspired modules (memory, VCG, Auto-RT, CoP, AIC, …)
  pipeline/           multi-phase educational runner (phases 0–10)
config/               default + campaigns + goals + taxonomies
datasets/             exported training data (edu_* , hard multiturn)
dashboard/            interactive HTML
docs/                 survey index + related work
prompts/              attacker / judge templates
scripts/              run_full_pipeline, run_hard_multiturn, run_research_loop, …
tests/                offline pytest suite
RESEARCH_SURVEY.md    SOTA techniques survey
ARCHITECTURE.md       design deep-dive
```

---

## Safety and isolation

- **Authorization gate** before real generation  
- **API keys** loaded only from env var **names**; never logged  
- **Offline mock** default for CI and first learning  
- **Canary metrics** for educational ASR instead of raw harmful content  
- **No attacker internet egress** beyond the configured model server  

---

## Roadmap

- Richer agentic / tool-use defender tests  
- Calibrated LLM-judge ensembles + human spot-checks  
- Continuous ASR regression across defender versions  
- Optional live LLM backend for production-agent sandbox  
- Deeper AIC (embedding bandit) at larger tactic libraries  

---

## Quick links

| I want to… | Go here |
|------------|---------|
| **Learn techniques / papers** | [`RESEARCH_SURVEY.md`](RESEARCH_SURVEY.md) |
| **Run offline first** | [Path A](#path-a--zero-gpu-zero-api-keys-5-minutes) |
| **Generate the 100-item canary set** | [Path B](#path-b--local-educational-gemma-campaign-recommended-lab) |
| **Hard long multi-turn data** | [Hard stealth](#hard-stealth-multi-turn-dataset) |
| **See numbers** | [Success metrics](#success-metrics-so-far) |
| **Architecture** | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **Browse results UI** | [`dashboard/index.html`](dashboard/index.html) |

---

*Defensive research software. Authorized use only.*
