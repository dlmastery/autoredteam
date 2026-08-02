# State of the Art: Automatic Red Teaming for Jailbreak Data & LLM Hardening

**Survey cutoff:** end of July 2026  
**Scope:** techniques that *automatically* generate adversarial prompts (especially **universal / transferable** jailbreaks) usable as **training data for jailbreak classification** and for **hardening** the target LLM (refusal, guardrails, adversarial training).  
**Intended use:** authorized defensive research only.

---

> **Location note:** Canonical copy lives at [`docs/SOTA_AUTO_REDTEAM_JULY_2026.md`](docs/SOTA_AUTO_REDTEAM_JULY_2026.md).  
> Companion paper list: [`docs/RELATED_WORK_PAPERS.md`](docs/RELATED_WORK_PAPERS.md).  
> This root-level file is the same survey for easy discovery.


## 1. Framing: two products of red teaming

Automatic red teaming is no longer only “find a cool ASR number.” By mid-2026 the industrially useful outputs are:

| Product | What you ship | How it hardens the model |
|--------|----------------|---------------------------|
| **A. Jailbreak / attack corpus** | Labeled prompts (and often multi-turn trajectories) spanning tactics × harm categories | Train **input classifiers**, guard models (Llama-Guard family, NemoGuard JailbreakDetect, etc.), or retrieval/filter layers |
| **B. Behavior-paired safety data** | `(adversarial_prompt, safe_refusal)` and hard-benign near-misses | **SFT / DPO / RLAIF / adversarial training** of the policy itself |
| **C. Universal artifacts** | Suffixes, prefixes, image/audio triggers, tactic compositions that transfer | Stress-test generalization; seed transfer attacks; build OOD test sets |

This survey prioritizes methods that scale **A** and **B**, with special attention to **universal** (one artifact, many goals) and **transferable** (one target, many models) attacks.

---

## 2. Landscape map (2023 → mid-2026)

```
                    ACCESS MODEL
         white-box ────────────── black-box
              │                      │
   GCG / IRIS / ReFAT-style     PAIR / TAP / BoN
   continuous AT (CAT/LAT)      Crescendo / GOAT
   UltraBreak (VLM)             WildTeaming / AIC
              │                      │
              └──────────┬───────────┘
                         │
              agentic multi-turn LRM attackers
              (2026: ~97% ASR in Nature Comm study)
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
     training data for          policy hardening
     jailbreak classifiers      (refusal AT, MART,
     (hard-neg mining: HASTE)    WildJailbreak SFT)
```

**Three research waves:**

1. **2023–early 2024 — Optimization & first auto loops**  
   GCG (universal transferable suffixes), PAIR, AutoDAN, early RL red-teamers (Perez et al.).

2. **2024–2025 — Composition, multi-turn, scale**  
   WildTeaming / WildJailbreak, Crescendo, TAP, GOAT, Best-of-N, HarmBench / JailbreakBench standardization.

3. **2025–July 2026 — Learning over attack space + lifelong agents + production-agent autoresearch**  
   Adaptive Instruction Composition (bandits over WildJailbreak), AutoDAN-Turbo lifelong tactics, **Auto-RT** (RL strategy exploration), **AutoRedTeamer** (multi-agent + memory-guided lifelong attack integration), **CoP**, HASTE hard-negative detector training, LRM-as-jailbreak-agent (~97% multi-model ASR), **AHA / Agent Hacks Agent** (autoresearch for *production agents*, Vulnerability Concept Graph), Jailbreak-autoresearch evolutionary loops, MLCommons mechanism-first jailbreak taxonomy, multimodal universals (UltraBreak ICLR 2026).

> **Name disambiguation (important):**  
> - **This local repo** (`steeringresearch/auto-redteam` → `~/autoredteam`) is an *educational* harness for chat LLM canary red-teaming and classifier data.  
> - **AutoRedTeamer** (Zhou et al., arXiv:2503.15754) is a *different* published multi-agent system with lifelong attack memory.  
> - **AHA / Auto-research-red-teaming** (Mao et al., arXiv:2607.11698; GitHub `henrymao2004/...`) targets *production coding agents* (Claude Code, Codex), not the same codebase.  
> Searching the web for “auto red team” will surface those papers—not this private/local project.

---

## 3. Technique families (SOTA inventory)

### 3.1 White-box / gradient-based universal attacks

| Method | Core idea | Universality | Good for classifier data? |
|--------|-----------|--------------|---------------------------|
| **GCG** (Zou et al., 2023) | Discrete token optimization of a **suffix** maximizing likelihood of affirmative harmful prefixes | Strong: one suffix, many behaviors; transfers open→closed | Yes — diverse suffix variants; but low stealth / high perplexity |
| **IRIS / stronger universal suffixes** (Huang et al., NAACL 2025-ish line) | Suppress refusal-related dynamics; improve transfer of single-behavior optimized suffixes | High transfer when refusal is shared | Yes — hard positives for detectors sensitive to gibberish suffixes |
| **AutoDAN / AutoDAN-Turbo** | Genetic / hierarchical search for **stealthier** readable prompts; Turbo stores lifelong strategies | Per-prompt or strategy-level reuse | Excellent — natural-language attacks closer to real user distribution |
| **Continuous AT (CAT)** | Continuous embedding-space adversarial training | Defense more than attack | Indirect: generates latent hard examples |
| **UltraBreak** (ICLR 2026, VLMs) | Universal **vision** adversarial patterns + semantic textual objectives | Cross-goal, cross-model VLM | Multimodal classifier / policy data |

**Practical note for hardening:** gradient universals remain the gold standard for **worst-case** robustness claims, but pure GCG-style suffixes overfit detectors to “weird tokens.” Mix with natural-language strategies for a realistic classifier prior.

### 3.2 Black-box iterative LLM-as-attacker

| Method | Loop | Strengths | Limits |
|--------|------|-----------|--------|
| **PAIR** | Attacker proposes → target answers → judge scores → attacker revises (~20 queries) | Cheap, black-box, readable | Narrow strategy diversity; often “same few persona tricks” |
| **TAP** (Tree of Attacks) | Tree search + pruning of attack branches | Broader exploration than PAIR | Compute grows with branching |
| **Rainbow Teaming / multi-objective** | Diversity-aware search | Better coverage of taxonomy | Harder to tune |
| **AutoDAN-Turbo** | Lifelong agent: store successful strategies, reuse | Strong HarmBench ASRs in published tables | Strategy library can collapse without diversity pressure |

These produce **labeled trajectories**: `(goal, attack_t, response_t, score_t)`. That is ideal for:

- training **process** judges (is this turn a jailbreak?),
- training **multi-turn** detectors,
- preference data (winning vs losing rewrites).

### 3.3 Multi-turn conversational escalation

| Method | Mechanism | Why it matters for data |
|--------|-----------|------------------------|
| **Crescendo** (Microsoft; USENIX Sec 2025) | Benign start → gradual escalation using model’s own prior answers | Single-turn classifiers fail; need **conversation-level** labels |
| **Crescendomation** (in PyRIT) | Automated Crescendo | Scalable multi-turn corpora |
| **GOAT** (Meta; ICML 2025) | Generative Offensive Agent Tester: multi-turn agent chaining techniques; high ASR@10 (reported ~97% Llama 3.1, ~88% GPT-4 on JailbreakBench) | Realistic “plain language” trajectories |
| **ActorAttack** | Self-discovered clues / multi-turn derailment | Additional multi-turn diversity |
| **Sequential / scaffolded jailbreaks** (DeepTeam et al.) | Narrative scaffolding over turns | Same |

**Hardening implication:** if you only train a prompt classifier on single-turn GCG/PAIR data, you systematically miss Crescendo/GOAT-style failures. Dual corpora are required.

### 3.4 Compositional / in-the-wild mining (best for bulk training data)

#### WildTeaming → WildJailbreak (Ai2 / AllenAI, 2024)

- **Mine** jailbreak tactics from real chat logs (LMSYS-Chat-1M, WildChat) → ~5.7K tactic clusters; tens of thousands of tactic strings.  
- **Compose** tactic(s) + harmful query via LLM rewriter.  
- **Prune** off-topic / low-risk.  
- **Product:** **WildJailbreak** — ~**262K** synthetic safety pairs with four buckets:
  1. Vanilla harmful  
  2. Vanilla benign (near-misses to fight over-refusal)  
  3. Adversarial harmful (composed jailbreaks)  
  4. Adversarial benign  

This is still the **reference open dataset** for training safer models *and* for bootstrapping jailbreak/benign classifiers without exaggerated refusals.

#### Adaptive Instruction Composition — AIC (Capital One, Apr 2026, arXiv:2604.21159)

SOTA upgrade on WildTeaming for **targeted data generation**:

- Action space: combine WildJailbreak queries (~50.5K) × tactics (~13.3K) → **trillions** of instruction compositions.  
- **Neural Thompson Sampling** bandit over SBERT (contrastive) embeddings of candidates (~2.2K parameters).  
- Modes: **subtle** (explore / diversity) vs **aggressive** (exploit / high ASR).  
- Reported: **>2× ASR** vs random WildTeaming on Mistral-7B / Llama-3-70B / Llama-3.3-70B; near-full HarmBench behavior coverage within 150 attempts/behavior after pretraining.  
- **Cross-model transfer** of bandit policies without retrain.  
- Explicit design goal: **safety training sets tailored to a specific deployed model’s vulnerabilities**.

**For classifier data pipelines, AIC is currently the best “coverage per GPU-hour” compositional generator** when you already have a tactic/query library.

#### CoP — Composition of Principles (NeurIPS 2025)

- Human supplies **red-team principles**; an agent composes them into strategies.  
- Extensible principle library → new strategies without full RL.  
- Complements AIC: principles ≈ interpretable tactic ontology.

### 3.5 Sampling / augmentation universals (Best-of-N)

**Best-of-N Jailbreaking** (Hughes et al.; NeurIPS 2025):

- Black-box: apply many **random modality-specific augmentations** (typos, case, separators, vision/audio transforms) until one works.  
- Reported high ASRs (e.g. ~89% GPT-4o, ~78% Claude 3.5 Sonnet at N≈10k text samples).  
- ASR vs N follows **power-law-like** scaling.  
- Composes with prefix/suffix methods (+~35% ASR in paper).  
- Circumvents some “circuit breaker” style defenses.

**Data value:** each failed/succeeded augmentation is a free **label** for invariance training of detectors (“same intent, different surface form”). Cheap negative/positive mining at scale.

### 3.6 Lifelong / RL / multi-agent auto red-team systems (2025–2026 SOTA wave)

These systems go beyond a fixed strategy menu: they **search, store, and reuse** attack knowledge.

| System | Venue / ID | Core idea | Relevance to classifier data & hardening |
|--------|------------|-----------|------------------------------------------|
| **Auto-RT** | arXiv:2501.01830; ICLR 2026 poster line | **RL** framework for automatic *jailbreak strategy* exploration. Early-terminated exploration + progressive reward tracking with intermediate “downgrade” models. Reports ~**16.6% higher** ASR and faster detection vs prior methods. | Strategy-level trajectories (not only prompts) → multi-turn detector training; curriculum for hard-neg mining |
| **AutoRedTeamer** | arXiv:2503.15754 (Zhou, Wu, Pinto, … Li); NeurIPS 2025 poster track | Fully automated **end-to-end** multi-agent red team: red-teaming agent + **strategy proposer** that reads research and implements new attacks; **memory-guided** lifelong attack integration. ~**+20% ASR** on HarmBench (Llama-3.1-70B) with **~46% less compute**. | Closest “productized research agent” pattern; attack memory ≈ lifelong library for data export |
| **CoP** (Composition of Principles) | NeurIPS 2025 | Agentic orchestration of human-provided red-team **principles** into new strategies | Interpretable tactic ontology for labeled data |
| **AHA — Agent Hacks Agent** (*Autoresearch for Production-Agent Red-Teaming*) | arXiv:2607.11698 (Mao, Zheng, Wang), Jul 2026; code: GitHub `henrymao2004/Auto-research-red-teaming*` | **Autoresearch** loop against *production agents* (Claude Code, Codex): hypothesis → falsifier → sandboxed attack → reflection → **Vulnerability Concept Graph (VCG)** (claim, enabling condition, transfer prediction, evidence). Frozen VCG beats discovery baselines by **+14.2 pp** single-shot; concepts transfer across scenarios/channels. | Artifact is *reusable vulnerability knowledge*, not just prompts—gold for agent-safety classifiers and patch validation |
| **Jailbreak-autoresearch** | open research loops (e.g. evolutionary header/footer search, multi-model scoring) | Greedy ratchet: edit one part, evaluate, keep/revert—inspired by “autoresearch” experiment loops | Cheap continuous data generation for classifiers |
| **AIC** (Adaptive Instruction Composition) | arXiv:2604.21159, Apr 2026 | Neural Thompson bandit over WildJailbreak query×tactic compositions | Model-specific high-coverage corpora |

**Takeaway for this repo:** chat-LLM canary campaigns (our pipeline phases 0–9) sit in the **PAIR/WildTeaming/BoN** family. The **2026 frontier for production** is shifting to (1) lifelong multi-agent systems like AutoRedTeamer and (2) **agent-vs-agent autoresearch** (AHA) that outputs concept graphs, not only ASR tables. Future work for hardening agents should import VCG-style *enabling conditions* into the four-way training set, not only prompt/response pairs.

### 3.7 Large reasoning models as autonomous red-team agents (2026 shift)

**Nature Communications (Feb 2026)** — *Large reasoning models are autonomous jailbreak agents*:

- LRMs (DeepSeek-R1, Gemini 2.5 Flash, Grok 3 Mini, Qwen3-235B, …) given a system prompt plan and execute multi-turn jailbreaks **without further supervision**.  
- **~97% overall ASR** across model combinations in their setup.  
- Framing: **alignment regression** — stronger reasoners become cheap commodity attackers.

**Implication for data generation (July 2026):**  
The attacker no longer needs a custom bandit or GCG GPU loop for many black-box targets. An LRM + harm taxonomy + judge already yields large, diverse multi-turn corpora. The research edge moves to:

1. **diversity control** (avoid mode collapse to the same 5 persuasion scripts),  
2. **judge reliability**,  
3. **closed-loop hardening** (immediately train on failures).

### 3.8 Multimodal universal jailbreaks

- **UltraBreak** (ICLR 2026): universal transferable **image** jailbreaks for VLMs; regularize vision-space patterns, optimize semantic text objectives.  
- **Odysseus** (NDSS 2026 line): commercial multimodal agent jailbreaks (optimization + domain transfer).  
- BoN extends to vision/audio with modality-specific augmentations.

For hardening: classifiers and policies need **cross-modal** attack data if the product is multimodal.

---

## 4. Universal jailbreaks: what “universal” means in 2026

| Sense of universal | Example | Transfer story |
|--------------------|---------|----------------|
| **Many behaviors, one artifact** | GCG suffix works on 50+ harmful goals | Train once, test on HarmBench behaviors |
| **Many models, one artifact** | Suffix/image/tactic transfers open→closed | Shared refusal geometry / shared pretraining |
| **Many surface forms, one intent** | BoN augmentations of one request | Detector must be intent-aware, not string-matched |
| **Many tactics, one composition policy** | AIC bandit policy transfers Mistral↔Llama | Learned “what works on this model class” |

**Mechanistic backdrop (still active 2025–26):**  
Attacks often act by **ablating / bypassing a refusal feature** in residual stream space (Arditi et al.; ReFAT). That explains why:

- universals transfer across prompts,  
- latent methods (RFA, LAT) can harden without enumerating every string attack,  
- detectors that only look at surface n-grams remain brittle.

---

## 5. Generating training data for **jailbreak classification**

Goal: supervised (or preference) data:

```
x = prompt | conversation | (prompt, response)
y ∈ {jailbreak_attempt, benign, borderline}
   optional: tactic_id, harm_category, success_vs_target, turn_index
```

### 5.1 Gold-standard dataset design (2026 practice)

Borrow the **WildJailbreak four-way split** (critical):

| Class | Purpose |
|-------|---------|
| Harmful vanilla | Positive for “should refuse / flag” |
| Harmful adversarial | Hard positives (jailbreak form) |
| Benign vanilla | Negatives |
| **Adversarial benign** | Hard negatives — look like jailbreaks but ask for allowed content; **kills over-refusal** |

Without adversarial-benign, jailbreak classifiers and over-cautious policies both degrade UX.

### 5.2 Recommended generation stack (defensive)

```
1. Seed goals     ← HarmBench / JailbreakBench / org policy taxonomy / OWASP LLM/ASI
2. Tactic library ← WildJailbreak tactics + CoP principles + in-house logs
3. Composer       ← AIC bandit (subtle mode for diversity) OR WildTeaming random
4. Multi-turn     ← Crescendo / GOAT / LRM agent on top of composed seeds
5. Universal layer← BoN augmentations + (optional) GCG/AutoDAN suffixes
6. Labelers       ← ensemble: rule heuristics + Llama-Guard-class + strong LLM judge
                     + human audit on stratified sample
7. Hard mining    ← HASTE-style: keep only samples that evade current detector
8. Export         ← classifier set (prompt-level) + policy set (prompt, safe_response)
```

### 5.3 HASTE — Hard-negative Attack Sample Training Engine (LAST-X / NDSS workshop 2026)

Palo Alto Networks framework for **proactive detector hardening**:

- Iteratively engineer **evasive** prompts that beat the current detector (~64% drop in detection in their baseline evasion study).  
- Retrain detector on hard negatives → fewer loops than naive regeneration.  
- Supports proactive (stress-test before deploy) and reactive (mimic newly observed attacks) modes.

**This is the SOTA pattern for jailbreak *classifiers* specifically:** red team not the LLM alone, but the **guard model**, in a closed loop.

### 5.4 Label quality (the silent failure mode)

- ASR numbers often use **Llama-Guard / HarmBench classifiers** that disagree with humans.  
- JailbreakBench-style practice: small expert-labeled set (~hundreds) to calibrate judges (FPR/FNR).  
- For multi-turn: label **both** “was the user attempting a jailbreak?” and “did the model comply?”  
- MLCommons (Feb 2026) **Jailbreak 0.7** pushes a **mechanism-first taxonomy** for single-turn attacks to improve reproducibility over ad-hoc outcome labels.

### 5.5 Open datasets & models (starting points)

| Resource | Role |
|----------|------|
| **WildJailbreak** (HF `allenai/wildjailbreak`) | 262K safety pairs; tactic composition DNA |
| **HarmBench** | Standardized behaviors + classifiers |
| **JailbreakBench / JBB-Behaviors** | Behaviors + judge calibration sets |
| **NemoGuard JailbreakDetect** (NVIDIA) | Open jailbreak detector baseline |
| **Llama-Guard** family | Input/output safety classifiers |
| In-the-wild logs (WildChat, LMSYS) | Mine *novel* tactics (WildTeaming MINE stage) |

---

## 6. Hardening the LLM (using the data)

Generating attacks is useless without a hardening path. SOTA options as of July 2026:

### 6.1 Data-centric safety training

- **SFT on safe refusals** for adversarial prompts (WildJailbreak recipe).  
- **Balance** adversarial-harmful with adversarial-benign to limit over-refusal.  
- Scaling laws from WildTeaming: more diverse adversarial data → better robustness with diminishing returns; mixture ratios matter.

### 6.2 Iterative attacker–defender co-training

- **MART** (Meta): multi-round automatic red teaming; adversarial LLM + target co-adapt; large reported drops in violation rate while preserving helpfulness.  
- Progressive / multi-round ART variants continue this line (2024–25).

### 6.3 Mechanistic / latent adversarial training

| Method | Idea | Why efficient |
|--------|------|----------------|
| **ReFAT** (ICLR 2025) | Simulate worst-case attacks by **refusal-feature ablation** during training | Avoids expensive discrete search each step |
| **CAT** | Continuous embedding attacks during training | Differentiable, scalable |
| **LAT** | Noise / adversarial noise in latent space; reorganizes refusal signal | Can improve cross-attack robustness |

2026 theory work (e.g. continuous AT + ICL theory, ICLR 2026) links CAT-style methods to regularization of embedding singular values — i.e. making adversarial token maps harder.

### 6.4 External guardrails (classifier layer)

- Input/output filters trained on hard-mined data (HASTE loop).  
- Ensemble: keyword/heuristic (fast) + small guard model + occasional strong judge.  
- Multi-turn state: escalate suspicion score across conversation (Crescendo defense).  
- Do **not** rely on static banlists alone (Group-IB 2026 threat reporting: commercial jailbreak template services continuously rotate wording).

### 6.5 System-level (beyond the weights)

- Rate limits / BoN cost asymmetry (attackers need N samples; make N expensive).  
- Tool/agent sandboxing (agentic jailbreaks abuse tools, not just text).  
- Continuous red team in CI/CD (2026 industry guidance: every fine-tune retriggers adversarial suite).

---

## 7. Benchmarks, taxonomies, tooling (July 2026)

### Benchmarks

| Benchmark | Use |
|-----------|-----|
| **HarmBench** | Comparable ASR across attack methods; behavior coverage |
| **JailbreakBench** | Open robustness benchmark; judge calibration |
| **MLCommons Jailbreak 0.7** | Mechanism-first single-turn taxonomy for defensible eval |
| Org-specific policy suites | Map to product ToS / regulated domains |

### Tooling (production / research)

| Tool | Notes |
|------|-------|
| **PyRIT** (Microsoft) | Orchestration, multi-turn, Crescendo; used in AIC simulations |
| **garak** | Probe library, broad vulnerability scanning |
| **Promptfoo** | Red-team strategies (BoN, GOAT, multi-turn) in CI |
| **DeepTeam** | Multi-turn strategies (Crescendo, sequential) |
| **Inspect** / commercial platforms | Enterprise continuous red team |

### Your local harness (`auto-redteam`)

Already aligned with several SOTA pieces:

- Strategy modules: single-turn, **crescendo**, mutation, **tree-of-attacks**  
- **Thompson / bandit selection** (same family as AIC)  
- Optional **attacker swarm** (generator + critic)  
- Taxonomy YAML (OWASP ASI-oriented)  
- Mock providers for offline corpus generation  

**Gaps vs July 2026 SOTA** (implementation opportunities):

1. WildJailbreak-style **query × tactic composition** + SBERT candidate embeddings  
2. **BoN** augmentation sampler for universal surface-form coverage  
3. **Hard-negative loop** against a local jailbreak classifier (HASTE-style)  
4. Explicit **four-way export** (vanilla/adv × harmful/benign) for classifier training  
5. Multi-turn trajectory schema + conversation-level labels  
6. Optional LRM attacker backend (cheap diversity) with diversity regularizer  
7. Transfer evaluation harness (train on model A, test artifacts on model B)

---

## 8. Recommended end-to-end pipeline (defensive product goal)

```text
                    ┌─────────────────────────────┐
   Policy taxonomy  │  Goal bank + harm categories │
   + OWASP/ASI      └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  Compositional           Multi-turn agents            Universal layer
  WildJailbreak+AIC       Crescendo/GOAT/LRM           BoN + AutoDAN/GCG
  (diversity mode)        (trajectory data)            (transfer artifacts)
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                    Ensemble judge + human audit
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         Classifier training set          Policy hardening set
         (HASTE hard-neg loop)            (SFT/DPO + ReFAT/CAT)
                    │                             │
                    ▼                             ▼
              Deploy guardrail              Deploy/update LLM
                    │                             │
                    └──────── continuous RT ──────┘
```

**Operating principles:**

1. **Diversity is a first-class metric** (unique tactics, embedding entropy, taxonomy coverage) — not only ASR.  
2. **Subtle bandit / explore mode** for corpus building; **aggressive mode** for finding residual holes after a patch.  
3. **Always include adversarial-benign** to protect helpfulness.  
4. **Close the loop:** every successful jailbreak becomes (a) classifier hard positive and (b) policy refusal target within days, not quarters.  
5. **Authorize & scope-gate** campaigns; log scope in run manifests (your harness already does this).

---

## 9. Open problems (still open as of July 2026)

1. **Judge–human gap:** high ASR can mean judge gaming, not true policy violation.  
2. **Multi-turn evaluation standardization:** still weaker than single-turn (MLCommons still mechanism-first for single-turn).  
3. **Agentic / tool-using systems:** safety training on chat does not fully transfer to agents.  
4. **Over-refusal vs robustness Pareto:** latent AT and more data both help, but product constraints dominate.  
5. **Dual use & template markets:** commercial jailbreak kits rotate faster than static filters.  
6. **Compute cost:** 10k-trial adaptive campaigns still ~70–120 GPU-hours class; need better sample efficiency.  
7. **Multimodal universals** outpace text-only guard training in many products.

---

## 10. Key references (grouped)

### Foundational universal / auto attacks
- Zou et al., *Universal and Transferable Attacks on Aligned Language Models* (GCG), 2023 — https://llm-attacks.org/  
- Chao et al., PAIR, 2023/24  
- Mehrotra et al., TAP (Tree of Attacks)  
- Liu et al., AutoDAN / AutoDAN-Turbo  
- Hughes et al., *Best-of-N Jailbreaking*, NeurIPS 2025 — arXiv:2412.03556  

### Multi-turn & agentic
- Russinovich, Salem, Eldan, *Crescendo*, USENIX Security 2025 — arXiv:2404.01833  
- Pavlova et al., *GOAT*, 2024/ICML 2025 — arXiv:2410.01606  
- Hagendorff et al., *Large reasoning models are autonomous jailbreak agents*, Nature Communications 2026  

### Lifelong / RL / multi-agent / production-agent autoresearch (2025–2026)
- Liu, Zhou, et al., **Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming LLMs**, arXiv:2501.01830 (ICLR 2026 track) — RL strategy exploration, early termination, progressive rewards  
- Zhou, Wu, Pinto, … Li, **AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**, arXiv:2503.15754 — multi-agent + memory-guided lifelong attacks; +20% HarmBench ASR, −46% compute  
- Mao, Zheng, Wang, **Agent Hacks Agent: Autoresearch for Production-Agent Red-Teaming (AHA)**, arXiv:2607.11698 (Jul 2026) — vulnerability concept graph for Claude Code / Codex-style agents; code related to `henrymao2004/Auto-research-red-teaming*`  
- Xiong et al., CoP (Composition of Principles), NeurIPS 2025  
- Jailbreak-autoresearch evolutionary loops (open research: multi-model scoring, greedy keep/revert)  
- Industry 2026 guides: continuous automated red teaming for agents (tool abuse, memory, CI)  

### Compositional data at scale
- Jiang et al., *WildTeaming at Scale* / WildJailbreak — arXiv:2406.18510; HF `allenai/wildjailbreak`  
- Zymet et al., *Adaptive Instruction Composition*, Apr 2026 — arXiv:2604.21159  

### Hardening & classifiers
- Ge et al., MART — arXiv:2311.07689  
- Yu et al., ReFAT (refusal feature AT), ICLR 2025 — arXiv:2409.20089  
- Continuous AT (CAT) line — arXiv:2405.15589; theory ICLR 2026 arXiv:2604.12817  
- Chen et al., HASTE, LAST-X 2026 (NDSS workshop)  
- NVIDIA NemoGuard JailbreakDetect  

### Multimodal
- UltraBreak, ICLR 2026 — arXiv:2602.01025  

### Benchmarks & surveys
- Mazeika et al., HarmBench  
- Chao et al., JailbreakBench  
- Lin et al., *Against the Achilles’ Heel* (JAIR survey on red teaming generative models)  
- Purpura et al., *Building Safe GenAI Applications… Red Teaming*, 2025 — arXiv:2503.01742  
- MLCommons Jailbreak 0.7 (Feb 2026)  
- SoK: Evaluating Jailbreak Guardrails, 2025 — arXiv:2506.10597  

### Tooling writeups (2026)
- Help Net Security on AIC learning layer (Apr 2026)  
- Industry tool roundups: PyRIT, garak, Promptfoo, DeepTeam, Inspect  

---

## 11. One-page takeaway for this repo

| Priority | Technique | Why for *your* goal |
|----------|-----------|---------------------|
| P0 | WildJailbreak four-way labels + tactic library | Best open prior for classifier + policy data |
| P0 | AIC-style bandit over compositions (you already have bandit hooks) | Model-specific, diverse, high yield |
| P0 | Multi-turn (Crescendo/GOAT/LRM) trajectories | Classifiers that only see single-turn will fail in prod |
| P1 | BoN augmentations | Cheap universal surface-form coverage |
| P1 | HASTE hard-neg loop vs local detector | Directly optimizes jailbreak **classification** |
| P1 | Ensemble judges + small human gold set | Prevents training on judge noise |
| P2 | GCG/AutoDAN universals + transfer tests | Worst-case + transfer reporting |
| P2 | ReFAT/CAT on open-weight targets | Weight-level hardening when you own the model |
| P2 | Continuous CI red team after each fine-tune | Prevents regression |

**Bottom line (July 2026):**  
State of the art for *hardening via data* is no longer “run PAIR until ASR is high.” It is a **closed loop**: compositional + multi-turn + universal generators → carefully labeled four-way corpora → hard-negative mining against the current guard → policy/AT update → re-attack with a transferred or re-trained adaptive composer. Universal jailbreaks matter because they stress **shared failure modes**; jailbreak classifiers matter because they are the deployable, low-latency shield while the policy catches up.

---

*Document generated for the `auto-redteam` project. Defensive research use only.*
