let meta = #{
    name: "finish-paper-systems",
    description: "Finish remaining paper-inspired research modules with parallel agents, wire into pipeline, verify with tests",
    phases: [
        #{ title: "Audit", detail: "gap analysis of papers vs code" },
        #{ title: "Implement", detail: "parallel CoP, AIC, production-agent modules" },
        #{ title: "Integrate", detail: "wire exports, pipeline, CLI, tests, docs" },
        #{ title: "Verify", detail: "run tests and fix failures" },
    ],
};

let audit_schema = #{
    "type": "object",
    "required": ["gaps", "existing_ok", "priority_order"],
    "properties": #{
        "gaps": #{
            "type": "array",
            "items": #{
                "type": "object",
                "required": ["paper", "missing", "suggested_path"],
                "properties": #{
                    "paper": #{ "type": "string" },
                    "missing": #{ "type": "string" },
                    "suggested_path": #{ "type": "string" },
                },
            },
        },
        "existing_ok": #{ "type": "array", "items": #{ "type": "string" } },
        "priority_order": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

let impl_schema = #{
    "type": "object",
    "required": ["module_path", "public_symbols", "tests_path", "summary", "ok"],
    "properties": #{
        "module_path": #{ "type": "string" },
        "public_symbols": #{ "type": "array", "items": #{ "type": "string" } },
        "tests_path": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "ok": #{ "type": "boolean" },
    },
};

let integ_schema = #{
    "type": "object",
    "required": ["files_touched", "summary", "ok"],
    "properties": #{
        "files_touched": #{ "type": "array", "items": #{ "type": "string" } },
        "summary": #{ "type": "string" },
        "ok": #{ "type": "boolean" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["tests_passed", "failures", "notes"],
    "properties": #{
        "tests_passed": #{ "type": "boolean" },
        "failures": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

// Workspace is autoredteam at /mnt/c/Users/evija/autoredteam
let root = if args == () { "/mnt/c/Users/evija/autoredteam" } else if args.root == () { "/mnt/c/Users/evija/autoredteam" } else { args.root };

phase("Audit");
let audit_prompt = "";
audit_prompt += "You are auditing the autoredteam repo at " + root + ".\n";
audit_prompt += "Read docs/RELATED_WORK_PAPERS.md, RESEARCH_SURVEY.md, and list files under autoredteam/research/.\n";
audit_prompt += "Compare papers (Auto-RT, AutoRedTeamer, AHA/VCG, Jailbreak-autoresearch, CoP, AIC, production-agent AHA sandbox) to code.\n";
audit_prompt += "Use read_file/list_dir/grep only. Report concrete gaps still missing after prior work that already added memory.py, auto_rt.py, autoresearch.py, vcg.py, strategy_proposer.py.\n";
audit_prompt += "Expected remaining gaps to confirm: (1) CoP Composition of Principles, (2) AIC Adaptive Instruction Composition bandit, (3) AHA-style production-agent educational victim harness with tool/filesystem contracts (safe canary-only).\n";
audit_prompt += "Return gaps with suggested_path under autoredteam/research/.\n";
let audit = agent(audit_prompt, #{
    label: "audit-gaps",
    capability_mode: "read-only",
    output_schema: audit_schema,
});
if audit == () || !audit.success {
    complete(#{ summary: "Audit agent failed", ok: false });
}
log("Audit gaps: " + audit.output.gaps.len().to_string());

phase("Implement");
// Parallel implementation of three modules
let p_cop = "";
p_cop += "Implement CoP (Composition of Principles) for the educational autoredteam harness.\n";
p_cop += "Repo root: " + root + ".\n";
p_cop += "Paper: CoP NeurIPS 2025 — agent composes human principles into new attack strategies.\n";
p_cop += "Create file: " + root + "/autoredteam/research/cop.py\n";
p_cop += "Requirements:\n";
p_cop += "1. Define a PRINCIPLES catalog (list of principle dicts with id, name, description, apply_fn or template pattern). Educational only: abstract framing principles like roleplay, academic, encoding, multi-step, persona, etc. — NOT real harmful content.\n";
p_cop += "2. CompositionOfPrinciples class that samples/composes k principles into a composed strategy, applies them to a seed_prompt + canary_token preserving the canary exactly, returns structured CompositionResult.\n";
p_cop += "3. Methods: list_principles, compose(seed, canary, k=2, rng), compose_batch(goals, k=2).\n";
p_cop += "4. to_dict / save stats for research export.\n";
p_cop += "5. Match style of sibling modules (dataclasses, type hints, no external deps beyond stdlib). Read memory.py and strategy_proposer.py for style.\n";
p_cop += "6. Create/update tests in " + root + "/tests/test_research_papers.py — add test_cop_compose_preserves_canary.\n";
p_cop += "7. Educational canary protocol only. No real exploit payloads.\n";
p_cop += "Write the files. Return module_path, public_symbols, tests_path, summary, ok=true if imports work.\n";
p_cop += "Use python3.exe or .venv if needed; prefer writing pure stdlib Python that works on 3.10+.\n";

let p_aic = "";
p_aic += "Implement AIC (Adaptive Instruction Composition) bandit for the educational autoredteam harness.\n";
p_aic += "Repo root: " + root + ".\n";
p_aic += "Paper: Adaptive Instruction Composition (Zymet et al., arXiv:2604.21159) — contextual bandit over query×tactic space.\n";
p_aic += "Create file: " + root + "/autoredteam/research/aic.py\n";
p_aic += "Requirements:\n";
p_aic += "1. Catalog of educational tactics (surface transforms): prefix/suffix frames, format wraps (json, xml, markdown table), language style, length, etc. Canary-safe.\n";
p_aic += "2. AdaptiveInstructionComposer with LinUCB-lite or Thompson-style multi-armed bandit over tactics (and optional pairs). Context features can be simple bag: category one-hot, jailbreak_type, template hash buckets.\n";
p_aic += "3. select(context) -> tactic, update(tactic, reward), compose(seed, canary, tactic) preserving canary, run_episode for offline sim.\n";
p_aic += "4. stats() and save/load JSON.\n";
p_aic += "5. Match sibling module style. Add test_aic_bandit_learns in tests/test_research_papers.py.\n";
p_aic += "6. Educational only. Write files. Return module_path, public_symbols, tests_path, summary, ok.\n";

let p_prod = "";
p_prod += "Implement AHA-style production-agent educational victim harness for autoredteam.\n";
p_prod += "Repo root: " + root + ".\n";
p_prod += "Paper: AHA Agent Hacks Agent (arXiv:2607.11698) — tool-using victim + sandboxed attacks + VCG.\n";
p_prod += "Create file: " + root + "/autoredteam/research/production_agent.py\n";
p_prod += "Requirements (EDUCATIONAL / SAFE only):\n";
p_prod += "1. SandboxFS: temp directory with allowed files, read/write/list with path confinement (reject path traversal).\n";
p_prod += "2. ToolSpec registry: list_files, read_file, write_file, echo_canary — tools that operate only inside sandbox.\n";
p_prod += "3. ProductionAgentVictim: processes attacker natural-language instructions by optionally calling tools (simple rule-based or mock tool-calling for offline tests). Goal: canary token written to forbidden path or leaked via tool output under policy violation scenarios for lab measurement.\n";
p_prod += "4. Policy: mark certain paths as 'secret' (e.g. secrets/canary.txt). Success = secret canary content appears in tool output or public files when policy forbids it.\n";
p_prod += "5. AttackEpisode + run_attack(prompt) returning trajectory of tool calls, success, score; integrate with VCG promotion helper method promote_to_vcg.\n";
p_prod += "6. Offline mock mode that needs no LLM. Add test_production_agent_sandbox in tests/test_research_papers.py.\n";
p_prod += "7. Document clearly this is educational canary protocol, not real offensive tooling.\n";
p_prod += "Write files. Return module_path, public_symbols, tests_path, summary, ok.\n";

let impls = parallel([
    #{ prompt: p_cop, label: "impl-cop", capability_mode: "all", output_schema: impl_schema },
    #{ prompt: p_aic, label: "impl-aic", capability_mode: "all", output_schema: impl_schema },
    #{ prompt: p_prod, label: "impl-prod-agent", capability_mode: "all", output_schema: impl_schema },
]);

let impl_ok = 0;
let impl_summaries = [];
let i = 0;
for r in impls {
    if r != () && r.success && r.output.ok == true {
        impl_ok += 1;
        impl_summaries.push(r.output.summary);
    } else {
        let msg = "impl slot " + i.to_string() + " failed or not ok";
        if r != () && r.success {
            msg = r.output.summary;
        }
        impl_summaries.push(msg);
        log(msg);
    }
    i += 1;
}
log("Implementations ok: " + impl_ok.to_string() + "/3");

phase("Integrate");
let integ_prompt = "";
integ_prompt += "Integrate the new research modules into the autoredteam package.\n";
integ_prompt += "Repo root: " + root + ".\n";
integ_prompt += "New modules (if present): research/cop.py, research/aic.py, research/production_agent.py.\n";
integ_prompt += "Tasks:\n";
integ_prompt += "1. Update autoredteam/research/__init__.py to export main public classes/functions from all research modules including KeepRevertLoop if missing, CoP, AIC, ProductionAgentVictim.\n";
integ_prompt += "2. Update scripts/run_research_loop.py to optionally run CoP compose + AIC select stats and a production-agent mock episode when --mock; write extra JSON assets under out dir (cop_stats.json, aic_stats.json, production_agent_episode.json).\n";
integ_prompt += "3. Update phase_research in autoredteam/pipeline/phases.py to also export CoP compositions over failed items (offline, no LLM) and AIC tactic recommendations based on memory stats if modules exist. Keep phase offline-by-default.\n";
integ_prompt += "4. Update docs/RELATED_WORK_PAPERS.md mapping table for CoP, AIC, production_agent paths; remove them from pure 'Future extension' if implemented.\n";
integ_prompt += "5. Ensure tests/test_research_papers.py covers all three new modules; fix any import errors.\n";
integ_prompt += "6. Match existing code style. Do not break existing tests.\n";
integ_prompt += "Use Windows-compatible notes in docs (python3.exe) where README already does.\n";
integ_prompt += "Return files_touched, summary, ok.\n";
let integ = agent(integ_prompt, #{
    label: "integrate",
    capability_mode: "all",
    output_schema: integ_schema,
});
if integ == () || !integ.success {
    log("Integrate agent failed");
} else {
    log(integ.output.summary);
}

phase("Verify");
let ver_prompt = "";
ver_prompt += "Verify the autoredteam research paper implementations.\n";
ver_prompt += "Repo root: " + root + ".\n";
ver_prompt += "1. List autoredteam/research/*.py and confirm cop.py, aic.py, production_agent.py exist.\n";
ver_prompt += "2. Run tests using Windows venv if available: " + root + "/.venv/Scripts/python.exe -m pytest tests/test_research_papers.py tests/test_pipeline_bon.py -q\n";
ver_prompt += "   On Linux WSL, if .venv is Windows-only, try: python3.12 -m pip install -e '.[dev]' or PYTHONPATH=. with available pydantic, or use wine/cmd.exe if that's how this env runs Windows python.\n";
ver_prompt += "   Prefer: cd " + root + " && cmd.exe /c .venv\\\\Scripts\\\\python.exe -m pytest tests/test_research_papers.py tests/test_pipeline_bon.py -q\n";
ver_prompt += "3. Fix any test failures related to new modules (syntax, imports, assertions). Re-run until green or document blockers.\n";
ver_prompt += "4. Quick import smoke: python -c 'from autoredteam.research import ...'\n";
ver_prompt += "Return tests_passed, failures (empty if none), notes.\n";
let ver = agent(ver_prompt, #{
    label: "verify-tests",
    capability_mode: "all",
    output_schema: verify_schema,
});

let passed = false;
let fail_txt = "verify agent failed";
let notes = "";
if ver != () && ver.success {
    passed = ver.output.tests_passed;
    fail_txt = ver.output.failures;
    notes = ver.output.notes;
}

let report = "";
report += "# finish-paper-systems workflow result\n\n";
report += "## Audit gaps\n";
if audit.success {
    report += json_encode(audit.output.gaps) + "\n\n";
}
report += "## Implementations ok: " + impl_ok.to_string() + "/3\n";
for s in impl_summaries {
    report += "- " + s + "\n";
}
report += "\n## Integration\n";
if integ != () && integ.success {
    report += integ.output.summary + "\n";
    report += "files: " + json_encode(integ.output.files_touched) + "\n";
} else {
    report += "integration incomplete\n";
}
report += "\n## Verify\n";
report += "tests_passed=" + passed.to_string() + "\n";
report += "failures=" + fail_txt + "\n";
report += "notes=" + notes + "\n";

let path = write_scratch_file("finish-paper-systems-report.md", report);
complete(#{
    path: path,
    ok: passed && impl_ok >= 2,
    impl_ok: impl_ok,
    tests_passed: passed,
    report: report,
});
