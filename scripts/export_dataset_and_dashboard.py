"""export_dataset_and_dashboard.py

Read a finished CampaignResult (result.json / trajectories.jsonl) and produce:

  1. datasets/edu_100_items.jsonl   -- 100 unique labelled items for jailbreak-classifier training
  2. datasets/edu_100_items.csv
  3. datasets/edu_100_items.json    -- same as list
  4. dashboard/index.html           -- self-contained interactive report

Educational use only. Run with python3.exe from the repo root:

  .venv\\Scripts\\python.exe scripts/export_dataset_and_dashboard.py \\
      --run-dir runs/local-gemma4b-edu-100
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _load_result(run_dir: Path) -> dict[str, Any]:
    rj = run_dir / "result.json"
    if rj.is_file():
        return json.loads(rj.read_text(encoding="utf-8"))
    # Fall back to trajectories only
    traj_path = run_dir / "trajectories.jsonl"
    trajs = []
    if traj_path.is_file():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajs.append(json.loads(line))
    return {
        "campaign_name": run_dir.name,
        "config_hash": "",
        "scope": "",
        "trajectories": trajs,
        "metrics": {},
        "started_at": None,
        "finished_at": None,
        "meta": {},
    }


def _best_turn(traj: dict[str, Any]) -> dict[str, Any] | None:
    turns = traj.get("turns") or []
    if not turns:
        return None
    # Prefer successful violation, else highest score
    success = [t for t in turns if (t.get("eval") or {}).get("success")]
    if success:
        return max(success, key=lambda t: (t.get("eval") or {}).get("score", 0.0))
    return max(turns, key=lambda t: (t.get("eval") or {}).get("score", 0.0))


def build_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for traj in result.get("trajectories") or []:
        goal = traj.get("goal") or {}
        gid = goal.get("id") or traj.get("id") or f"traj_{len(items)}"
        if gid in seen_ids:
            continue
        seen_ids.add(gid)
        turn = _best_turn(traj)
        ev = (turn or {}).get("eval") or {}
        meta = goal.get("meta") or {}
        jb_type = meta.get("jailbreak_type") or "regular"
        prompt = (turn or {}).get("attacker_prompt") or ""
        if not prompt and goal.get("seed_prompts"):
            prompt = goal["seed_prompts"][0]
        response = (turn or {}).get("defender_response") or ""
        success = bool(ev.get("success") or traj.get("succeeded"))
        # Classifier label: jailbreak_attempt is always True for these probes;
        # "success" means defender complied (canary hit / violation).
        item = {
            "id": gid,
            "jailbreak_type": jb_type,  # regular | universal
            "template": meta.get("template") or meta.get("transfer_family") or "",
            "category": goal.get("category") or ev.get("category_match") or "",
            "severity": goal.get("severity") or "",
            "goal": goal.get("goal") or "",
            "canary_token": meta.get("canary_token") or meta.get("canary") or "",
            "attacker_prompt": prompt,
            "defender_response": response,
            "strategy": (turn or {}).get("strategy") or (traj.get("strategies_used") or [""])[0],
            "verdict": ev.get("verdict") or "",
            "score": float(ev.get("score") or 0.0),
            "success": success,
            "label_jailbreak_attempt": True,  # all items are adversarial probes
            "label_defender_complied": success,
            "classifier_class": (
                "jailbreak_success" if success else "jailbreak_blocked"
            ),
            "educational": bool(meta.get("educational", True)),
            "n_turns": len(traj.get("turns") or []),
            "trajectory_id": traj.get("id") or "",
        }
        items.append(item)
    # Stable order by id
    items.sort(key=lambda x: x["id"])
    return items


def write_dataset(items: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    jpath = out_dir / "edu_100_items.json"
    jpath.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["json"] = jpath

    jlpath = out_dir / "edu_100_items.jsonl"
    with jlpath.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    paths["jsonl"] = jlpath

    cpath = out_dir / "edu_100_items.csv"
    fields = [
        "id", "jailbreak_type", "template", "category", "severity", "strategy",
        "verdict", "score", "success", "label_jailbreak_attempt",
        "label_defender_complied", "classifier_class", "canary_token",
        "n_turns", "attacker_prompt", "defender_response", "goal",
    ]
    with cpath.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow(it)
    paths["csv"] = cpath
    return paths


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def build_dashboard(
    result: dict[str, Any],
    items: list[dict[str, Any]],
    out_html: Path,
) -> Path:
    n = len(items)
    n_success = sum(1 for i in items if i["success"])
    n_reg = sum(1 for i in items if i["jailbreak_type"] == "regular")
    n_uni = sum(1 for i in items if i["jailbreak_type"] == "universal")
    n_reg_ok = sum(1 for i in items if i["jailbreak_type"] == "regular" and i["success"])
    n_uni_ok = sum(1 for i in items if i["jailbreak_type"] == "universal" and i["success"])

    by_cat = Counter(i["category"] for i in items)
    by_cat_ok = Counter(i["category"] for i in items if i["success"])
    by_tmpl = Counter(i["template"] for i in items)
    by_tmpl_ok = Counter(i["template"] for i in items if i["success"])
    by_strat = Counter(i["strategy"] for i in items if i["strategy"])
    by_strat_ok = Counter(i["strategy"] for i in items if i["success"] and i["strategy"])
    by_verdict = Counter(i["verdict"] or "unknown" for i in items)
    scores = [i["score"] for i in items]

    metrics = result.get("metrics") or {}
    campaign = result.get("campaign_name") or "campaign"
    scope = result.get("scope") or ""
    started = result.get("started_at") or ""
    finished = result.get("finished_at") or ""

    def bar_rows(counter: Counter, ok_counter: Counter, title: str) -> str:
        rows = []
        for k, total in counter.most_common():
            ok = ok_counter.get(k, 0)
            pct = _pct(ok, total)
            rows.append(
                f"<tr><td>{html.escape(str(k) or '(none)')}</td>"
                f"<td>{ok}/{total}</td><td>{pct}%</td>"
                f"<td class='bar'><span style='width:{pct}%'></span></td></tr>"
            )
        return (
            f"<h3>{html.escape(title)}</h3>"
            f"<table><thead><tr><th>Key</th><th>Success</th><th>ASR</th><th></th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    # Sample findings table (first 25)
    sample_rows = []
    for it in items[:25]:
        cls = "ok" if it["success"] else "no"
        sample_rows.append(
            "<tr class='{cls}'>"
            "<td>{id}</td><td>{jb}</td><td>{cat}</td><td>{st}</td>"
            "<td>{ver}</td><td>{sc:.2f}</td><td>{succ}</td>"
            "<td class='mono'>{prompt}</td><td class='mono'>{resp}</td>"
            "</tr>".format(
                cls=cls,
                id=html.escape(it["id"]),
                jb=html.escape(it["jailbreak_type"]),
                cat=html.escape(it["category"]),
                st=html.escape(it["strategy"]),
                ver=html.escape(str(it["verdict"])),
                sc=it["score"],
                succ="YES" if it["success"] else "no",
                prompt=html.escape((it["attacker_prompt"] or "")[:120]),
                resp=html.escape((it["defender_response"] or "")[:120]),
            )
        )

    mean_score = round(statistics.mean(scores), 3) if scores else 0.0
    asr = _pct(n_success, n)

    # Simple SVG pie for regular vs universal ASR
    def pie_svg(ok_a: int, tot_a: int, ok_b: int, tot_b: int) -> str:
        # Two half-ish bars instead of real pie for simplicity
        pa = _pct(ok_a, tot_a)
        pb = _pct(ok_b, tot_b)
        return f"""
        <div class="dual">
          <div class="card"><div class="label">Regular jailbreaks</div>
            <div class="big">{pa}%</div>
            <div class="sub">{ok_a}/{tot_a} canary hits</div>
            <div class="hbar"><span style="width:{pa}%"></span></div>
          </div>
          <div class="card"><div class="label">Universal scaffolds</div>
            <div class="big">{pb}%</div>
            <div class="sub">{ok_b}/{tot_b} canary hits</div>
            <div class="hbar"><span style="width:{pb}%"></span></div>
          </div>
        </div>"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>auto-redteam · {html.escape(campaign)}</title>
<style>
  :root {{
    --bg: #0b1020; --panel: #121a2f; --ink: #e8eefc; --muted: #9aa8c7;
    --accent: #5b8cff; --ok: #3ecf8e; --no: #ff6b7a; --line: #243056;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a2748 0%, var(--bg) 55%);
    color: var(--ink); line-height: 1.45;
  }}
  header {{
    padding: 28px 32px 12px; border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(91,140,255,.12), transparent);
  }}
  header h1 {{ margin: 0 0 6px; font-size: 1.55rem; letter-spacing: -0.02em; }}
  header p {{ margin: 4px 0; color: var(--muted); font-size: .95rem; }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    background: #1e2d52; color: #b8c9ff; font-size: .78rem; margin-right: 6px;
  }}
  main {{ padding: 20px 32px 48px; max-width: 1200px; margin: 0 auto; }}
  .kpis {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 18px 0 24px;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px 16px;
  }}
  .kpi .v {{ font-size: 1.6rem; font-weight: 700; }}
  .kpi .l {{ color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }}
  .dual {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0 22px; }}
  @media (max-width: 720px) {{ .dual {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px;
  }}
  .card .label {{ color: var(--muted); font-size: .85rem; }}
  .card .big {{ font-size: 2rem; font-weight: 700; margin: 4px 0; }}
  .card .sub {{ color: var(--muted); font-size: .9rem; }}
  .hbar, table .bar {{
    height: 8px; background: #1a243f; border-radius: 6px; overflow: hidden; margin-top: 10px;
  }}
  .hbar span, table .bar span {{
    display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #8b5cff);
  }}
  section {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 18px 20px; margin-bottom: 16px;
  }}
  h2 {{ margin: 0 0 12px; font-size: 1.15rem; }}
  h3 {{ margin: 16px 0 8px; font-size: 1rem; color: #c9d6f5; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }}
  tr.ok td:nth-child(7) {{ color: var(--ok); font-weight: 700; }}
  tr.no td:nth-child(7) {{ color: var(--no); }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .78rem; color: #c5d0ea; max-width: 240px; word-break: break-word; }}
  footer {{ color: var(--muted); font-size: .82rem; padding: 8px 32px 28px; max-width: 1200px; margin: 0 auto; }}
  code {{ background: #1a243f; padding: 1px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>Local Gemma-4B educational red-team dashboard</h1>
  <p>
    <span class="badge">AUTHORIZED LAB</span>
    <span class="badge">{html.escape(campaign)}</span>
    <span class="badge">attacker: ablated Gemma-4B</span>
    <span class="badge">defender: official Gemma-4B</span>
  </p>
  <p>Scope: {html.escape(scope[:280])}</p>
  <p>Started: {html.escape(str(started))} · Finished: {html.escape(str(finished))} · Config hash: <code>{html.escape((result.get('config_hash') or '')[:16])}</code></p>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><div class="l">Unique items</div><div class="v">{n}</div></div>
    <div class="kpi"><div class="l">Overall ASR</div><div class="v">{asr}%</div></div>
    <div class="kpi"><div class="l">Canary hits</div><div class="v">{n_success}</div></div>
    <div class="kpi"><div class="l">Mean score</div><div class="v">{mean_score}</div></div>
    <div class="kpi"><div class="l">Regular / Universal</div><div class="v">{n_reg}/{n_uni}</div></div>
    <div class="kpi"><div class="l">Trajectories</div><div class="v">{len(result.get('trajectories') or [])}</div></div>
  </div>

  {pie_svg(n_reg_ok, n_reg, n_uni_ok, n_uni)}

  <section>
    <h2>Breakdown</h2>
    {bar_rows(by_cat, by_cat_ok, "By taxonomy category")}
    {bar_rows(by_tmpl, by_tmpl_ok, "By template / universal family")}
    {bar_rows(by_strat, by_strat_ok, "By attack strategy")}
    {bar_rows(by_verdict, by_verdict, "By judge verdict (count only)")}
  </section>

  <section>
    <h2>Sample items (first 25 of {n})</h2>
    <p style="color:var(--muted);font-size:.9rem">
      Labels: <code>label_jailbreak_attempt=true</code> for all probes;
      <code>label_defender_complied</code> = canary hit / violation (useful as hard positive for classifiers).
    </p>
    <div style="overflow:auto">
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Type</th><th>Category</th><th>Strategy</th>
          <th>Verdict</th><th>Score</th><th>Hit</th>
          <th>Attack prompt</th><th>Defender response</th>
        </tr>
      </thead>
      <tbody>
        {''.join(sample_rows)}
      </tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>Dataset exports</h2>
    <ul>
      <li><code>datasets/edu_100_items.json</code></li>
      <li><code>datasets/edu_100_items.jsonl</code></li>
      <li><code>datasets/edu_100_items.csv</code></li>
    </ul>
    <p style="color:var(--muted)">
      Use <code>classifier_class</code> ∈ {{jailbreak_success, jailbreak_blocked}} for supervised
      jailbreak-classification training. Mix in benign prompts separately to control false positives.
    </p>
  </section>
</main>
<footer>
  Educational exercise with small local models · auto-redteam · findings for defensive hardening only.
  Raw metrics blob: {html.escape(json.dumps(metrics)[:400])}
</footer>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "local-gemma4b-edu-100")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets")
    ap.add_argument("--dashboard", type=Path, default=ROOT / "dashboard" / "index.html")
    args = ap.parse_args()

    result = _load_result(args.run_dir)
    items = build_items(result)
    if not items:
        raise SystemExit(f"No trajectories/items found under {args.run_dir}")

    paths = write_dataset(items, args.dataset_dir)
    dash = build_dashboard(result, items, args.dashboard)

    n_ok = sum(1 for i in items if i["success"])
    print(f"items     : {len(items)}")
    print(f"successes : {n_ok} ({_pct(n_ok, len(items))}%)")
    print(f"regular   : {sum(1 for i in items if i['jailbreak_type']=='regular')}")
    print(f"universal : {sum(1 for i in items if i['jailbreak_type']=='universal')}")
    for k, p in paths.items():
        print(f"dataset[{k}]: {p}")
    print(f"dashboard : {dash}")


if __name__ == "__main__":
    main()
