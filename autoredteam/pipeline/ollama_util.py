"""Ollama HTTP helpers shared by pipeline phases (Windows python3.exe friendly)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HOST = "http://127.0.0.1:11434"
DEFENDER_MODEL = "gemma4:e4b"
ATTACKER_CANDIDATES = [
    "huihui_ai/gemma-4-abliterated:e4b",
    "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b",
    "huihui_ai/gemma-4-abliterated:latest",
]


def tags(host: str = DEFAULT_HOST) -> list[str]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models") or []]


def model_present(name: str, tag_list: list[str]) -> bool:
    if name in tag_list:
        return True
    base = name.split(":")[0]
    return any(t == name or t.startswith(base + ":") or t == base for t in tag_list)


def find_ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(cand) if cand.is_file() else None


def pull(model: str, host: str = DEFAULT_HOST) -> None:
    exe = find_ollama_exe()
    if not exe:
        raise RuntimeError("ollama.exe not found on PATH or LOCALAPPDATA")
    print(f"[ollama] pull {model}")
    r = subprocess.run([exe, "pull", model], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ollama pull failed for {model} (exit {r.returncode})")


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 320,
    keep_alive: str = "30m",
    host: str = DEFAULT_HOST,
    timeout: float = 300,
    retries: int = 3,
) -> str:
    import time
    import urllib.error

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return ((data.get("message") or {}).get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def unload(model: str, host: str = DEFAULT_HOST) -> None:
    try:
        body = {"model": model, "keep_alive": 0, "prompt": ""}
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=60).read()
        print(f"[vram] unloaded {model}")
    except Exception as exc:  # noqa: BLE001
        print(f"[vram] unload {model}: {exc}")


def ensure_models(
    *,
    skip_pull: bool = False,
    host: str = DEFAULT_HOST,
    defender: str = DEFENDER_MODEL,
    attacker_candidates: list[str] | None = None,
) -> tuple[str, str]:
    try:
        tag_list = tags(host)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ollama not reachable at {host}: {exc}") from exc
    print(f"[ollama] present: {tag_list}")

    if not model_present(defender, tag_list):
        if skip_pull:
            raise RuntimeError(f"missing defender model {defender}")
        pull(defender)
        tag_list = tags(host)

    cands = attacker_candidates or ATTACKER_CANDIDATES
    attacker = next((c for c in cands if model_present(c, tag_list)), None)
    if attacker is None:
        if skip_pull:
            raise RuntimeError(f"missing attacker; tried {cands}")
        for c in cands:
            try:
                pull(c)
                attacker = c
                break
            except RuntimeError as e:
                print(f"[ollama] {e}")
        if attacker is None:
            raise RuntimeError("no attacker model available")

    print(f"[models] attacker={attacker}")
    print(f"[models] defender={defender}")
    return attacker, defender
