"""Zero-shot LLM routers: hosted Claude (via headless Claude Code) and local Qwen (via Ollama).

Both get the same label definitions and return a label plus a verbalized
confidence, which becomes the top-label probability for calibration metrics.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time

import httpx
import numpy as np

from bioroute.data import ROUTE_DEFINITIONS, ROUTES

DEFS = "\n".join(f"- {k}: {v}" for k, v in ROUTE_DEFINITIONS.items())
SYSTEM = (
    "You route requests typed into a biotech company's internal assistant. "
    "Pick exactly one route per request and give your confidence (0 to 1) that it is correct.\n"
    f"Routes:\n{DEFS}\n"
    "If a request fits human_review and another route, choose human_review."
)


def _probs(label: str, conf: float) -> np.ndarray:
    """Spread the remaining mass evenly over the other routes."""
    k = len(ROUTES)
    conf = float(min(max(conf, 1.0 / k), 1.0))
    p = np.full(k, (1.0 - conf) / (k - 1))
    p[ROUTES.index(label) if label in ROUTES else ROUTES.index("out_of_scope")] = conf
    return p


def _claude_bin() -> str:
    exe = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if exe:
        return exe
    hits = sorted(glob.glob(os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
    return hits[-1]


def _claude_call(texts: list[str], model: str) -> tuple[list[dict], dict]:
    schema = {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "route": {"type": "string", "enum": ROUTES},
            "confidence": {"type": "number"}}, "required": ["i", "route", "confidence"]}}},
        "required": ["results"],
    }
    prompt = "Route each request:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
    cmd = [_claude_bin(), "-p", prompt, "--output-format", "json", "--model", model, "--tools", "",
           "--system-prompt", SYSTEM, "--json-schema", json.dumps(schema), "--no-session-persistence",
           "--strict-mcp-config", "--setting-sources", "", "--effort", "low"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/tmp")
    wall = time.perf_counter() - start
    ev = json.loads(proc.stdout)
    out = ev.get("structured_output") or json.loads(ev.get("result") or "{}")
    return out.get("results", []), {"cost": ev.get("total_cost_usd") or 0.0, "wall_s": wall, "usage": ev.get("usage", {})}


def claude_classify(texts: list[str], model: str = "claude-haiku-5-5", batch: int = 15) -> dict:
    probs = np.zeros((len(texts), len(ROUTES)))
    probs[:] = 1.0 / len(ROUTES)
    cost = 0.0
    for s in range(0, len(texts), batch):
        chunk = texts[s : s + batch]
        results, meta = _claude_call(chunk, model)
        cost += meta["cost"]
        for r in results:
            if 0 <= r["i"] < len(chunk):
                probs[s + r["i"]] = _probs(r["route"], r["confidence"])
    # Per-request latency: measured separately on single-request calls.
    lat = []
    for t in texts[:12]:
        _, meta = _claude_call([t], model)
        lat.append(meta["wall_s"] * 1000)
    return {"probs": probs, "cost_per_1k_usd": 1000 * cost / len(texts), "latency_ms_p50": float(np.median(lat))}


def ollama_classify(texts: list[str], model: str = "qwen2.5:7b") -> dict:
    probs = np.zeros((len(texts), len(ROUTES)))
    lat = []
    for i, t in enumerate(texts):
        prompt = f'{SYSTEM}\n\nRequest: {t}\nReturn only JSON: {{"route": "<one route>", "confidence": <0-1>}}'
        start = time.perf_counter()
        r = httpx.post("http://localhost:11434/api/generate", json={
            "model": model, "prompt": prompt, "format": "json", "stream": False,
            "options": {"temperature": 0}}, timeout=120)
        lat.append((time.perf_counter() - start) * 1000)
        try:
            out = json.loads(r.json()["response"])
            probs[i] = _probs(str(out.get("route", "")), float(out.get("confidence", 0.5)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            probs[i] = 1.0 / len(ROUTES)
    return {"probs": probs, "cost_per_1k_usd": 0.0, "latency_ms_p50": float(np.median(lat))}
