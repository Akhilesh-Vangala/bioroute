"""Generate training queries with a local open-weight model (Qwen 2.5 7B via Ollama).

Training data comes from a different model family than the hosted classifier
under test (Claude), so the hosted baseline is not graded on its own phrasing.
The test set (data/test.jsonl) is written separately and never generated here.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import httpx

ROUTES = {
    "drug_label": "questions about what an FDA drug label says: indications, boxed warnings, warnings and precautions, contraindications, adverse reactions, dosing, use in pregnancy or specific populations",
    "clinical_trials": "questions about clinical trials: which trials exist, phases, recruiting status, sponsors, endpoints, enrollment, trial locations",
    "coverage": "questions about insurance coverage and reimbursement: Medicare national or local coverage determinations, prior authorization, billing codes, payer policies, patient access",
    "human_review": "requests that must go to a human compliance reviewer: writing promotional or marketing copy, claiming one drug is superior to another, promoting off-label uses, treatment advice for a specific patient, or requests containing patient identifiers",
    "out_of_scope": "requests unrelated to drug evidence: IT help, HR policies, travel booking, meeting scheduling, general coding help, small talk, company cafeteria",
}
PERSONAS = ["medical science liaison", "brand marketing manager", "market access analyst",
            "government affairs lead", "medical information specialist", "field sales representative",
            "health economics researcher", "new employee"]
DRUGS = ["Tecentriq", "Ocrevus", "Herceptin", "Kadcyla", "Avastin", "Polivy", "Columvi", "Hemlibra",
         "Xolair", "Vabysmo", "Lunsumio", "Itovebi", "Keytruda", "Opdivo", "Imfinzi", "giredestrant",
         "glofitamab", "inavolisib", "atezolizumab", "trastuzumab"]
OLLAMA = "http://localhost:11434/api/generate"


def ask(route: str, desc: str, rng: random.Random) -> list[str]:
    persona = rng.choice(PERSONAS)
    drugs = ", ".join(rng.sample(DRUGS, 4))
    prompt = (
        f"You write realistic messages that employees at a biotech company type into an internal assistant.\n"
        f"Write 25 different messages of this type: {desc}.\n"
        f"Write them as a {persona} would. Mention drugs such as {drugs} where natural.\n"
        "Vary length (4 to 40 words), tone, and phrasing; include some typos and some very short ones.\n"
        'Return only JSON: {"messages": ["...", "..."]}'
    )
    r = httpx.post(OLLAMA, json={"model": "qwen2.5:7b", "prompt": prompt, "format": "json", "stream": False,
                                 "options": {"temperature": 0.9, "seed": rng.randint(0, 10**6)}}, timeout=300)
    try:
        msgs = json.loads(r.json()["response"]).get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return []
    return [m.strip() for m in msgs if isinstance(m, str) and 3 <= len(m.split()) <= 60]


def main(calls_per_route: int = 5, out: str = "data/train.jsonl") -> None:
    rng = random.Random(42)
    seen: set[str] = set()
    rows = []
    for route, desc in ROUTES.items():
        for _ in range(calls_per_route):
            for m in ask(route, desc, rng):
                key = re.sub(r"\W+", " ", m.lower()).strip()
                if key not in seen:
                    seen.add(key)
                    rows.append({"text": m, "label": route})
            print(route, len([r for r in rows if r["label"] == route]), flush=True)
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", len(rows), out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
