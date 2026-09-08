#!/usr/bin/env python3
"""Probe Bedrock Converse access for the shortlist (issue #7).

Does not send prompts, images, or microphone audio — only a one-word ping.
Writes evals/results/bedrock_probe.json. Never prints credentials.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

SHORTLIST = [
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def main() -> int:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("boto3 missing")
        return 2

    session = boto3.Session()
    creds = session.get_credentials()
    regions = []
    if session.region_name:
        regions.append(session.region_name)
    for extra in ("us-west-2", "us-east-1"):
        if extra not in regions:
            regions.append(extra)

    rows = []
    for region in regions:
        client = boto3.client("bedrock-runtime", region_name=region)
        for mid in SHORTLIST:
            try:
                client.converse(
                    modelId=mid,
                    messages=[{"role": "user", "content": [{"text": "ping"}]}],
                    inferenceConfig={"maxTokens": 4, "temperature": 0},
                )
                rows.append({"region": region, "model_id": mid, "status": "ok"})
                print(f"OK  {region} {mid}")
            except ClientError as e:
                err = e.response.get("Error", {})
                rows.append({
                    "region": region,
                    "model_id": mid,
                    "status": err.get("Code", "ClientError"),
                    "message": (err.get("Message") or "")[:240],
                })
                print(f"NO  {region} {mid} {err.get('Code')}")
            except Exception as e:
                rows.append({
                    "region": region, "model_id": mid,
                    "status": type(e).__name__, "message": str(e)[:240],
                })
                print(f"ERR {region} {mid} {type(e).__name__}")

    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "credentials_present": creds is not None,
        "credential_method": getattr(creds, "method", None) if creds else None,
        "rows": rows,
        "any_ok": any(r["status"] == "ok" for r in rows),
        "note": "AccessDenied means Mode B cannot score this identity. "
                "Mitra does not silently switch models. "
                "Use inference-profile IDs (us.* / global.*); bare foundation-model "
                "IDs often return ValidationException. Claude Sonnet 4 (20250514) is "
                "legacy; prefer us.anthropic.claude-sonnet-4-6.",
    }
    path = _ROOT / "evals" / "results" / "bedrock_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0 if out["any_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
