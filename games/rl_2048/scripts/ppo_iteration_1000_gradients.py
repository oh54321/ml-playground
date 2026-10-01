"""Run the four fixed-bank gradient decompositions for iteration 1000."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from games.rl_2048.scripts.audit_ppo_update import sha256, write_json
from games.rl_2048.scripts import ppo_broad_gradient_audit as broad


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / "artifacts/ppo_2048/ppo_longer_run_post_refactor_iter1000_audit"


def run(audit: Path):
    audit = audit.resolve()
    native = audit / "native"
    plan = json.loads((audit / "locked_plan.json").read_text())
    manifest = json.loads((native / "selection.json").read_text())
    capture = native / "capture.pt"
    if sha256(capture) != plan["capture_sha256"]:
        raise ValueError("capture hash mismatch")
    # The reusable fixed-bank implementation accepts existing/new cohorts. Here
    # all 96 states are new to this checkpoint, so provide a derived schema while
    # preserving the original immutable lock separately.
    compat = dict(plan)
    compat["existing_boards"] = []
    compat["new_boards"] = plan["boards"]
    write_json(audit / "gradient_compat_plan.json", compat)
    write_json(audit / "lockedplan.json", compat)
    target_capture = audit / "capture.pt"
    if target_capture.exists():
        if sha256(target_capture) != plan["capture_sha256"]:
            raise ValueError("derived capture path contains a different checkpoint capture")
    else:
        shutil.copy2(capture, target_capture)
    # The imported decomposition routine uses the saved native selection to
    # rebuild model/config, and the real 96 training pools under this audit root.
    broad.SOURCE = native
    broad.OLD = ROOT / "artifacts/ppo_2048/ppo_balanced_sampling_iter100"
    broad.OUT = audit
    broad._gradient_report(SimpleNamespace(folder=audit))
    result = json.loads((audit / "gradient_decomposition.json").read_text())
    if result["aggregate"]["n"] != 96:
        raise RuntimeError("gradient decomposition did not cover all 96 locked states")
    result["protocol_sha256"] = sha256(audit / "prebranch_protocol_v1.json")
    result["locked_plan_sha256"] = plan["lock_sha256"]
    write_json(audit / "gradient_decomposition.json", result)
    print(f"Wrote 96-state gradient decomposition under {audit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    run(args.audit)


if __name__ == "__main__":
    main()
