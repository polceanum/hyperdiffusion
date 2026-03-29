#!/usr/bin/env python3
import json
from pathlib import Path

RUNS = {
    "Classification": Path("runs/classification_v2/summary.json"),
    "Regression": Path("runs/regression_v2/summary.json"),
    "Bandit": Path("runs/bandit_v2/summary.json"),
    "Control": Path("runs/control_v2/summary.json"),
}


def get_main_metrics(task, data):
    overall = data["overall"]
    if task == "Classification":
        return {
            "metric": "acc",
            "encoder": overall["encoder_acc"],
            "diffusion": overall["diffusion_acc"],
            "baseline": overall["baseline_acc"],
        }
    return {
        "metric": "r2",
        "encoder": overall["encoder_r2"],
        "diffusion": overall["diffusion_r2"],
        "baseline": overall["baseline_r2"],
    }


def _extract_metric_triplet(block):
    if not block:
        return None
    if "encoder_acc" in block:
        return {
            "metric": "acc",
            "encoder": block["encoder_acc"],
            "diffusion": block["diffusion_acc"],
            "baseline": block["baseline_acc"],
        }
    if "encoder_r2" in block:
        return {
            "metric": "r2",
            "encoder": block["encoder_r2"],
            "diffusion": block["diffusion_r2"],
            "baseline": block["baseline_r2"],
        }
    return None


def get_ood_metrics(task, data):
    gen = data.get("generalization", {}).get("eval_summary", {}).get("overall")
    return _extract_metric_triplet(gen)


def main():
    print("RESULT ARTIFACT AUDIT")
    print("=" * 72)
    for task, path in RUNS.items():
        data = json.loads(path.read_text())
        m = get_main_metrics(task, data)
        protocol = data.get("protocol", {})
        cfg = data.get("config", {})
        fam = protocol.get("train_families") or cfg.get("families") or []
        eval_fam = protocol.get("eval_families") or cfg.get("eval_families") or []
        print(f"\n{task}")
        print(f"  source: {path}")
        print(f"  split config: train={len(fam)} eval={len(eval_fam)}")
        print(f"  main metrics: fw_det={m['encoder']:.4f} fw_diff={m['diffusion']:.4f} baseline={m['baseline']:.4f}")
        ood = get_ood_metrics(task, data)
        if ood is None:
            print("  ood metrics: not present in this summary")
        else:
            print(f"  ood metrics:  fw_det={ood['encoder']:.4f} fw_diff={ood['diffusion']:.4f} baseline={ood['baseline']:.4f}")


if __name__ == "__main__":
    main()
