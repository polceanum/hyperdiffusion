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


def get_ood_metrics(task, data):
    gen = data.get("generalization", {}).get("eval_summary", {}).get("overall")
    if not gen:
        return None
    return {
        "encoder": gen["encoder_r2"],
        "diffusion": gen["diffusion_r2"],
        "baseline": gen["baseline_r2"],
    }


def main():
    print("RESULT ARTIFACT AUDIT")
    print("=" * 72)
    for task, path in RUNS.items():
        data = json.loads(path.read_text())
        m = get_main_metrics(task, data)
        cfg = data.get("config", {})
        fam = cfg.get("families") or []
        eval_fam = cfg.get("eval_families") or []
        print(f"\n{task}")
        print(f"  source: {path}")
        print(f"  split config: train={len(fam)} eval={len(eval_fam)}")
        print(f"  main metrics: encoder={m['encoder']:.4f} diffusion={m['diffusion']:.4f} baseline={m['baseline']:.4f}")
        ood = get_ood_metrics(task, data)
        if ood is None:
            print("  ood metrics: not present in this summary")
        else:
            print(f"  ood metrics:  encoder={ood['encoder']:.4f} diffusion={ood['diffusion']:.4f} baseline={ood['baseline']:.4f}")


if __name__ == "__main__":
    main()
