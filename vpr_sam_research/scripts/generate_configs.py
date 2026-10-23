"""Generate versioned templates, never launch anything."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    smoke = {
        "schema": 1, "purpose": "smoke", "seed": 17, "device": "cpu", "threads": 1,
        "deterministic": True, "output": "runs/smoke/vpr", "model": "tiny",
        "data": {"kind": "synthetic", "root": "data/cifar10", "download": False,
                 "validation_size": 4, "synthetic_train": 16, "synthetic_test": 8,
                 "augment": True, "cutout": 4, "workers": 0},
        "optimizer": {"method": "vpr", "lr": .01, "momentum": .9, "weight_decay": .0005,
                      "nesterov": False, "rho": .02},
        "geometry": {"gamma": .001, "dtype": "float64", "warmup_steps": 0,
                     "volume_atol": 1e-10, "volume_rtol": 1e-12, "static_std": .1},
        "training": {"epochs": 2, "batch_size": 4, "scheduler": "cosine", "eta_min": 0.,
                     "checkpoint_interval_steps": 0},
        "pruning": {"n": 1, "m": 3, "scope": "all", "score": "magnitude", "tail": "dense"},
        "recovery": {"epochs": 1, "method": "sgd", "lr": .005, "momentum": .9,
                     "weight_decay": .0005, "rho": .02, "scheduler": "cosine", "eta_min": 0.},
        "evaluation": {"batch_size": 4, "probe_rho": .02, "probe_batches": 1},
    }
    for method in ("sgd", "sam", "vpr", "static", "positive", "fixed_l2"):
        c = copy.deepcopy(smoke)
        c["optimizer"]["method"] = method
        c["output"] = f"runs/smoke/{method}"
        save(f"smoke/{method}.json", c)
        formal = copy.deepcopy(c)
        formal.update(purpose="formal", seed=None, device="cuda", model="resnet18", output=f"runs/formal/{method}_SET_SEED")
        formal["data"].update(kind="cifar10", validation_size=None, synthetic_train=None,
                              synthetic_test=None, augment=None, cutout=None, workers=None)
        for key in formal["optimizer"]:
            if key != "method":
                formal["optimizer"][key] = None
        for key in formal["geometry"]:
            formal["geometry"][key] = None
        for section in ("training", "pruning", "recovery", "evaluation"):
            for key in formal[section]:
                formal[section][key] = None
        save(f"formal/{method}.json", formal)
    resnet = copy.deepcopy(smoke)
    resnet.update(model="resnet18", output="runs/smoke/resnet18")
    resnet["data"].update(synthetic_train=8, validation_size=4)
    resnet["training"].update(epochs=1, batch_size=2)
    save("smoke/resnet18.json", resnet)
    save("pruning/example_1of3_smoke.json", smoke["pruning"])


def save(name, value):
    p = ROOT / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
