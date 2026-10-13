import argparse
import json
from nmsam.data.cifar10 import cifar_loaders
from nmsam.models.tiny_cnn import TinyCNN
from nmsam.core.batch import StepBatch
from nmsam.core.loss import classification_loss
from nmsam.sam.train_step import train_step
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.experiment.metrics import step_metrics
from nmsam.utils.seed import seed_everything


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive-path", required=True)
    p.add_argument("--cache-dir", default="../N-M-SAM-DATA-CACHE")
    p.add_argument("--max-batches", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.max_batches <= 2 or args.epochs != 1:
        p.error("smoke restricted to 1 epoch and <=2 batches")
    seed_everything(0)
    data = cifar_loaders(args.archive_path, args.cache_dir)
    model = TinyCNN()
    optimizer = create_optimizer(model.parameters())
    metrics = []
    for step, (x, y) in enumerate(data["train"], 1):
        metrics.append(step_metrics(train_step(model, optimizer, StepBatch(x,y), classification_loss),step))
        if step == args.max_batches:
            break
    print(json.dumps({"data":data["provenance"], "metrics": metrics}, indent=2))
