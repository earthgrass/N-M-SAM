import argparse
import json
from nmsam.core.batch import StepBatch
from nmsam.core.loss import classification_loss
from nmsam.data.fake_data import fake_loader
from nmsam.data.cifar10 import cifar_loaders
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.models.tiny_cnn import TinyCNN
from nmsam.sam.train_step import train_step
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sparsity.registry import MaskRegistry
from nmsam.sparsity.training import sparse_step
from nmsam.utils.seed import seed_everything
from nmsam.experiment.metrics import step_metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--archive-path");
    p.add_argument("--cache-dir",default="../N-M-SAM-DATA-CACHE"); a=p.parse_args()
    seed_everything(0)
    data=cifar_loaders(a.archive_path,a.cache_dir) if a.archive_path else None
    loader=data["train"] if data else fake_loader(batch_size=4,steps=2,seed=0)
    batches=[]
    for x,y in loader:
        batches.append(StepBatch(x,y))
        if len(batches)==2: break
    results={}
    for sparse,rho in [(False,0),(False,.05),(True,0),(True,.05)]:
        seed_everything(0); m=TinyCNN() if data else TinyMLP(); opt=create_optimizer(m.parameters())
        registry=MaskRegistry(m) if sparse else None
        metrics=[]
        for step,b in enumerate(batches,1):
            report=sparse_step(m,opt,b,registry,rho=rho) if sparse else train_step(m,opt,b,classification_loss,rho=rho)
            metrics.append(step_metrics(report,step))
        results[f"{'Sparse' if sparse else 'Dense'}-{'SAM' if rho else 'SGD'}"]=metrics
    print(json.dumps({"data":data["provenance"] if data else "FakeData","methods":results},indent=2))
