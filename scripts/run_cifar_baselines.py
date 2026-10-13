"""Fair bounded six-method comparison; no downloads or automatic long runs."""
import argparse
import json
from nmsam.core.batch import StepBatch
from nmsam.data.fake_data import fake_loader
from nmsam.data.cifar10 import cifar_loaders
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.models.tiny_cnn import TinyCNN
from nmsam.models.resnet_cifar import resnet18_cifar
from nmsam.experiment.training import BaselineTrainer,METHODS
from nmsam.experiment.artifacts import save_result
from nmsam.utils.seed import seed_everything


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive-path");p.add_argument("--cache-dir",default="../N-M-SAM-DATA-CACHE")
    p.add_argument("--max-batches",type=int,default=2);p.add_argument("--model",choices=["tiny","resnet18"],default="tiny")
    p.add_argument("--seed",type=int,default=0);a=p.parse_args()
    if not 1<=a.max_batches<=16: p.error("bounded CPU protocol permits 1..16 batches")
    seed_everything(a.seed)
    data=cifar_loaders(a.archive_path,a.cache_dir,batch_size=2,seed=a.seed) if a.archive_path else None
    loader=data["train"] if data else fake_loader(batch_size=4,steps=a.max_batches,seed=a.seed)
    batches=[]
    for x,y in loader:
        batches.append(StepBatch(x,y))
        if len(batches)==a.max_batches:break
    result={}
    for method in METHODS:
        seed_everything(a.seed)
        model=(resnet18_cifar() if a.model=="resnet18" else TinyCNN()) if data else TinyMLP()
        trainer=BaselineTrainer(model,method)
        result[method]=[trainer.train(b) for b in batches]
    config={"seed":a.seed,"max_batches":a.max_batches,"model":a.model,
        "lr":.05,"rho":.05,"l2_reg":.001,"mask_interval":1,"scheduler":"constant",
        "batch_size":2 if data else 4,"augmentation":"shared materialized crop/flip" if data else "ToTensor"}
    path,record=save_result("baselines",config,{"data":data["provenance"] if data else "FakeData","methods":result})
    print(json.dumps({"path":str(path),"final_CE":{k:v[-1]["task_loss"] for k,v in result.items()}},indent=2))
