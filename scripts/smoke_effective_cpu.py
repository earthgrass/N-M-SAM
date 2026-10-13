import argparse
import json
import torch
from torch.utils.data import DataLoader
from nmsam.data.cifar10 import cifar_loaders
from nmsam.data.sampler import StatefulBatchSampler
from nmsam.core.batch import StepBatch
from nmsam.core.functional_model import EffectiveLoss
from nmsam.models.tiny_cnn import TinyCNN
from nmsam.experiment.training import BaselineTrainer
from nmsam.experiment.checkpoint import save_checkpoint
from nmsam.experiment.artifacts import save_result
from nmsam.diagnostics.hvp import hvp,finite_difference_hvp
from nmsam.utils.seed import seed_everything


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--archive-path")
    p.add_argument("--cache-dir",default="../N-M-SAM-DATA-CACHE");p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:
        print(json.dumps({"dry_run":True,"status":"GPU_VALIDATION_PENDING"}));raise SystemExit(0)
    if not a.archive_path:p.error("--execute requires --archive-path")
    seed_everything(0)
    data=cifar_loaders(a.archive_path,a.cache_dir,batch_size=2)
    sampler=StatefulBatchSampler(len(data["train"].dataset),2)
    loader=DataLoader(data["train"].dataset,batch_sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(0))
    trainer=BaselineTrainer(TinyCNN().double(),"2:4-SAM-Magnitude")
    for step,(x,y) in enumerate(loader):
        trainer.train(StepBatch(x.double(),y))
        if step==1:break
    x,y=next(iter(data["calibration"]));batch=StepBatch(x.double(),y)
    loss=EffectiveLoss(trainer.model,batch,trainer.registry)
    v=torch.randn_like(loss.z);v=v/v.norm()
    true=hvp(loss,loss.z,v);fd=finite_difference_hvp(loss,loss.z,v)
    path,record=save_result("effective_smoke",{"seed":0,"steps":2},
        {"data":data["provenance"],"labels":loss.labels,"task_CE":float(loss(loss.z)),
         "hvp_fd_error":float((true-fd).norm())})
    metadata={"config_sha256":record["config_sha256"],"split_sha256":data["provenance"]["split_sha256"],"git_sha":record["git_sha"]}
    save_checkpoint("results/tiny_cifar_snapshot.pt",trainer,sampler,metadata)
    print(json.dumps({"path":str(path),"task_CE":record["task_CE"],"hvp_fd_error":record["hvp_fd_error"],
                      "checkpoint":"results/tiny_cifar_snapshot.pt"},indent=2))
