import ast
import copy
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from data import Cutout,Images,load_split,make_loader,sample_rng
from model import build_model

ROOT=Path(__file__).resolve().parents[1]


def test_model_shapes_and_frozen_split():
    model=build_model(); c=yaml.safe_load((ROOT/'config.yaml').read_text())
    assert model.conv1.kernel_size==(3,3) and model.conv1.stride==(1,1) and model.conv1.padding==(1,1)
    assert [len(getattr(model,f'layer{i}')) for i in range(1,5)]==[2,2,2,2]
    assert [getattr(model,f'layer{i}')[0].conv1.out_channels for i in range(1,5)]==[64,128,256,512]
    assert [getattr(model,f'layer{i}')[0].conv1.stride for i in range(1,5)]==[(1,1),(2,2),(2,2),(2,2)]
    assert model.linear.out_features==10 and sum(p.numel() for p in model.parameters())==11173962
    manifest=load_split(ROOT,c)
    assert len(manifest['train_indices'])==45000 and len(manifest['validation_indices'])==5000


def test_post_normalize_cutout_and_validation_transform():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    dataset=[(Image.fromarray(np.full((32,32,3),255,dtype=np.uint8)),2)]
    training=Images(dataset,[0],c,True)
    assert [type(t).__name__ for t in training.transform.transforms]==['RandomCrop','RandomHorizontalFlip','ToTensor','Normalize','Cutout']
    validation=Images(dataset,[0],c)
    x,y,i=validation[0]
    torch.testing.assert_close(x[:,0,0],(torch.ones(3)-torch.tensor(c['normalize_mean']))/torch.tensor(c['normalize_std']))
    with sample_rng(6):
        result=Cutout(16,1.0)(torch.ones(3,32,32)*2.)
    assert (result==0).any() and (result==2).any()
    assert len(validation.transform.transforms)==2


def test_sample_rng_independent_and_tail_preserved():
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    dataset=[(Image.fromarray(np.full((32,32,3),i%256,dtype=np.uint8)),i%10) for i in range(129)]
    images=Images(dataset,list(range(129)),c,True)
    state=torch.get_rng_state().clone(); a=images[0][0]; b=images[0][0]
    assert torch.equal(a,b) and torch.equal(state,torch.get_rng_state())
    loader=make_loader(images,c,shuffle=True,workers=0)
    assert [len(batch[1]) for batch in loader]==[128,1]
