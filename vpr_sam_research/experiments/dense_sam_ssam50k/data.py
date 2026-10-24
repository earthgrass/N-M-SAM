"""SSAM2022 official transforms and RandomSampler, full official CIFAR10 splits."""
import hashlib
import inspect
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10


class WorkerWitness(Dataset):
    def __len__(self): return 2
    def __getitem__(self,index):
        worker=torch.utils.data.get_worker_info()
        return {'worker_id':worker.id,'torch_seed':torch.initial_seed(),
                'python_first':random.random(),'numpy_first':float(np.random.rand()),
                'torch_first':float(torch.rand(()))}


def verify_worker_rng():
    """Check PyTorch's actual default worker loop, without changing training RNG."""
    from torch.utils.data._utils.worker import _generate_state,_worker_loop
    saved=random.getstate(),np.random.get_state(),torch.get_rng_state()
    try:
        rows=[]
        for batch in DataLoader(WorkerWitness(),batch_size=1,num_workers=2):
            row={k:v.item() for k,v in batch.items()}
            worker_id=int(row['worker_id']); worker_seed=int(row['torch_seed']); base=worker_seed-worker_id
            assert row['python_first']==random.Random(worker_seed).random()
            assert row['numpy_first']==float(np.random.RandomState(_generate_state(base,worker_id)).rand())
            assert row['torch_first']==float(torch.rand((),generator=torch.Generator().manual_seed(worker_seed)))
            rows.append(row)
        assert {r['worker_id'] for r in rows}=={0,1}
        return {'status':'PASS','base_seed':int(rows[0]['torch_seed']), 'workers':rows,
                'policy':'Default PyTorch worker seeding: Python/Torch base_seed+worker_id; NumPy _generate_state(base_seed,worker_id)',
                'worker_loop_source_sha256':hashlib.sha256(inspect.getsource(_worker_loop).encode()).hexdigest(),
                'parent_seed':0,'parent_rng_preserved':True}
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])


class Cutout:
    def __init__(self,size=16,p=0.5):
        self.size,self.p=size,p

    def __call__(self,image):
        if torch.rand([1]).item()>self.p:
            return image
        h,w=image.size(1),image.size(2)
        mask=np.ones((h,w),np.float32)
        x,y=np.random.randint(w),np.random.randint(h)
        half=self.size//2
        mask[max(0,y-half):min(h,y+half),max(0,x-half):min(w,x+half)]=0.
        return image*torch.from_numpy(mask).expand_as(image)


class Images(Dataset):
    def __init__(self,dataset,config,training=False):
        self.dataset=dataset
        parts=[]
        if training:
            parts=[transforms.RandomCrop(32,padding=config['crop_padding']),
                   transforms.RandomHorizontalFlip(config['horizontal_flip_probability'])]
        parts += [transforms.ToTensor(),transforms.Normalize(config['normalize_mean'],config['normalize_std'])]
        if training:
            parts.append(Cutout(config['cutout_size'],config['cutout_probability']))
        self.transform=transforms.Compose(parts)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        image,label=self.dataset[index]
        return self.transform(image),label,index


def make_loader(dataset,c,training=False,workers=None):
    return DataLoader(dataset,sampler=RandomSampler(dataset) if training else SequentialSampler(dataset),
                      batch_size=c['batch_size'] if training else c['evaluation_batch_size'],
                      num_workers=c['workers'] if workers is None else workers,pin_memory=c['pin_memory'],
                      drop_last=c['drop_last'] if training else False)


def real_data(c):
    location=Path(c['data_root'])
    train=CIFAR10(str(location),train=True,download=False)
    test=CIFAR10(str(location),train=False,download=False)
    if (len(train),len(test))!=(50000,10000):
        raise ValueError('Official CIFAR10 sample counts failed')
    records={}
    for p in sorted((location/'cifar-10-batches-py').iterdir()):
        if p.is_file():
            records[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'identity':'official-cifar10-md5-verified','train_size':50000,'test_size':10000,
              'validation_size':0,'train_indices':list(range(50000)),'test_indices':list(range(10000)),
              'train_classes':np.bincount(train.targets,minlength=10).tolist(),
              'test_classes':np.bincount(test.targets,minlength=10).tolist(),'files_sha256':records,
              'train_drop_last':True,'test_drop_last':False,'test_used_for_epoch_selection':True}
    return Images(train,c,True),Images(test,c),manifest
