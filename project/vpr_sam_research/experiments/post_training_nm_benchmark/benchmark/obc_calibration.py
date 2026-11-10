"""Pinned OBC image augmentation, explicitly adapted to CIFAR size/parent Normalize.

Every layer replays identical shuffled augmented views. CPU RNG is private to the
calibration stream; it never consumes recovery RNG or changes the dense teacher.
"""
import hashlib,random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset

class CalibrationImages(Dataset):
    def __init__(self,dataset,ids,data):
        self.dataset,self.ids=dataset,ids
        self.transform=transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.2,1.0),ratio=(0.75,4/3)),
            transforms.RandomHorizontalFlip(0.5),transforms.ToTensor(),
            transforms.Normalize(data['normalize_mean'],data['normalize_std'])])
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        idx=self.ids[i];image,label=self.dataset[idx]
        return self.transform(image),label,idx

def factory(dataset,ids,data,calibration,check=lambda:None,receipt=None):
    def batches():
        saved=random.getstate(),np.random.get_state(),torch.get_rng_state()
        order=hashlib.sha256();images=hashlib.sha256();count=0;passes=0
        try:
            seed=calibration['random_seed']
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            ds=CalibrationImages(dataset,ids,data)
            loader=DataLoader(ds,batch_size=calibration['batch_size'],shuffle=True,
                              num_workers=0,pin_memory=data['pin_memory'],drop_last=False)
            for _ in range(calibration['rounds']):
                for x,y,idx in loader:
                    check();order.update(idx.numpy().astype('<i8').tobytes());images.update(x.numpy().tobytes())
                    count+=len(idx);passes+=1
                    yield x,y,idx
            if receipt is not None:
                receipt.append({'samples_including_rounds':count,'batches':passes,
                    'order_sha256':order.hexdigest(),'augmented_tensor_sha256':images.hexdigest(),
                    'workers':0,'shuffle':True,'rounds':calibration['rounds']})
        finally:
            random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])
    return batches
