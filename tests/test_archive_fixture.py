"""Synthetic ten-image fixture only: never reads a user's real dataset."""
import hashlib
import io
import pickle
import tarfile
import numpy as np
import pytest
from torchvision.datasets import CIFAR10
from nmsam.data.cifar_archive import prepare_archive
from nmsam.data.transforms import cifar_transform


def test_tiny_archive_extract_integrity_idempotence_and_parse(tmp_path,monkeypatch):
    payload={name:pickle.dumps({"data":np.zeros((2,3072),dtype=np.uint8),"labels":[0,1]})
             for name in [f"data_batch_{i}" for i in range(1,6)]+["test_batch"]}
    payload["batches.meta"]=pickle.dumps({"label_names":[str(i) for i in range(10)]})
    digest=lambda name:hashlib.md5(payload[name]).hexdigest()
    monkeypatch.setattr(CIFAR10,"train_list",[(f"data_batch_{i}",digest(f"data_batch_{i}")) for i in range(1,6)])
    monkeypatch.setattr(CIFAR10,"test_list",[("test_batch",digest("test_batch"))])
    monkeypatch.setattr(CIFAR10,"meta",{"filename":"batches.meta","key":"label_names","md5":digest("batches.meta")})
    archive=tmp_path/"tiny.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for name,value in payload.items():
            member=tarfile.TarInfo("cifar-10-batches-py/"+name);member.size=len(value)
            tar.addfile(member,io.BytesIO(value))
    root,sha=prepare_archive(archive,tmp_path/"cache")
    assert prepare_archive(archive,tmp_path/"cache")== (root,sha)
    dataset=CIFAR10(str(root),train=True,transform=cifar_transform(False),download=False)
    assert len(dataset)==10 and dataset[0][0].shape==(3,32,32)
    (root/"cifar-10-batches-py/data_batch_1").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError,match="corrupt"):prepare_archive(archive,tmp_path/"cache")
