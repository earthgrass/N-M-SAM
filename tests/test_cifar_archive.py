import io
import tarfile
import pytest
from nmsam.data.cifar_archive import checked_members, prepare_archive
from nmsam.data.cifar10 import fixed_split


@pytest.mark.parametrize("name,kind", [("../escape",tarfile.REGTYPE),
    ("/absolute",tarfile.REGTYPE), ("C:/escape",tarfile.REGTYPE),
    ("cifar-10-batches-py/link",tarfile.SYMTYPE), ("cifar-10-batches-py/hard",tarfile.LNKTYPE)])
def test_unsafe_tar_rejected(name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w") as tar:
        item=tarfile.TarInfo(name); item.type=kind; tar.addfile(item)
    stream.seek(0)
    with tarfile.open(fileobj=stream,mode="r") as tar, pytest.raises(ValueError):
        checked_members(tar)


def test_split_disjoint_reproducible():
    import numpy as np
    train,val,digest=fixed_split()
    assert len(train)==45000 and len(val)==5000
    assert not set(train)&set(val) and len(set(train)|set(val))==50000
    assert np.array_equal(train,fixed_split()[0]) and digest==fixed_split()[2]


def test_truncated_archive_reports_and_leaves_no_cache(tmp_path):
    file=tmp_path/"broken.tar.gz"; file.write_bytes(b"bad data")
    with pytest.raises(ValueError,match="invalid/truncated"):
        prepare_archive(file,tmp_path/"cache")
    assert not list((tmp_path/"cache").iterdir())
