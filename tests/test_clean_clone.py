from pathlib import Path
import subprocess


def test_data_source_tracked_and_not_ignored():
    root = Path(__file__).resolve().parents[1]
    source = "src/nmsam/data/fake_data.py"
    result = subprocess.run(["git", "check-ignore", "--no-index", source],
                            cwd=root, capture_output=True)
    assert result.returncode == 1
    tracked = subprocess.check_output(["git", "ls-files", "src/nmsam/data"], cwd=root).decode()
    assert source in tracked and "src/nmsam/data/__init__.py" in tracked
