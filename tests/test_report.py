import pytest
from nmsam.experiment.compare import report_records


def test_empty_report_is_pending_not_scientific_evidence():
    text=report_records([])
    assert "GPU_VALIDATION_PENDING" in text and "matched-backward" in text
    with pytest.raises(ValueError):report_records([{}])
