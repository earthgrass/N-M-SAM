from nmsam.core.step_types import SAMStepReport
from nmsam.experiment.metrics import step_metrics


def test_metrics_serialization_without_training():
    report=SAMStepReport(1.,1.1,.1,.5,2.,3.,.05,3.,0.,2,False)
    result=step_metrics(report,1)
    assert result["task_loss"]==1. and result["optimizer_loss"]==1.1
    assert result["forward_passes"]==2 and "trace" not in result

