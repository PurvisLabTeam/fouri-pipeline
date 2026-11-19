import pytest

from fouri.app.steps import settings_for
from fouri.status import PER_WELL_STEPS, Step


@pytest.mark.parametrize("step", PER_WELL_STEPS)
@pytest.mark.parametrize("overwrite", [True, False])
def test_overwrite_is_sent_to_the_step(step: Step, overwrite: bool) -> None:
    assert settings_for(step, {}, overwrite=overwrite, workers=None)["overwrite"] is overwrite
