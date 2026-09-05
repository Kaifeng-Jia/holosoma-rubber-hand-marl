from dataclasses import replace

import pytest

from holosoma.config_values import simulator
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (simulator.isaacgym, SimulatorType.ISAACGYM),
        (simulator.isaacsim, SimulatorType.ISAACSIM),
        (simulator.isaacsim_dual_robot, SimulatorType.ISAACSIM),
        (simulator.mujoco, SimulatorType.MUJOCO),
    ],
)
def test_supported_simulator_targets_map_to_backend(config, expected):
    set_simulator_type(config)
    assert get_simulator_type() == expected


def test_dual_isaacsim_still_rejects_inconsistent_config_name():
    bad = replace(
        simulator.isaacsim_dual_robot,
        config=replace(simulator.isaacsim_dual_robot.config, name="mujoco"),
    )

    with pytest.raises(ValueError, match="Config mismatch"):
        set_simulator_type(bad)
