from holosoma.config_values import simulator


def test_plan5_dual_robot_simulator_is_opt_in() -> None:
    assert simulator.isaacsim._target_ == "holosoma.simulator.isaacsim.isaacsim.IsaacSim"
    assert simulator.isaacsim_dual_robot._target_ == (
        "holosoma.simulator.isaacsim.dual_robot_isaacsim.DualRobotIsaacSim"
    )
    assert simulator.DEFAULTS["isaacsim"] is simulator.isaacsim
    assert simulator.DEFAULTS["isaacsim_dual_robot"] is simulator.isaacsim_dual_robot
    assert simulator.isaacsim_dual_robot.config is simulator.isaacsim.config
