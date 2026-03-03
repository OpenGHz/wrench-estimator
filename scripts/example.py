if __name__ == "__main__":
    import os
    from airbot_py.arm import AIRBOTPlay, RobotMode
    import numpy as np
    from wrench_estimator.wrench_estimator import WrenchEstimator
    import mujoco

    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(
        script_dir,
        "model_assets",
        "mjcf",
        "manipulator",
        "robot_airbot_play_force.xml",
    )
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    estimator = WrenchEstimator(model, ndof=6, ee_body_name="link6")

    airbot_play = AIRBOTPlay("localhost", 50051)
    airbot_play.connect()
    airbot_play.switch_mode(RobotMode.PLANNING_POS)
    target_pose = [[0.259, -0.026, 0.176], [0.0, 0.707, 0.0, 0.707]]
    airbot_play.move_to_cart_pose(target_pose)
    input("Press Enter to continue")
    # convert current to torque using 0.6A per torque unit
    coeff = np.array([0.6, 0.6, 0.6, 1.35474, 1.32355, 1.5])
    get_joint_eff = airbot_play.get_joint_eff
    # airbot_play.get_joint_eff = lambda: np.asarray(get_joint_eff()) / coeff

    estimator.update_state(
        airbot_play.get_joint_pos(),
        airbot_play.get_joint_vel(),
        airbot_play.get_joint_eff(),
    )

    print("Estimated External Force:", estimator.get_ext_force())
    print("Current Pose:", airbot_play.get_end_pose())

    input("Press Enter to exit...")
    airbot_play.disconnect()

    print("Done.")
