if __name__ == "__main__":
    import os
    import numpy as np
    import mujoco
    import time
    import rerun as rr
    from contextlib import suppress
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    from wrench_estimator.wrench_estimator import WrenchEstimator

    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mjcf_path = os.path.join(
        script_dir,
        "../model_assets",
        "mjcf",
        "manipulator",
        "robot_airbot_play_force.xml",
    )
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    estimator = WrenchEstimator(model, ndof=6, ee_body_name="link6")

    airbot_play = AIRBOTPlay("localhost", 50051)
    airbot_play.connect()
    airbot_play.set_speed_profile(SpeedProfile.FAST)
    airbot_play.switch_mode(RobotMode.PLANNING_POS)
    target_pose = [[0.259, -0.026, 0.176], [0.0, 0.707, 0.0, 0.707]]
    airbot_play.move_to_cart_pose(target_pose)

    # airbot_play.move_eef_pos(0.074)
    # input("Apply load to the end-effector and press Enter...")
    # airbot_play.move_eef_pos(0.0)

    # convert current to torque using 0.6A per torque unit
    coeff = np.array([0.6, 0.6, 0.6, 1.35474, 1.32355, 1.5])
    get_joint_eff = airbot_play.get_joint_eff
    airbot_play.get_joint_eff = lambda: np.asarray(get_joint_eff()) / coeff

    rr.init("wrench_estimation_test", spawn=True)
    with suppress(KeyboardInterrupt):
        while True:
            estimator.update_state(
                airbot_play.get_joint_pos(),
                airbot_play.get_joint_vel(),
                airbot_play.get_joint_eff(),
            )
            wrench = estimator.get_ext_wrench()
            force, torque = wrench[:3], wrench[3:]
            rr.set_time("timestamp", timestamp=time.time())
            rr.log(
                "wrench/force",
                rr.Arrows3D(
                    vectors=[force],
                    origins=[[0, 0, 0]],
                    colors=[[255, 0, 0]],
                    radii=0.02,
                    labels=["force"],
                ),
            )
            # rr.log(
            #     "wrench/force",
            #     rr.SeriesLines(
            #         colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            #         names=["x", "y", "z"],
            #     ),
            #     static=True,
            # )
            for field, value in zip(("x", "y", "z"), force):
                rr.log(f"wrench/force/{field}", rr.Scalars(value))
            f_magnitude = np.linalg.norm(force)
            rr.log("force/Magnitude", rr.Scalars(f_magnitude))
            # for field, value in zip(("x", "y", "z"), torque):
            #     rr.log(f"wrench/torque/{field}", rr.Scalars(value))
            print("Estimated External Wrench:", wrench)
            print("Current Pose:", airbot_play.get_end_pose())
            time.sleep(0.1)

    airbot_play.disconnect()

    print("Done.")
