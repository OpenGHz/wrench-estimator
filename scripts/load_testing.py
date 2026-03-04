if __name__ == "__main__":
    import os
    import numpy as np
    import mujoco
    import pickle
    import json
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    from wrench_estimator.wrench_estimator import WrenchEstimator
    from pathlib import Path

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

    # convert current to torque using 0.6A per torque unit
    coeff = np.array([0.6, 0.6, 0.6, 1.35474, 1.32355, 1.5])
    get_joint_eff = airbot_play.get_joint_eff
    # airbot_play.get_joint_eff = lambda: np.asarray(get_joint_eff()) / coeff
    all_recorded = {}
    for load in (0, 0.5):
        print(f"\n=== Testing with {load}kg load ===")
        input(f"Apply {load}kg load to the end-effector and press Enter...")
        record = []
        trails = 5
        for i in range(trails):
            print(f"\nTrail {i + 1}/{trails} for load {load}kg")
            estimator.update_state(
                airbot_play.get_joint_pos(),
                airbot_play.get_joint_vel(),
                airbot_play.get_joint_eff(),
            )
            state = estimator.state
            item = state | {
                "ext_wrench": estimator.get_ext_wrench(),
                "ee_pose": airbot_play.get_end_pose(),
            }

            print("Estimated External Wrench:", estimator.get_ext_wrench())
            print("Current Pose:", airbot_play.get_end_pose())

            record.append(item)

            input(
                "Manually pull the end of the robotic arm to cause fluctuations in the estimate."
            )
        all_recorded[load] = record

    path = Path("data/load_test_data.pkl")
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_recorded, f)
    json_path = path.with_suffix(".json")

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(all_recorded, f, indent=4, default=default)
    print("Saved recorded data to", path.parent)

    airbot_play.disconnect()
    print("Done.")
