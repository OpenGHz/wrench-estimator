if __name__ == "__main__":
    import os
    import numpy as np
    import mujoco
    import pickle
    import json
    import time
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    from wrench_estimator.wrench_estimator import WrenchEstimator
    from pathlib import Path
    from pprint import pprint

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
    airbot_play.get_joint_eff = lambda: np.asarray(get_joint_eff()) / coeff
    all_recorded = {}
    load_stats = {}
    loads = (0, 0.5, 1.0, 1.5, 2.0)
    loads = reversed(loads)
    # loads = (0,)
    for load in loads:
        print(f"\n=== Testing with {load}kg load ===")
        input("Press Enter to open the gripper")
        airbot_play.move_eef_pos(0.074)
        if load > 0:
            input(f"Apply {load}kg load to the end-effector and press Enter...")
            airbot_play.move_eef_pos(0.0)
        trail_records = []
        trails = 1
        for i in range(trails):
            print(f"\nTrail {i + 1}/{trails} for load {load}kg")
            # if i > 0:
            #     input(
            #         "Manually pull the end of the robotic arm to cause fluctuations in the estimate."
            #     )
            # else:
            #     print("Wait a moment for the initial estimate to stabilize...")
            #     time.sleep(2)
            input("Press Enter to start recording data for this trail...")
            record = []
            for j in range(30):
                print(j, end=" ", flush=True)
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
                record.append(item)
                time.sleep(0.2)
            else:
                print()  # for new line after the progress numbers
                force_z = [r["ext_wrench"][2] for r in record]
                stats = {
                    "z_force_mean": np.mean(force_z),
                    "z_force_std": np.std(force_z),
                    "z_force_min": np.min(force_z),
                    "z_force_max": np.max(force_z),
                }
                pprint(stats)
                load_stats[load] = stats

            print("Estimated External Wrench:", estimator.get_ext_wrench())
            print("Current Pose:", airbot_play.get_end_pose())

            trail_records.append(record)
        all_recorded[load] = trail_records

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

    stats_path = Path("data/load_test_stats.json")
    with open(stats_path, "w") as f:
        json.dump(load_stats, f, indent=4)
    print("Saved load test stats to", stats_path)

    airbot_play.disconnect()
    print("Done.")
