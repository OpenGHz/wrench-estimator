import mujoco
import numpy as np


class WrenchEstimator:
    def __init__(self, mj_model, ndof, ee_body_name):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.ndof = ndof
        self.tau_measured = None
        self.jacp = np.zeros((3, self.mj_model.nv))
        self.jacr = np.zeros((3, self.mj_model.nv))
        self.point = np.zeros(3, dtype=np.float64)
        self.ee_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )

    def update_state(self, q, dq, tau=None):
        q = np.asarray(q, copy=True)
        dq = np.asarray(dq, copy=True)
        tau = np.asarray(tau, dtype=np.float64, copy=True) if tau is not None else None

        if q.shape[0] < self.ndof or dq.shape[0] < self.ndof:
            raise ValueError("q/dq length is smaller than ndof")
        if tau is not None and tau.shape[0] < self.ndof:
            raise ValueError("tau length is smaller than ndof")

        self.mj_data.qpos[: self.ndof] = q.copy()
        self.mj_data.qvel[: self.ndof] = dq.copy()
        self.tau_measured = tau[: self.ndof].copy() if tau is not None else None
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_ext_force(self):
        if self.tau_measured is None:
            raise ValueError(
                "Measured joint torque is required for external force estimation"
            )

        # Save state to avoid corrupting next estimation call.
        qvel_backup = self.mj_data.qvel.copy()
        qacc_backup = self.mj_data.qacc.copy()

        # Inverse dynamics with qacc=0 gives model torque needed to hold current q,dq.
        self.mj_data.qacc[:] = 0.0
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        tau_model = self.mj_data.qfrc_inverse[: self.ndof].copy()

        # Restore state.
        self.mj_data.qvel[:] = qvel_backup
        self.mj_data.qacc[:] = qacc_backup

        # Compute the Jacobian at end-effector body origin (world coordinates).
        self.point[:] = self.mj_data.xpos[self.ee_id]
        mujoco.mj_jac(
            self.mj_model, self.mj_data, self.jacp, self.jacr, self.point, self.ee_id
        )

        # Use only the first ndof joints of the controlled arm.
        J = np.vstack((self.jacp[:, : self.ndof], self.jacr[:, : self.ndof]))

        tau_ext_joint = self.tau_measured - tau_model

        # Solve J^T * wrench = tau_ext_joint in least-squares sense.
        force_ext = np.linalg.pinv(J.T) @ -tau_ext_joint
        return force_ext


if __name__ == "__main__":
    import os
    from airbot_py.arm import AIRBOTPlay, RobotMode

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
