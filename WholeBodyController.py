import numpy as np
from FullBodyDynamics import *
from scipy.spatial.transform import Rotation as R
import nlopt
import os


class optimize_data:
    def __init__(self, p_init, R_init, twist_init, b_footends, ref_p, ref_R):
        self.p_init = p_init
        self.R_init = R_init
        self.twist_init = twist_init
        self.b_footends = b_footends
        self.ref_p = ref_p
        self.ref_R = ref_R


class WholeBodyController:
    def __init__(self, init_p, init_R, init_v, init_w, init_mods):
        self.T_sb = HomoTransform(init_R, init_p)
        self.V_sb = twist(init_v, init_w)

        self.dim_wb = 0.44
        self.dim_l = 0.6
        self.dim_w = 0.4
        self.dim_h = 0.15
        r_ = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        T_ba = HomoTransform(r_, np.array([self.dim_wb / 2, self.dim_w / 2, 0]))
        T_bb = HomoTransform(r_, np.array([self.dim_wb / 2, -self.dim_w / 2, 0]))
        T_bc = HomoTransform(r_, np.array([-self.dim_wb / 2, -self.dim_w / 2, 0]))
        T_bd = HomoTransform(r_, np.array([-self.dim_wb / 2, self.dim_w / 2, 0]))
        self.T_bmod = [T_ba, T_bb, T_bc, T_bd]

        self.footends = init_mods
        self.bfootends = []
        for i, ft in enumerate(self.footends):
            self.bfootends.append(self.legFrame2baseFrame(i, ft))
            print(self.legFrame2baseFrame(i, ft))

        # T_sb_0 = HomoTransform(np.eye(3), np.array([0, 0, 0.2]))
        # r = R.from_rotvec(np.array([0, -15, 0]), degrees=True)
        # T_sb_1 = HomoTransform(r.as_matrix(), np.array([0, 0, 0.2]))

        # optimization
        self.optimize_data = optimize_data(
            init_p, init_R, self.V_sb, self.bfootends, init_p, init_R
        )

        pass

    def controller(self, ref_p_k, ref_R_k, p_k, R_k, v_k, w_k, mod_footends, dt=0.001):
        # update controller state
        self.T_sb = HomoTransform(R_k, p_k)
        self.V_sb = twist(v_k, w_k)
        T_sb_1 = HomoTransform(ref_R_k, ref_p_k)

        bfootends_1 = []
        for i, ft in enumerate(mod_footends):
            self.bfootends[i] = self.legFrame2baseFrame(i, ft)
            bfootends_1.append(
                self.footendTransform(self.bfootends[i], self.T_sb, T_sb_1)
            )

        self.optimize_data = optimize_data(
            self.T_sb.p, self.T_sb.R, self.V_sb, bfootends_1, ref_p_k, ref_R_k
        )
        self.forceOptimizer()

    def forceOptimizer(self):
        # footend_state w.r.t. base frame
        # optimize_state [afx, afz, bfx, bfz, cfx, cfz, dfx, dfz]

        b_fx_0 = 0
        b_fz_0 = 50

        opt = nlopt.opt(nlopt.LN_COBYLA, 8)
        opt.set_lower_bounds([-10, 0, -10, 0, -10, 0, -10, 0])
        opt.set_upper_bounds([10, 80, 10, 80, 10, 80, 10, 80])

        opt.set_xtol_rel(1e-2)
        # opt.set_xtol_abs(1e-2)
        # opt.set_maxeval(10000)
        opt.set_min_objective(self.objectiveFunc)
        f = opt.optimize(
            [b_fx_0, b_fz_0, b_fx_0, b_fz_0, b_fx_0, b_fz_0, b_fx_0, b_fz_0]
        )

        minf = opt.last_optimum_value()
        print(
            "optimum at ",
            f[0],
            f[1],
            " | ",
            f[2],
            f[3],
            "\n           ",
            f[4],
            f[5],
            " | ",
            f[6],
            f[7],
        )
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())

    def objectiveFunc(self, fs, _):
        p_k = self.optimize_data.p_init
        R_k = self.optimize_data.R_init
        V_sb = self.optimize_data.twist_init
        bfootends = self.optimize_data.b_footends
        ref_p = self.optimize_data.ref_p
        ref_R = self.optimize_data.ref_R

        fb_ = FloatingBase(p_k, R_k, V_sb)
        t0, tf = [0, dt]
        wrenches = []
        for i in range(4):
            fx = fs[2 * i]
            fz = fs[2 * i + 1]
            b_f_ = np.array([fx, 0, fz])
            r_ = bfootends[i]
            w = wrench(np.cross(r_, b_f_), b_f_)
            wrenches.append(w)

        traj = fb_.solveFowardDyamic(
            t0, tf, self.T_sb, self.V_sb, wrenches=wrenches, dt=dt
        )
        T_sb_1 = traj[1][1]
        p_1 = T_sb_1.p
        R_1 = T_sb_1.R

        # R_10 = R_1.T @ R_k
        R_10 = ref_R.T @ R_1
        r_ = R.from_matrix(R_10)
        rv = r_.as_rotvec()
        err_r = np.linalg.norm(rv)
        err_p = np.linalg.norm(ref_p.reshape(3) - p_1.reshape(3))
        # print("ref_rot, ", R.from_matrix(ref_R).as_rotvec(degrees=True))
        # print("opt_rot, ", R.from_matrix(R_1).as_rotvec(degrees=True))
        # print("err_rot, ", R.from_matrix(R_10).as_rotvec(degrees=True))
        # print("err_p, ", ref_p.reshape(3) - p_1.reshape(3))
        # print("error, ", err_r + err_p)
        # print("--")
        # print("rotvec, ", r_.as_rotvec(degrees=True))

        return err_r + err_p

    def footendTransform(self, b0_P_ft, T_sb_0, T_sb_1):
        # Transform footend pose from current body frame to next body frame
        # T_sb_0, T_sb_1 -> HomoTransform object
        # T_sb_1_ = HomoTransform(T_sb_1[:3, :3], T_sb_1[:3, 3])
        # print("s_ft,", (T_sb_0.Mat @ b0_P_ft_).reshape(4))
        b0_P_ft_ = np.vstack((b0_P_ft.reshape((-1, 1)), 1))
        b1_P_ft = T_sb_1.inverse().Mat @ T_sb_0.Mat @ b0_P_ft_
        return b1_P_ft[:3, 0]

    def legFrame2baseFrame(self, idx, leg_xyz):
        p_leg = np.block([[leg_xyz.reshape((-1, 1))], [1]])
        p_base = self.T_bmod[idx].Mat @ p_leg
        return p_base[:3, 0]

    def baseFrame2legFrame(self, idx, base_xyz):
        p_base = np.block([[base_xyz.reshape((-1, 1))], [1]])
        p_leg = self.T_bmod[idx].inverse().Mat @ p_base
        return p_leg[:3, 0]


if __name__ == "__main__":
    os.nice(-20)
    # initial condition
    init_p = np.array([0, 0, 0.2])
    init_R = np.eye(3)
    init_v = np.array([0, 0, 0])
    init_w = np.array([0, 0, 0])

    init_modA = np.array([0, -0.2, 0])
    init_modB = np.array([0, -0.2, 0])
    init_modC = np.array([0, -0.2, 0])
    init_modD = np.array([0, -0.2, 0])
    init_mods = [init_modA, init_modB, init_modC, init_modD]

    t0, tf = [0, 5]
    dt = 0.001
    ts = np.arange(t0, tf + dt, dt)

    ref_p = init_p
    ref_R = init_R
    for t in ts:
        pitch_k = 15 * np.sin(t)
        r_ = R.from_euler("zyx", [0, 0, pitch_k], degrees=True)
        R_ = r_.as_matrix()
        ref_p = np.vstack((ref_p, init_p))
        ref_R = np.vstack((ref_R, R_))

    wbc = WholeBodyController(init_p, init_R, init_v, init_w, init_mods)
    k = 3

    r = R.from_rotvec([0, +0.005, 0], degrees=True)
    ref_R_ = r.as_matrix()
    start_time = time.time()
    wbc.controller(
        ref_p[k, :],
        # ref_R[2 * k : 2 * k + 3, :],
        # np.array([0, 0, 0]),
        ref_R_,
        init_p,
        init_R,
        init_v,
        init_w,
        init_mods,
    )
    print(ref_p[k, :])
    print("--- %s seconds ---" % (time.time() - start_time))
    # wbc.solveForce(ref_p[k, :], ref_R[2 * k : 2 * k + 3, :])
