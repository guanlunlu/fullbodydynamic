import numpy as np
from scipy.spatial.transform import Rotation as R

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# mpl.rcParams["figure.dpi"] = 150

# for animation
import visualize as vs
import threading
import time


def skewsym_mat(vx):
    x = vx.reshape(3)
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


class twist:
    def __init__(self, v, w):
        self.v = v.reshape((3, 1))
        self.w = w.reshape((3, 1))
        # self.Mat = np.block([[self.v], [self.w]])
        self.Mat = np.block([[self.w], [self.v]])

    def adjoint(self):
        ad11 = skewsym_mat(self.w)
        ad12 = np.zeros((3, 3))
        ad21 = skewsym_mat(self.v)
        ad22 = skewsym_mat(self.w)
        return np.block([[ad11, ad12], [ad21, ad22]])

    def convertFrame(self, T_base2target):
        v_now = self.Mat
        v_new = T_base2target.adjoint() @ v_now
        # print("adjoint")
        # print(T_base2target.adjoint())
        # print("twist_now: ", v_now)
        # print("twist_new: ", v_new)
        twist_new = twist(v_new[3:], v_new[0:3])
        return twist_new

    def integrate(self, dvdt, dwdt, dt=0.001):
        return twist(self.v + dvdt * dt, self.w + dwdt * dt)


class HomoTransform:
    def __init__(self, R, p):
        # SE(3) group
        self.R = R.astype("double")
        self.p = p.reshape((3, 1))
        self.Mat = np.block([[self.R, self.p], [np.zeros((1, 3)), 1]])

    def adjoint(self):
        ad11 = self.R
        ad12 = np.zeros((3, 3))
        # ad21 = skewsym_mat(self.p) @ self.R
        ad21 = skewsym_mat(self.p) @ self.R
        ad22 = self.R
        return np.block([[ad11, ad12], [ad21, ad22]])

    def inverse(self):
        t11 = np.transpose(self.R)
        t12 = -t11 @ self.p
        return HomoTransform(t11, t12)

    def integrate(self, twist_b, dt=0.001):
        w_norm = np.linalg.norm(twist_b.w)
        if w_norm > 0.0000001:
            # w_hat = (twist_b.w / w_norm)
            # rot = R.from_rotvec(w_hat.reshape(3), w_norm * dt)
            rot = R.from_rotvec(twist_b.w.reshape(3) * dt)
            rotm = rot.as_matrix()
        else:
            rotm = np.eye(3)
        # rot = R.from_rotvec(twist_b.w.reshape(3) * dt)
        # rotm = rot.as_matrix()
        R_n = rotm @ self.R
        p_n = twist_b.v * dt + self.p
        return HomoTransform(R_n, p_n)

    def plot(self, axs, scale=0.1):
        vx = self.R[:, 0] * scale
        vy = self.R[:, 1] * scale
        vz = self.R[:, 2] * scale
        p_ = self.p.reshape(3)
        ax = Arrow3D(
            [p_[0], p_[0] + vx[0]],
            [p_[1], p_[1] + vx[1]],
            [p_[2], p_[2] + vx[2]],
            mutation_scale=3,
            lw=1,
            arrowstyle="-|>",
            color="r",
        )
        ay = Arrow3D(
            [p_[0], p_[0] + vy[0]],
            [p_[1], p_[1] + vy[1]],
            [p_[2], p_[2] + vy[2]],
            mutation_scale=3,
            lw=1,
            arrowstyle="-|>",
            color="g",
        )
        az = Arrow3D(
            [p_[0], p_[0] + vz[0]],
            [p_[1], p_[1] + vz[1]],
            [p_[2], p_[2] + vz[2]],
            mutation_scale=3,
            lw=1,
            arrowstyle="-|>",
            color="b",
        )
        axs.add_artist(ax)
        axs.add_artist(ay)
        axs.add_artist(az)
        axs.scatter([self.p[0]], [self.p[1]], [self.p[2]])


class wrench:
    def __init__(self, moment, force):
        self.m = moment.reshape((3, 1))
        self.f = force.reshape((3, 1))
        self.Mat = np.vstack((self.m, self.f))

    def convertFrame(self, T_base2target):
        # convert wrench from current base frame to target frame
        F_base = self.Mat
        F_target = np.transpose(T_base2target.adjoint()) @ F_base
        return wrench(F_target[:3], F_target[3:])


class FloatingBase:
    def __init__(self, init_p, init_R, init_vb):
        # input param.
        # init_R -> 3x3 matrix
        # init_p -> 3x1 vector
        # init_vb -> twist object

        # Floating Base physic prop.
        self.m = 21
        self.g = -9.81
        self.w = 577.5 * 0.001
        self.h = 144 * 0.001
        self.l = 329.5 * 0.001
        self.Ixx = 1 / 12 * self.m * (self.w**2 + self.h**2)
        self.Iyy = 1 / 12 * self.m * (self.l**2 + self.h**2)
        self.Izz = 1 / 12 * self.m * (self.l**2 + self.w**2)

        self.Ib = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.Mb = np.diag([self.m, self.m, self.m])
        self.Mq = np.block([[self.Ib, np.zeros((3, 3))], [np.zeros((3, 3)), self.Mb]])

        # initial configuration w.r.t. fixed frame (s)
        self.T_sb = HomoTransform(init_R, init_p)
        self.V_b = init_vb

        print("---")
        print("Floating Base initialized at \n p: ", init_p, ",\n R: ", init_R)
        print(" T_sb0: \n", self.T_sb.Mat)
        print(" V_b0:\n", self.V_b.Mat.reshape((1, -1)))
        print("gravity_bodyframe")
        print(self.gravityWrench(self.T_sb).Mat.reshape((1, -1)))
        print("---")

    def gravityWrench(self, T_sb):
        mg_s = np.array([0, 0, self.m * self.g])
        grav_s = wrench(np.cross(T_sb.p.reshape((1, -1)), mg_s), mg_s)
        grav_b = grav_s.convertFrame(T_sb)
        return grav_b

    def solveFowardDyamic(self, t0, tf, init_T, init_V, dt=0.001):
        ts = np.arange(t0 + dt, tf, dt)
        T_sb = init_T  # Init Transform
        V_b = init_V  # Body twist
        traj = [[t0, T_sb, V_b, None]]

        R_k = np.eye(3)

        for t in ts:
            # wrench in body frame
            r_ = R.from_matrix(T_sb.Mat[:3, :3])
            # print("Euler, ", r_.as_euler("zyx"))

            tau = self.gravityWrench(T_sb)
            # body twist rate
            Vq = (
                np.block(
                    [
                        [skewsym_mat(V_b.w * dt), np.zeros((3, 3))],
                        [np.zeros((3, 3)), skewsym_mat(V_b.w * dt)],
                    ]
                )
                @ self.Mq
            )

            # dVdt = np.linalg.inv(self.Mq) @ (tau.Mat)
            dVdt = np.linalg.inv(self.Mq) @ (tau.Mat + Vq @ V_b.Mat)

            # Transform twist vector from previous frame to current frame
            V_b.v = np.transpose(R_k[:3, :3]) @ V_b.v
            V_b.w = np.transpose(R_k[:3, :3]) @ V_b.w
            V_b = V_b.integrate(dVdt[3:], dVdt[:3], dt)
            V_s = r_.as_matrix() @ V_b.Mat[3:]

            # Transform to Base position to next world frame
            rot = R.from_rotvec(V_b.w.reshape(3) * dt)
            R_k = np.block(
                [
                    [
                        rot.as_matrix(),
                        np.array([[0], [0], [0]]) * dt,
                    ],
                    [np.array([0, 0, 0, 1])],
                ]
            )
            T_rot = T_sb.Mat @ R_k
            P_ = np.block(
                [
                    [np.eye(3), np.transpose(R_k[:3, :3]) @ V_b.Mat[3:] * dt],
                    [np.array([0, 0, 0, 1])],
                ]
            )
            T_trans = T_rot @ P_
            T_sb = HomoTransform(T_trans[:3, :3], T_trans[:3, 3])

            traj.append([t, T_sb, V_b, dVdt])

        return traj


def aniRun(i, *fargs):
    ax = fargs[0]
    ax.clear()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    traj = fargs[1]
    k = i * 20
    if k < len(traj):
        traj[k][1].plot(ax, scale=1)
    pass


def animation(cv, traj, stepfactor=1, timefactor=1):
    cnt = 0
    dcnt = int(16 * stepfactor)
    dt = 0.016 / timefactor
    while True:
        R_ = traj[cnt][1].R
        P_ = traj[cnt][1].p
        cv.update_bodytf(R_, P_)
        cnt += dcnt
        if cnt >= len(traj):
            cnt = 0
        time.sleep(dt)


if __name__ == "__main__":
    print("Full Body Dynamic")

    rot = R.from_euler("zyx", [0, 0, 0], degrees=True)
    R0 = rot.as_matrix()
    p0 = np.array([0, 0, 0.2])
    v0 = twist(np.array([5, 0, 5]), np.array([0, 5, 0]))

    robot = FloatingBase(p0, R0, v0)
    start = time.time()
    traj = robot.solveFowardDyamic(0, 1, robot.T_sb, robot.V_b, 0.001)

    print("time elapsed: ", time.time() - start)
    print("steps: ", len(traj))

    cv = vs.corgiVisualize()
    t = threading.Thread(target=animation, args=(cv, traj, 0.1, 1))
    t.start()
    vs.app.run()
    cv.writer.close()
