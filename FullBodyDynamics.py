import numpy as np
from scipy.spatial.transform import Rotation as R

import time
import os

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# mpl.rcParams["figure.dpi"] = 150


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
        self.w = 0.1
        self.h = 0.1
        self.l = 0.2
        self.Ixx = 1 / 12 * self.m * (self.w**2 + self.h**2)
        self.Iyy = 1 / 12 * self.m * (self.l**2 + self.h**2)
        self.Izz = 1 / 12 * self.m * (self.l**2 + self.w**2)

        self.Ib = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.Mb = np.diag([self.m, self.m, self.m])
        self.Mq = np.block([[self.Ib, np.zeros((3, 3))], [np.zeros((3, 3)), self.Mb]])
        self.Vq = self.VqMat(init_vb)

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

    def VqMat(self, twist):
        return np.transpose(twist.adjoint()) @ self.Mq

    def gravityWrench(self, T_sb):
        mg_s = np.array([0, 0, self.m * self.g])
        grav_s = wrench(np.cross(T_sb.p.reshape((1, -1)), mg_s), mg_s)
        # print("grav_s\n", grav_s.Mat.reshape(6))
        grav_b = grav_s.convertFrame(T_sb)
        # print("grav_b\n", grav_b.Mat.reshape(6))
        return grav_b

    def solveFowardDyamic(self, t0, tf, init_T, init_V, dt=0.001):
        ts = np.arange(t0 + dt, tf, dt)
        T_sb = init_T  # Init Transform
        V_b = init_V  # Body twist
        traj = [[t0, T_sb, V_b, None]]

        fig = plt.figure(figsize=(15, 15), dpi=90)
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-0.005, 0.005)
        ax.set_ylim(-0.005, 0.005)
        ax.set_zlim(-0.005, 0.005)
        ax.view_init(elev=0, azim=-90, roll=0)
        ax.set_proj_type("ortho")
        T_sb.plot(ax)
        R_acc_ = np.eye(4)

        for t in ts:
            # wrench in body frame
            tau = self.gravityWrench(T_sb)

            # body twist rate
            dVdt = np.linalg.inv(self.Mq) @ (tau.Mat - self.VqMat(V_b) @ V_b.Mat)

            V_b = V_b.integrate(dVdt[3:], dVdt[:3], dt)

            T_bn = HomoTransform(np.eye(3), np.array([0, 0, 0]))
            T_bn = T_bn.integrate(V_b, dt)
            rot = R.from_rotvec(V_b.Mat[:3].reshape(3) * dt)

            T_bn = np.block(
                [
                    [
                        np.transpose(rot.as_matrix()),
                        np.transpose(rot.as_matrix()) @ V_b.Mat[3:] * dt,
                    ],
                    [np.array([0, 0, 0, 1])],
                ]
            )
            T_bn = HomoTransform(T_bn[:3, :3], T_bn[:3, 3]).inverse()

            R_ = np.block(
                [
                    [
                        rot.as_matrix(),
                        np.array([[0], [0], [0]]) * dt,
                    ],
                    [np.array([0, 0, 0, 1])],
                ]
            )

            P = np.block(
                [
                    [np.eye(3), V_b.Mat[3:] * dt],
                    [np.array([0, 0, 0, 1])],
                ]
            )

            R_acc_ = R_acc_ @ R_

            # T_trans = T_sb.Mat @ P
            # HomoTransform(T_trans[:3, :3], T_trans[:3, 3]).plot(ax)
            # T_rot = T_trans @ R_
            # HomoTransform(T_rot[:3, :3], T_rot[:3, 3]).plot(ax)
            # T_sb = HomoTransform(T_rot[:3, :3], T_rot[:3, 3])

            T_rot = T_sb.Mat @ R_
            # HomoTransform(T_rot[:3, :3], T_rot[:3, 3]).plot(ax, scale=0.001)

            P_ = np.block(
                [
                    [np.eye(3), np.transpose(T_rot[:3, :3]) @ V_b.Mat[3:] * dt],
                    [np.array([0, 0, 0, 1])],
                ]
            )

            T_trans = T_rot @ P_
            # HomoTransform(T_trans[:3, :3], T_trans[:3, 3]).plot(ax, scale=0.001)
            T_sb = HomoTransform(T_trans[:3, :3], T_trans[:3, 3])

            # T_sn = T_sb.Mat @ T_bn.inverse().Mat
            T_sn = T_sb.Mat @ T_bn.Mat
            T_sb = HomoTransform(T_sn[:3, :3], T_sn[:3, 3])

            """
            T_sn = T_sb.Mat @ R_ @ P

            # T_sn = T_bn @ T_sb.Mat
            T_sb = HomoTransform(T_sn[:3, :3], T_sn[:3, 3])
            """
            # T_sb.plot(ax)
            # T_sb = T_sb.integrate(V_b, dt)
            # T_sb = T_sb.integrate(V_b.convertFrame(T_sb), dt)

            traj.append([t, T_sb, V_b, dVdt])
            print("tau")
            print(tau.Mat.reshape(6))
            # print("dVdt")
            # print(dVdt.reshape(6))
            # print("body twist")
            # print(V_b.Mat.reshape(6))
            # print("spatial twist")
            # print(V_b.convertFrame(T_sb).Mat.reshape(6))
            # print("T_sb")
            # print(T_sb.Mat)
            # print("T_bn")
            # # print(T_bn.Mat)
            print("rotation")
            print(V_b.Mat[:3].reshape(3) * dt)
            print("translation")
            print(np.transpose(rot.as_matrix()) @ V_b.Mat[3:] * dt)
            # print(V_b.Mat[3:].reshape(3) * dt)
            print("T_sb")
            # print(T_sb.Mat)
            print("--")

            """
            print("t:", t)
            print("dvdt: ")
            print(dVdt.reshape((1, -1)))
            print("V_b: ")
            print(V_b.Mat.reshape((1, -1)))
            print("T_sb: ")
            print(T_sb.Mat)
            print("==")
            """

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


if __name__ == "__main__":
    print("Full Body Dynamic")

    # os.nice(-19)
    rot = R.from_euler("zyx", [0, 0, 0], degrees=True)
    # rot = R.from_rotvec(np.array([0, 30, 0]), degrees=True)

    # R0 = np.eye(3)
    R0 = rot.as_matrix()
    p0 = np.array([0, 0, 0])
    v0 = twist(np.array([0, 0, 0]), np.array([0, 10, 0]))

    robot = FloatingBase(p0, R0, v0)
    start = time.time()
    traj = robot.solveFowardDyamic(0, 0.5, robot.T_sb, robot.V_b, 0.001)
    # traj = robot.solveFowardDyamic(0, 0.5, robot.T_sb, robot.V_b, 0.01)
    plt.show()

    print("time elapsed: ", time.time() - start)
    print("steps: ", len(traj))

    # animation
    if len(traj) > 20:
        fig = plt.figure(figsize=(15, 15), dpi=90)
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-0.5, 5)
        ax.set_ylim(-0.5, 5)
        ax.set_zlim(-0.5, 5)

        ani = animation.FuncAnimation(
            fig, aniRun, frames=int(traj[-1][0] / 0.05), interval=20, fargs=(ax, traj)
        )

        plt.show()
        ani.save("animation.gif", fps=50)  # 儲存為 gif
