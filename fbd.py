import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class fullbody:
    def __init__(self):
        m = 21
        w = 0.1
        h = 0.1
        l = 0.2
        Ixx = 1/12 * m * (w**2 + h**2)
        Iyy = 1/12 * m * (l**2 + h**2)
        Izz = 1/12 * m * (l**2 + w**2)
        Ib = np.diag([Ixx, Iyy, Izz])
        Mb = np.diag([m, m, m])

        self.Mq = np.vstack(( np.hstack((Ib,np.zeros((3,3)))), np.hstack((np.zeros((3,3)), Mb))))
        self.de_ts = np.linspace(0,10,1000)
        
        fig = plt.figure(figsize=(15,15), dpi=200)
        self.ax = fig.add_subplot(projection='3d')
        

    def asymMat(self, x):
        x1, x2, x3 = x
        return np.array([[0, -x3, x2],
                         [x3, 0, -x1],
                         [-x2, x1, 0]])

    def VqMat(self, twist):
        # twist = [wb, vb]
        wb = twist[0:3]
        vb = twist[3:]
        r1 = np.hstack((self.asymMat(wb), np.zeros((3,3))))
        r2 = np.hstack((np.zeros((3,3)), self.asymMat(vb)))
        v_ = np.vstack((r1, r2))
        Vq = v_ @ self.Mq @ twist
        Vq = np.reshape(Vq, (6,1))
        return Vq

    def GqMat(self):
        g = np.array([0,0,0,0,0,-9.81])
        g = np.reshape(g, (6,1))
        return self.Mq @ g
    
    def Tauq(self, t):
        return np.array([0,0,0,0,0,20*9.81])
    
    def DE(self, t, y):
        tau = np.array([0, 0, 1, 0, 0, 20*9.81])
        # tau = np.array([0,0,0,0,0,0])
        tau = np.reshape(tau, (6,1))
        dqdt = np.linalg.inv(self.Mq) @ (tau - self.VqMat(y) + self.GqMat())
        dqdt = np.reshape(dqdt, (1,6))[0]
        print("t, ", t)
        print(dqdt)
        return dqdt
    
    def solveDE(self):
        y0 = np.array([0,0,0,0,0,0])
        t = np.linspace(0,10,10000)
        sol = solve_ivp(self.DE, [0,10], y0.flatten(), t_eval=np.linspace(0, 10, 10000))
        
        xtrans0 = np.array([0,0,0])
        xorient0 = np.eye(3)
        
        xtrans_t, xrot_t = self.integralTrajectory(xtrans0, xorient0, sol.y)
        self.visualize(xtrans_t, xrot_t)
        
        
    def integralTrajectory(self, Xtrans_0, Xrot_0, twist_t, dt=0.001):
        wx = twist_t[0, :]
        wy = twist_t[1, :]
        wz = twist_t[2, :]
        vx = twist_t[3, :]
        vy = twist_t[4, :]
        vz = twist_t[5, :]
        vb = np.vstack((vx,vy,vz))
        wb = np.vstack((wx,wy,wz))
        Xtrans_0 = Xtrans_0.reshape((-1,1))
        Xtrans_t = Xtrans_0 + np.cumsum(dt * vb, axis=1)
        
        Xrot_t = []
        Xrot_ = Xrot_0
        
        iter_ = twist_t.shape[1]
        for i in range(iter_):
            w = wb[:, i]
            rot_ = R.from_rotvec(w*dt)
            Xrot_ = rot_.as_matrix() @ Xrot_
            Xrot_t.append(Xrot_)
        
        return [Xtrans_t, Xrot_t]
    
    
    def plotFrame(self, xtrans, xrot, scale=0.1):
        vx = xrot[:,0] * scale
        vy = xrot[:,1] * scale
        vz = xrot[:,2] * scale
        ax = Arrow3D([xtrans[0], xtrans[0]+vx[0]], [xtrans[1], xtrans[1]+vx[1]], [xtrans[2], xtrans[2]+vx[2]], mutation_scale=3, lw=1, arrowstyle="-|>", color="r")
        ay = Arrow3D([xtrans[0], xtrans[0]+vy[0]], [xtrans[1], xtrans[1]+vy[1]], [xtrans[2], xtrans[2]+vy[2]], mutation_scale=3, lw=1, arrowstyle="-|>", color="g")
        az = Arrow3D([xtrans[0], xtrans[0]+vz[0]], [xtrans[1], xtrans[1]+vz[1]], [xtrans[2], xtrans[2]+vz[2]], mutation_scale=3, lw=1, arrowstyle="-|>", color="b")
        self.ax.add_artist(ax)
        self.ax.add_artist(ay)
        self.ax.add_artist(az)
        self.ax.scatter([xtrans[0]], [xtrans[1]], [xtrans[2]])
        
    def visualize(self, xtrans_t, xrot_t):
        for i in range(xtrans_t.shape[1]):
            if i%200 == 0:
                xtrans = xtrans_t[:,i]
                xrot = xrot_t[i]
                self.plotFrame(xtrans, xrot)
                print("i, xtrans: ", i, xtrans)
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
        plt.show()
        
if __name__ == "__main__":
    fbd = fullbody()
    fbd.solveDE()
    

