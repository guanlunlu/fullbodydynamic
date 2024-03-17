import numpy as np
import threading
from vispy import app, scene
from vispy.visuals import transforms
from scipy.spatial.transform import Rotation as R
import time
import imageio


class corgiVisualize:
    def __init__(
        self,
        init_pose=np.array([0, 0, 0.2]),
        init_zyx=np.array([0, 0, 0]),
        interval_ms=16,
    ):
        self.canvas = scene.SceneCanvas(
            title="Corgi Full Body Dynamic",
            keys="interactive",
            size=(1920, 1080),
            bgcolor="#d8e6df",
            show=True,
            app="pyqt5",
            vsync=True,
        )

        # setup camera
        self.vb = self.canvas.central_widget.add_view()
        self.vb.camera = "turntable"
        self.vb.camera.rect = (-20, -20, 20, 20)
        self.vb.camera.azimuth = 15

        self.dim_w = 577.5 * 0.001
        self.dim_l = 329.5 * 0.001
        self.dim_h = 144 * 0.001
        self.body = scene.visuals.Box(
            width=self.dim_w,
            height=self.dim_h,
            depth=self.dim_l,
            color=("#d9ffa1"),
            edge_color="green",
        )
        self.body_axis = scene.visuals.XYZAxis(width=5)
        self.world_axis = scene.visuals.XYZAxis(width=5)
        self.grid = scene.visuals.GridLines(color=("#011c09"))
        self.grid.set_gl_state("translucent", cull_face=False)

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.surface = scene.visuals.SurfacePlot(
            x=X, y=Y, z=Z, shading="smooth", color=("#d4ffe0")
        )

        self.vb.add(self.body_axis)
        self.vb.add(self.body)
        self.vb.add(self.world_axis)
        self.vb.add(self.surface)
        self.vb.add(self.grid)

        self.world_axis.transform = transforms.STTransform(
            translate=(0.0, 0.0, 0.0), scale=(1000.0, 1000.0, 1000.0)
        )

        # set body to initial configuration (w.r.t. fixed frame)
        rot = R.from_euler("zyx", init_zyx)
        rot_vec = rot.as_rotvec(degrees=True)
        rot_angle = np.linalg.norm(rot_vec)
        if rot_angle != 0:
            rot_vec = rot_vec / rot_angle
        else:
            rot_vec = np.array([0, 0, 1])

        self.angle = 0
        self.body_angle = rot_angle
        self.body_rotvec = rot_vec
        self.body_translation = init_pose

        axis_transform = scene.MatrixTransform()
        axis_transform.rotate(rot_angle, rot_vec)  # Rotate around the y-axis
        axis_transform.translate(init_pose)
        self.body_axis.transform = axis_transform
        self.body.transform = axis_transform

        # animation setup
        self.timer = app.Timer(interval=interval_ms)
        self.timer.connect(self.update)
        self.timer.start(0)

        # writer
        self.writer = imageio.get_writer("viz.mp4", fps=60, quality=9)
        # self.writer = imageio.get_writer("viz.gif", fps=60, quality=9)

    def update_bodytf(self, R_, P_):
        rot = R.from_matrix(R_)
        rot_vec = rot.as_rotvec(degrees=True)
        rot_angle = np.linalg.norm(rot_vec)
        if rot_angle != 0:
            rot_vec = rot_vec / rot_angle
        else:
            rot_vec = np.array([0, 0, 1])

        self.body_angle = rot_angle
        self.body_rotvec = rot_vec
        self.body_translation = P_.reshape(3)

    def update(self, ev):
        body_tf = scene.MatrixTransform()
        body_tf.rotate(self.body_angle, self.body_rotvec)  # Rotate around the y-axis
        body_tf.translate(self.body_translation)
        self.body.transform = body_tf

        axis_tf = scene.MatrixTransform()
        axis_tf.scale((0.4, 0.4, 0.4))
        axis_tf.rotate(self.body_angle, self.body_rotvec)  # Rotate around the y-axis
        axis_tf.translate(self.body_translation)
        self.body_axis.transform = axis_tf

        im = self.canvas.render(alpha=True, size=(1920, 1080))
        self.writer.append_data(im)


def MoveRobot(cv):
    t = 0
    while True:
        z = np.sin(t)
        cv.update_bodytf(np.eye(3), np.array([0, 0, z]))
        time.sleep(0.016)
        t += 0.016


if __name__ == "__main__":
    import sys

    if sys.flags.interactive != 1:
        cv = corgiVisualize(init_pose=np.array([0, 0, 1]), init_zyx=np.array([0, 0, 0]))
        t = threading.Thread(target=MoveRobot, args=(cv,))
        t.start()
        app.run()
