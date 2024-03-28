import numpy as np
from vispy import app, scene
from vispy.visuals import transforms

canvas = scene.SceneCanvas(
    keys="interactive", bgcolor="black", show=True, app="pyqt5", vsync=True
)
# canvas = scene.SceneCanvas(keys="interactive", bgcolor="gray", app="pyqt5", vsync=True)


vb = canvas.central_widget.add_view()
vb.camera = "turntable"
vb.camera.rect = (-10, -10, 20, 20)


box = scene.visuals.Box(
    width=1, height=2, depth=3, color=(1, 1, 1, 0.5), edge_color="green"
)
axis = scene.visuals.XYZAxis(width=5)
axis1 = scene.visuals.XYZAxis(width=5)
grid = scene.visuals.GridLines(color=(0, 1, 0, 1))

vb.add(axis)
vb.add(axis1)
vb.add(box)
vb.add(grid)


# Define a scale and translate transformation :
box.transform = transforms.STTransform(translate=(3.0, 3.0, 0.0), scale=(1.0, 1.0, 1.0))
axis.transform = transforms.STTransform(
    translate=(0.0, 0.0, 0.0), scale=(1000.0, 1000.0, 1000.0)
)
axis1.transform = transforms.STTransform(
    translate=(3.0, 3.0, 0.0), scale=(3.0, 3.0, 3.0)
)


@canvas.events.key_press.connect
def on_key_press(ev):
    tr = np.array(box.transform.translate)
    sc = np.array(box.transform.scale)
    if ev.text in "+":
        tr[0] += 0.1
    elif ev.text == "-":
        tr[0] -= 0.1
    elif ev.text == "(":
        sc[0] += 0.1
    elif ev.text == ")":
        sc[0] -= 0.1
    box.transform.translate = tr
    box.transform.scale = sc
    # print("Translate (x, y, z): ", list(tr), "\nScale (x, y, z): ", list(sc), "\n")


t = 0


def update(ev):
    global t
    z = np.sin(t)
    box.transform = transforms.STTransform(
        translate=(3.0, 3.0, z), scale=(1.0, 1.0, 1.0)
    )
    axis1.transform = transforms.STTransform(
        translate=(3.0, 3.0, z), scale=(3.0, 3.0, 3.0)
    )
    t += 0.016


timer = app.Timer(interval=16)
timer.connect(update)
timer.start(0)

if __name__ == "__main__":
    import sys

    if sys.flags.interactive != 1:
        app.run()
