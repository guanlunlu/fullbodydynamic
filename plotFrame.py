import plotly.graph_objs as go
import plotly.express as px
import numpy as np


def plotFrame(
    xtrans, xrot, arrowlen=0.05, arrow_tip_ratio=0.5, arrow_starting_ratio=0.98
):
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t

    p1 = go.Scatter3d(
        x=[xtrans[0]],
        y=[xtrans[1]],
        z=[xtrans[2]],
        mode="markers",
        marker=dict(color="aqua"),
    )

    vx = xrot[:, 0]
    vy = xrot[:, 1]
    vz = xrot[:, 2]
    vx_x = np.linspace(xtrans[0], xtrans[0] + arrowlen * vx[0], 2)
    vx_y = np.linspace(xtrans[1], xtrans[1] + arrowlen * vx[1], 2)
    vx_z = np.linspace(xtrans[2], xtrans[2] + arrowlen * vx[2], 2)

    vy_x = np.linspace(xtrans[0], xtrans[0] + arrowlen * vy[0], 2)
    vy_y = np.linspace(xtrans[1], xtrans[1] + arrowlen * vy[1], 2)
    vy_z = np.linspace(xtrans[2], xtrans[2] + arrowlen * vy[2], 2)

    vz_x = np.linspace(xtrans[0], xtrans[0] + arrowlen * vz[0], 2)
    vz_y = np.linspace(xtrans[1], xtrans[1] + arrowlen * vz[1], 2)
    vz_z = np.linspace(xtrans[2], xtrans[2] + arrowlen * vz[2], 2)

    lx = go.Scatter3d(
        x=vx_x,
        y=vx_y,
        z=vx_z,
        mode="lines",
        line=dict(color="red", width=5),
        showlegend=False,
    )
    ly = go.Scatter3d(
        x=vy_x,
        y=vy_y,
        z=vy_z,
        mode="lines",
        line=dict(color="green", width=5),
        showlegend=False,
    )
    lz = go.Scatter3d(
        x=vz_x,
        y=vz_y,
        z=vz_z,
        mode="lines",
        line=dict(color="blue", width=5),
        showlegend=False,
    )

    arrow_tip_ratio = 0.3
    arrow_starting_ratio = 0.98

    ax = go.Cone(
        x=[xtrans[0] + arrow_starting_ratio * vx[0] * arrowlen],
        y=[xtrans[1] + arrow_starting_ratio * vx[1] * arrowlen],
        z=[xtrans[2] + arrow_starting_ratio * vx[2] * arrowlen],
        u=[arrow_tip_ratio * vx[0] * arrowlen],
        v=[arrow_tip_ratio * vx[1] * arrowlen],
        w=[arrow_tip_ratio * vx[2] * arrowlen],
        showlegend=False,
        showscale=False,
        colorscale=[[0, "rgb(255,0,0)"], [1, "rgb(255,0,0)"]],
    )

    ay = go.Cone(
        x=[xtrans[0] + arrow_starting_ratio * vy[0] * arrowlen],
        y=[xtrans[1] + arrow_starting_ratio * vy[1] * arrowlen],
        z=[xtrans[2] + arrow_starting_ratio * vy[2] * arrowlen],
        u=[arrow_tip_ratio * vy[0] * arrowlen],
        v=[arrow_tip_ratio * vy[1] * arrowlen],
        w=[arrow_tip_ratio * vy[2] * arrowlen],
        showlegend=False,
        showscale=False,
        colorscale=[[0, "rgb(0,255,0)"], [1, "rgb(0,255,0)"]],
    )

    az = go.Cone(
        x=[xtrans[0] + arrow_starting_ratio * vz[0] * arrowlen],
        y=[xtrans[1] + arrow_starting_ratio * vz[1] * arrowlen],
        z=[xtrans[2] + arrow_starting_ratio * vz[2] * arrowlen],
        u=[arrow_tip_ratio * vz[0] * arrowlen],
        v=[arrow_tip_ratio * vz[1] * arrowlen],
        w=[arrow_tip_ratio * vz[2] * arrowlen],
        showlegend=False,
        showscale=False,
        colorscale=[[0, "rgb(0,0,255)"], [1, "rgb(0,0,255)"]],
    )

    return [p1, lx, ly, lz, ax, ay, az]


def plotGround(height=0):
    x = np.linspace(-10, 10, 2)
    y = np.linspace(-10, 10, 2)
    z = height * np.ones((2, 2))
    return go.Surface(
        z=z,
        x=x,
        y=y,
        autocolorscale=False,
        showscale=False,
        showlegend=False,
        colorscale=[[0, "rgb(102, 102, 102)"], [1, "rgb(102, 102, 102)"]],
        opacity=0.3,
    )


if __name__ == "__main__":
    t0 = plotFrame(np.array([0, 0, 0.0]), np.eye(3), arrowlen=0.2)
    t1 = plotFrame(np.array([0, 0, 0.3]), np.eye(3), arrowlen=0.2)
    t2 = plotFrame(np.array([1, 0, 0.3]), np.eye(3), arrowlen=0.2)
    t3 = plotGround(0)

    fig = go.Figure(data=t0)
    # fig.add_traces(data=t1)
    # fig.add_traces(data=t2)
    fig.add_traces(data=t3)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-0, 2],
            ),
            yaxis=dict(
                range=[-1, 1],
            ),
            zaxis=dict(
                range=[-0.01, 1.99],
            ),
        ),
        margin=dict(r=20, l=10, b=10, t=10),
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=3, y=-3, z=1.25),
    )

    updatemenus = [
        dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])],
        )
    ]

    N = 100
    xs = np.linspace(0, 1, N)

    # fig_ani = go.Figure(
    #     data=[t0, t3],
    #     layout=go.Layout(
    #         xaxis=dict(range=[-0, 2], autorange=False, zeroline=False),
    #         yaxis=dict(range=[-1, 1], autorange=False, zeroline=False),
    #         title_text="Corgi Motion",
    #         hovermode="closest",
    #         updatemenus=[
    #             dict(
    #                 type="buttons",
    #                 buttons=[dict(label="Play", method="animate", args=[None])],
    #             )
    #         ],
    #     ),
    #     frames=[
    #         go.Frame(
    #             data=go.Scatter3d(
    #                 x=[xs[k]],
    #                 y=[0],
    #                 z=[0],
    #                 mode="markers",
    #                 marker=dict(color="red", size=10),
    #             )
    #             # data=plotFrame(np.array([xs[k], 0, 0.2]), np.eye(3), arrowlen=0.2)[0]
    #         )
    #         for k in range(N)
    #     ],
    # )

    frames = [
        go.Frame(data=plotFrame(np.array([xs[k], 0, 0.2]), np.eye(3), arrowlen=0.2))
        for k in range(N)
    ]
    fig.update(frames=frames)

    # fig.update_layout(scene_camera=camera)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    redraw=True,
                                    fromcurrent=True,
                                    mode="immediate",
                                    frame={"duration": 0.1, "redraw": False},
                                )
                            ),
                        ],
                    )
                ],
            )
        ]
    )

    fig.show()
