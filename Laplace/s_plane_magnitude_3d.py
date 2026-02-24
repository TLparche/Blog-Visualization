import numpy as np
import plotly.graph_objects as go

a = 1.5

sigma = np.linspace(-3.5, 3.5, 220)
omega = np.linspace(-12, 12, 240)
SIG, OMG = np.meshgrid(sigma, omega)
s = SIG + 1j * OMG

F = 1.0 / (s - a)
mag_db = 20 * np.log10(np.clip(np.abs(F), 1e-9, 1e9))
mag_db = np.clip(mag_db, -40, 25)

z_floor = -40 * np.ones_like(mag_db)

colorscale = [
    [0.0, "#00ff66"],
    [1.0, "#ff1a1a"],
]

surface = go.Surface(
    x=SIG, y=OMG, z=mag_db,
    colorscale=colorscale,
    cmin=-40, cmax=25,
    opacity=0.82,
    showscale=True,
    colorbar=dict(
        title="|F(s)| (dB)",
        thickness=18
    )
)

floor = go.Surface(
    x=SIG, y=OMG, z=z_floor,
    showscale=False,
    opacity=0.10,
    colorscale=[[0, "cyan"], [1, "cyan"]],
    showlegend=False
)

x_axis_line = go.Scatter3d(
    x=[sigma.min(), sigma.max()], y=[0, 0], z=[-40, -40],
    mode="lines",
    line=dict(color="white", width=4),
    name="ω=0",
    showlegend=False
)

y_axis_line = go.Scatter3d(
    x=[0, 0], y=[omega.min(), omega.max()], z=[-40, -40],
    mode="lines",
    line=dict(color="white", width=4),
    name="σ=0",
    showlegend=False
)

pole = go.Scatter3d(
    x=[a], y=[0], z=[-40],
    mode="markers+text",
    marker=dict(size=6, color="white", symbol="x"),
    text=[f"pole: s={a:g}"],
    textposition="top center",
    name="pole",
    showlegend=False
)

fig = go.Figure(data=[floor, surface, x_axis_line, y_axis_line, pole])

fig.update_layout(
    template="plotly_dark",
    width=1100,
    height=700,
    margin=dict(l=10, r=10, t=50, b=10),
    scene=dict(
        xaxis=dict(title="실수부 σ", showbackground=False, zeroline=False, gridcolor="rgba(255,255,255,0.15)"),
        yaxis=dict(title="허수부 ω", showbackground=False, zeroline=False, gridcolor="rgba(255,255,255,0.15)"),
        zaxis=dict(title="", showbackground=False, showticklabels=False, zeroline=False, gridcolor="rgba(255,255,255,0.0)"),
        camera=dict(
            eye=dict(x=1.45, y=-1.55, z=0.75)
        ),
        aspectmode="manual",
        aspectratio=dict(x=1.6, y=1.2, z=0.6),
    )
)

fig.show()