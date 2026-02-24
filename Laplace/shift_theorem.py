import numpy as np
import plotly.graph_objects as go

a = 1.5

sigma = np.linspace(-3.5, 3.5, 220)
omega = np.linspace(-12, 12, 240)
SIG, OMG = np.meshgrid(sigma, omega)
s = SIG + 1j * OMG

zmin, zmax = -40, 25

def mag_db(F):
    m = 20 * np.log10(np.clip(np.abs(F), 1e-9, 1e9))
    return np.clip(m, zmin, zmax)

def base_figure(surface_z, colorbar_title):
    z_floor = zmin * np.ones_like(surface_z)

    colorscale = [
        [0.0, "#00ff66"],
        [1.0, "#ff1a1a"],
    ]

    surface = go.Surface(
        x=SIG, y=OMG, z=surface_z,
        colorscale=colorscale,
        cmin=zmin, cmax=zmax,
        opacity=0.82,
        showscale=True,
        colorbar=dict(
            title=colorbar_title,
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
        x=[sigma.min(), sigma.max()], y=[0, 0], z=[zmin, zmin],
        mode="lines",
        line=dict(color="white", width=4),
        showlegend=False
    )

    y_axis_line = go.Scatter3d(
        x=[0, 0], y=[omega.min(), omega.max()], z=[zmin, zmin],
        mode="lines",
        line=dict(color="white", width=4),
        showlegend=False
    )

    pole = go.Scatter3d(
        x=[a], y=[0], z=[zmin],
        mode="markers+text",
        marker=dict(size=6, color="white", symbol="x"),
        text=[f"pole: s={a:g}"],
        textposition="top center",
        showlegend=False
    )

    fig = go.Figure(data=[floor, surface, x_axis_line, y_axis_line, pole])

    fig.update_layout(
        template="plotly_dark",
        width=1100,
        height=700,
        margin=dict(l=10, r=10, t=55, b=10),
        scene=dict(
            xaxis=dict(title="실수부 σ", showbackground=False, zeroline=False, gridcolor="rgba(255,255,255,0.15)"),
            yaxis=dict(title="허수부 ω", showbackground=False, zeroline=False, gridcolor="rgba(255,255,255,0.15)"),
            zaxis=dict(title="", showbackground=False, showticklabels=False, zeroline=False, gridcolor="rgba(255,255,255,0.0)"),
            camera=dict(eye=dict(x=1.45, y=-1.55, z=0.75)),
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.2, z=0.6),
        ),
        showlegend=False
    )

    return fig

def show_s_shift():
    alpha_vals = np.linspace(-2.0, 2.0, 41)

    F0 = 1.0 / (s - a)
    fig = base_figure(mag_db(F0), "|F(s)| (dB)")

    frames = []
    for alpha in alpha_vals:
        pole_new = a - alpha
        F_shift = 1.0 / (s - pole_new)
        z = mag_db(F_shift)

        frames.append(
            go.Frame(
                name=f"a={alpha:.2f}",
                data=[
                    go.Surface(z=z),
                    go.Scatter3d(
                        x=[pole_new], y=[0], z=[zmin],
                        mode="markers+text",
                        marker=dict(size=6, color="white", symbol="x"),
                        text=[f"pole: s={pole_new:g}"],
                        textposition="top center",
                        showlegend=False
                    )
                ],
                traces=[1, 4]
            )
        )

    fig.frames = frames

    steps = []
    for alpha in alpha_vals:
        steps.append(
            dict(
                method="animate",
                args=[
                    [f"a={alpha:.2f}"],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
                ],
                label=f"{alpha:.2f}"
            )
        )

    fig.update_layout(
        title="s-shift:  f(t)e^{-αt}  ↔  F(s+α)  (pole: a → a-α)",
        sliders=[dict(
            active=len(alpha_vals)//2,
            currentvalue=dict(prefix="α = "),
            pad=dict(t=35),
            steps=steps
        )]
    )

    fig.show()

def show_t_shift():
    tau_vals = np.linspace(0.0, 1.2, 31)

    F0 = 1.0 / (s - a)
    fig = base_figure(mag_db(F0), "|e^{-τs}F(s)| (dB)")

    frames = []
    for tau in tau_vals:
        F_shift = np.exp(-tau * s) * F0
        z = mag_db(F_shift)

        frames.append(
            go.Frame(
                name=f"t={tau:.2f}",
                data=[go.Surface(z=z)],
                traces=[1]
            )
        )

    fig.frames = frames

    steps = []
    for tau in tau_vals:
        steps.append(
            dict(
                method="animate",
                args=[
                    [f"t={tau:.2f}"],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
                ],
                label=f"{tau:.2f}"
            )
        )

    fig.update_layout(
        title="t-shift:  f(t-τ)u(t-τ)  ↔  e^{-τs}F(s)  (|·|은 σ방향으로 e^{-τσ} 스케일)",
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="τ = "),
            pad=dict(t=35),
            steps=steps
        )]
    )

    fig.show()

if __name__ == "__main__":
    show_s_shift()
    show_t_shift()