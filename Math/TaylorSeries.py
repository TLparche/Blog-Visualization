'''
실행방법
터미널에서 이 디렉토리로 오신 뒤에
python .\TaylorSeries.py
실행하시면 댑니다.

변수 변경하시고 버튼 누르신다음에 refresh 누르셔야 그래프 그려져용
'''

import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy import latex

DEFAULT_XRANGE = (-5.0, 5.0)
DEFAULT_YRANGE = (-2.0, 2.0)
DEFAULT_A = 0.0
POINTS = 1200
MAX_N = 40

x = sp.Symbol("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

SAFE_LOCALS = {
    "x": x,
    "pi": sp.pi,
    "E": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
}

_RANGE_RE = re.compile(
    r"^\s*[\(\[\{]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*[, ]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*[\)\]\}]?\s*$"
)

def parse_range(text: str, name: str):
    m = _RANGE_RE.match(text)
    if not m:
        raise ValueError(f'{name} format: (min,max)')
    lo, hi = float(m.group(1)), float(m.group(2))
    if lo >= hi:
        raise ValueError(f"{name}: min < max required")
    return lo, hi

def parse_float(text: str, name: str):
    try:
        return float(text.strip())
    except Exception:
        raise ValueError(f"{name} must be a number")

def parse_function(s: str):
    if not s.strip():
        raise ValueError("f(x) is empty")
    expr = parse_expr(s, local_dict=SAFE_LOCALS, transformations=TRANSFORMS, evaluate=True)
    return sp.simplify(expr)

def derivs_at(expr, a, max_n):
    out = []
    for k in range(max_n + 1):
        try:
            out.append(float(sp.N(sp.diff(expr, x, k).subs(x, a))))
        except Exception:
            out.append(float("nan"))
    return out

def taylor_eval(xs, a, d, n):
    y = np.zeros_like(xs, dtype=float)
    dx = xs - a
    p = np.ones_like(xs)
    fact = 1.0
    for k in range(n + 1):
        if k > 0:
            fact *= k
        if not np.isfinite(d[k]):
            return np.full_like(xs, np.nan)
        y += (d[k] / fact) * p
        p *= dx
    return y

def safe_eval(f, xs):
    try:
        y = f(xs).astype(float)
        y[~np.isfinite(y)] = np.nan
        return y
    except Exception:
        return np.full_like(xs, np.nan)

def taylor_expr(expr: sp.Expr, a: float, n: int) -> sp.Expr:
    series = 0
    for k in range(n + 1):
        series += sp.diff(expr, x, k).subs(x, a) / sp.factorial(k) * (x - a) ** k
    return series

class TaylorUI:
    def __init__(self):
        self.expr_str = "sin(x)"
        self.expr = None
        self.f_num = None

        self.xrange = DEFAULT_XRANGE
        self.yrange = DEFAULT_YRANGE
        self.a = DEFAULT_A
        self.n = 5

        self.fig = plt.figure(figsize=(12, 7))
        self._layout_axes()

        self.line_true, = self.ax.plot([], [], lw=2, label="f(x)")
        self.line_tay, = self.ax.plot([], [], "--", lw=2, label="Taylor")
        self.vline = self.ax.axvline(self.a, lw=1)

        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left")

        self.btn_fx.on_clicked(self._apply_fx)
        self.btn_rng.on_clicked(self._apply_ranges)
        self.btn_a.on_clicked(self._apply_a)

        self.pending_n = int(self.sl_n.val)
        self.sl_n.on_changed(self._on_n_change)

        self.tb_fx.on_submit(lambda _: self._apply_fx(None))
        self.tb_xr.on_submit(lambda _: self._apply_ranges(None))
        self.tb_yr.on_submit(lambda _: self._apply_ranges(None))
        self.tb_a.on_submit(lambda _: self._apply_a(None))

        self.btn_refresh.on_clicked(lambda _: self._redraw(full = True))

        self._set_fx(self.expr_str)

    def _layout_axes(self):
        graph_left, graph_bottom, graph_width, graph_height = (0.08, 0.4, 0.86, 0.55)
        self.ax = self.fig.add_axes((graph_left, graph_bottom, graph_width, graph_height))

        self.ax_fx = self.fig.add_axes((0.12, 0.26, 0.56, 0.05))
        self.tb_fx = TextBox(self.ax_fx, "f(x) =  ", initial=self.expr_str)
        self.ax_fx_btn = self.fig.add_axes((0.7, 0.26, 0.10, 0.05))
        self.btn_fx = Button(self.ax_fx_btn, "Apply")

        self.ax_xr = self.fig.add_axes((0.12, 0.2, 0.24, 0.05))
        self.tb_xr = TextBox(self.ax_xr, "x range  ", initial=str(self.xrange))
        self.ax_yr = self.fig.add_axes((0.44, 0.2, 0.24, 0.05))
        self.tb_yr = TextBox(self.ax_yr, "y range  ", initial=str(self.yrange))
        self.ax_rng_btn = self.fig.add_axes((0.7, 0.2, 0.1, 0.05))
        self.btn_rng = Button(self.ax_rng_btn, "Set ranges")

        self.ax_a = self.fig.add_axes((0.12, 0.145, 0.18, 0.05))
        self.tb_a = TextBox(self.ax_a, "a =   ", initial=str(self.a))
        self.ax_a_btn = self.fig.add_axes((0.32, 0.145, 0.1, 0.05))
        self.btn_a = Button(self.ax_a_btn, "Set a")

        self.ax_n = self.fig.add_axes((0.12, 0.085, 0.62, 0.05))
        self.sl_n = Slider(self.ax_n, "degree n", 0, MAX_N, valinit=self.n, valstep=1)

        self.status = self.fig.text(0.08, 0.03, "", fontsize=12, va="bottom", ha="left")

        self.ax_refresh = self.fig.add_axes((0.82, 0.085, 0.12, 0.05))
        self.btn_refresh = Button(self.ax_refresh, "Refresh")

    def _msg(self, s):
        self.status.set_text(s)
        self.fig.canvas.draw_idle()

    def _set_fx(self, s):
        self.expr = parse_function(s)
        self.f_num = sp.lambdify(x, self.expr, "numpy")
        self.expr_str = s
        self._msg(f"OK: f(x) = {sp.sstr(self.expr)}")

    def _redraw(self, full = False):
        n = int(self.pending_n)

        xs = np.linspace(*self.xrange, POINTS)
        y_true = safe_eval(self.f_num, xs)
        d = derivs_at(self.expr, self.a, MAX_N)
        y_tay = taylor_eval(xs, self.a, d, n)

        self.line_true.set_data(xs, y_true)
        self.line_tay.set_data(xs, y_tay)
        self.vline.set_xdata([self.a, self.a])

        self.ax.set_xlim(*self.xrange)
        self.ax.set_ylim(*self.yrange)
        self.ax.set_title(f"Taylor about a={self.a:.4g}, n={n}")

        if full:
            self._update_latex(n)
        self.fig.canvas.draw_idle()

    def _apply_fx(self, _):
        try:
            self._set_fx(self.tb_fx.text)
        except Exception as e:
            self._msg(str(e))

    def _apply_ranges(self, _):
        try:
            self.xrange = parse_range(self.tb_xr.text, "x range")
            self.yrange = parse_range(self.tb_yr.text, "y range")
            self._msg(f"x={self.xrange}, y={self.yrange}")

        except Exception as e:
            self._msg(str(e))

    def _apply_a(self, _):
        try:
            self.a = parse_float(self.tb_a.text, "a")
            self._msg(f"a = {self.a}")

        except Exception as e:
            self._msg(str(e))

    def _on_n_change(self, _):
        self.pending_n = int(self.sl_n.val)
        self.ax.set_title(f"Taylor about a={self.a:.4g}, n={self.pending_n} (pending)")
        self.fig.canvas.draw_idle()

    def _update_latex(self, n: int):
        t_expr = taylor_expr(self.expr, self.a, n)

        fx_tex = latex(self.expr)
        tx_tex = latex(t_expr, fold_short_frac=True)

        MAX_CHARS = 300
        if len(tx_tex) > MAX_CHARS:
            tx_tex = tx_tex[:MAX_CHARS] + r"\;\cdots"

        self._msg(rf"$f(x)={fx_tex}$" + "\n" + rf"$T_{{{n}}}(x)={tx_tex}$")

def main():
    TaylorUI()
    plt.show()

if __name__ == "__main__":
    main()