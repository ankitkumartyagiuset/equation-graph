import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

st.set_page_config(page_title="Equation Graph", page_icon="📈", layout="wide")

x, y, z, t, u, v = sp.symbols("x y z t u v")

TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
)

LOCAL_DICT = {
    "x": x, "y": y, "z": z, "t": t, "u": u, "v": v,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log,
    "exp": sp.exp, "abs": sp.Abs,
    "pi": sp.pi, "e": sp.E,
}

SUPERSCRIPTS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
})

def clean(s):
    s = s.strip()
    s = s.translate(SUPERSCRIPTS)
    s = (s.replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("×", "*").replace("·", "*").replace("÷", "/")
           .replace("π", "pi"))
    s = re.sub(r"\b(sin|cos|tan)\s*(\d+)\s*\(", r"\1(\2*", s)
    # Repair common sin3(t) style after unicode superscript conversion.
    s = re.sub(r"\b(sin|cos|tan)(\d+)\s*\(([^()]*)\)", r"\1(\3)**\2", s)
    s = s.replace("^", "**")
    return s

def parse_math(s):
    return parse_expr(clean(s), local_dict=LOCAL_DICT,
                      transformations=TRANSFORMS, evaluate=True)

def equations(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Allow x(t)=..., y(t)=..., z(u,v)=...
        parts = [p.strip() for p in re.split(r"\s*,\s*(?=[xyz](?:\([^)]*\))?\s*=)", line)]
        out.extend([p for p in parts if p])
    return out

def lhs_rhs(line):
    if "=" not in line:
        return None, None
    a, b = line.split("=", 1)
    return a.strip(), b.strip()

def parse_parametric(text):
    result = {}
    for line in equations(text):
        left, right = lhs_rhs(line)
        if left is None:
            continue
        m = re.fullmatch(r"([xyz])(?:\(([^)]*)\))?", left.replace(" ", ""))
        if not m:
            continue
        name = m.group(1)
        args = (m.group(2) or "").split(",")
        result[name] = (args, parse_math(right))
    return result

def npfunc(expr, vars_):
    return sp.lambdify(vars_, expr, modules=["numpy"])

st.title("📈 Universal Equation Grapher")
st.caption("Paste equations exactly like x(t)=..., y(t)=... or x(u,v)=..., y(u,v)=..., z(u,v)=...")

examples = {
    "❤️ Heart": """x(t) = 16sin³(t)
y(t) = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)""",
    "❤️ 3D Heart": """x(u,v) = sin(u)(15sin(v) - 4sin(3v))
y(u,v) = 8cos(u)
z(u,v) = sin(u)(15cos(v) - 5cos(2v) - 2cos(3v) - cos(4v))""",
    "🌎 Sphere": """x(u,v) = sin(u)cos(v)
y(u,v) = sin(u)sin(v)
z(u,v) = cos(u)""",
    "⭕ Circle": """x(t) = cos(t)
y(t) = sin(t)""",
}

with st.sidebar:
    st.header("⚙️ Settings")
    example = st.selectbox("Example", ["Custom"] + list(examples))
    if example != "Custom":
        default = examples[example]
    else:
        default = """x(t) = cos(t)
y(t) = sin(t)"""

    graph_mode = st.selectbox(
        "Graph type",
        ["Auto", "2D Function", "2D Parametric", "2D Implicit",
         "Polar", "3D Surface", "3D Parametric"]
    )

    tmin = st.number_input("Parameter/range minimum", value=0.0)
    tmax = st.number_input("Parameter/range maximum", value=6.283185)
    resolution = st.slider("Resolution", 300, 2000, 900, 100)

text = st.text_area(
    "✍️ Enter your equations",
    value=default,
    height=170,
    placeholder="""Example:
x(t) = 16sin³(t)
y(t) = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)

Or 3D:
x(u,v) = sin(u)cos(v)
y(u,v) = sin(u)sin(v)
z(u,v) = cos(u)""",
)

if st.button("🚀 Generate Graph", type="primary", use_container_width=True):
    try:
        p = parse_parametric(text)
        clean_text = text.lower()

        # ---------- 3D Parametric ----------
        if graph_mode == "3D Parametric" or (
            graph_mode == "Auto" and all(k in p for k in ("x", "y", "z"))
            and any("u" in a or "v" in a for a, _ in p["x"] + p["y"] + p["z"])
        ):
            if not all(k in p for k in ("x", "y", "z")):
                st.error("3D parametric graph ke liye x(u,v), y(u,v), z(u,v) teenon do.")
                st.stop()

            ex, ey, ez = p["x"][1], p["y"][1], p["z"][1]
            U = np.linspace(tmin, tmax if tmax > tmin else 2*np.pi, resolution // 2)
            V = np.linspace(0, 2*np.pi, resolution // 2)
            U, V = np.meshgrid(U, V)

            X = npfunc(ex, (u, v))(U, V)
            Y = npfunc(ey, (u, v))(U, V)
            Z = npfunc(ez, (u, v))(U, V)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0,
                            antialiased=True, alpha=0.95)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D Parametric Graph")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ---------- 2D Parametric ----------
        elif graph_mode == "2D Parametric" or (
            graph_mode == "Auto" and "x" in p and "y" in p
            and not "z" in p
        ):
            if not ("x" in p and "y" in p):
                st.error("2D parametric graph ke liye x(t)=... aur y(t)=... do.")
                st.stop()

            ex, ey = p["x"][1], p["y"][1]
            T = np.linspace(tmin, tmax, resolution)
            X = npfunc(ex, (t))(T)
            Y = npfunc(ey, (t))(T)

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot(X, Y, linewidth=3)
            ax.scatter([X[0]], [Y[0]], s=55, label="Start")
            ax.scatter([X[-1]], [Y[-1]], s=55, label="End")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("2D Parametric Graph")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ---------- 3D Surface ----------
        elif graph_mode == "3D Surface":
            line = equations(text)[0]
            left, right = lhs_rhs(line)
            if left is None:
                raise ValueError("Format: z = expression")
            expr = parse_math(right)
            X = np.linspace(tmin, tmax, 180)
            Y = np.linspace(tmin, tmax, 180)
            X, Y = np.meshgrid(X, Y)
            Z = npfunc(expr, (x, y))(X, Y)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0)
            ax.set_title("3D Surface")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ---------- 2D Implicit ----------
        elif graph_mode == "2D Implicit":
            line = equations(text)[0]
            left, right = lhs_rhs(line)
            if left is None:
                raise ValueError("Format: expression = expression")
            expr = parse_math(left) - parse_math(right)

            X = np.linspace(tmin, tmax, 600)
            Y = np.linspace(tmin, tmax, 600)
            XX, YY = np.meshgrid(X, Y)
            F = npfunc(expr, (x, y))(XX, YY)

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.contour(XX, YY, F, levels=[0], linewidths=3)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_title("2D Implicit Graph")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ---------- 2D Function ----------
        else:
            line = equations(text)[0]
            left, right = lhs_rhs(line)
            if left is not None:
                expr = parse_math(right)
            else:
                expr = parse_math(line)

            X = np.linspace(tmin, tmax, resolution)
            Y = npfunc(expr, (x))(X)

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot(X, Y, linewidth=3)
            ax.axhline(0, linewidth=1, alpha=0.5)
            ax.axvline(0, linewidth=1, alpha=0.5)
            ax.grid(alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("y = f(x)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except Exception as e:
        st.error("Equation samajhne mein problem hui.")
        st.code(str(e))

st.divider()
st.markdown(
    "**Tip:** `sin³(t)` bhi chalega, `sin(t)^3` bhi, aur `16sin³(t)` bhi. "
    "3D ke liye teen lines `x(u,v)`, `y(u,v)`, `z(u,v)` mein do."
            )

