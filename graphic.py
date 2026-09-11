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
    # sin³(t) may have become sin3(t) above, so detect the original pattern
    # separately if needed.
    normalized = re.sub(
        r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|sqrt|log|ln|exp|abs|sign|floor|ceil)"
        r"\s*(\d+)\s*\(([^()]*)\)",
        lambda m: f"{m.group(1)}({m.group(3)})**{m.group(2)}",
        normalized,
        flags=re.I,
    )

    return sympy_parse_expr(
        normalized,
        local_dict=MATH,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )

def equation_to_expression(text, variable=None):
    """
    Convert:
      y = x^2       -> x^2
      x = -2        -> x + 2
      x^2+y^2 = 1  -> x^2+y^2-1
      y = ...       -> RHS
    """
    text = clean_math(text)
    text = text.strip()

    if "=" in text:
        left, right = text.split("=", 1)
        left = left.strip()
        right = right.strip()

        # For y=f(x), return f(x)
        if left == "y":
            return parse_expr(right)

        # For x=f(y), return f(y)
        if left == "x":
            return parse_expr(right)

        return parse_expr(left) - parse_expr(right)

    # If no equals sign, treat as an expression.
    # In function mode, bare expression means y = expression.
    return parse_expr(text)


def safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------
# Question interpreter
# ---------------------------------------------------------
def extract_equations(question):
    """
    Pull equation-like lines from a natural-language question.
    It deliberately stays conservative instead of executing arbitrary text.
    """
    q = question.strip()
    lines = [z.strip() for z in q.splitlines() if z.strip()]

    found = []

    for line in lines:
        # Remove common question prefixes.
        s = re.sub(
            r"^(plot|graph|draw|sketch|show|find|solve|consider|given)\s*[:\-]?\s*",
            "",
            line,
            flags=re.I
        )

        # Remove bullets / numbering.
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s)

        # Keep lines containing likely graph syntax.
        if re.search(r"(=|\by\s*=|\bx\s*=|\br\s*=|\bz\s*=)", s, re.I):
            # Split common inline pairs such as:
            # x(t)=cos(t), y(t)=sin(t)
            parts = re.split(r"\s*,\s*(?=(?:x|y|z|r)\s*(?:\([^)]*\))?\s*=)", s, flags=re.I)
            found.extend(parts if len(parts) > 1 else [s])

    # Also handle a single sentence containing an equation.
    if not found:
        for match in re.findall(
            r"(?:x|y|z|r)\s*(?:\([^)]*\))?\s*=\s*[^,;\n]+",
            q,
            flags=re.I
        ):
            found.append(match.strip())

    return found


def detect_graph_type(question, equations):
    q = question.lower()

    if any(k in q for k in ["butterfly", "butter fly"]):
        return "Butterfly"

    if "heart" in q or "love" in q:
        return "Heart"

    if "polar" in q or any(e.lower().startswith("r") and "=" in e for e in equations):
        return "Polar"

    if "parametric" in q or any("(t)" in e.lower() for e in equations):
        return "2D Parametric"

    if any(k in q for k in ["animate", "animation", "rotating", "moving"]):
        if re.search(r"\bx\s*\(\s*u\s*,\s*v\s*\)", " ".join(equations), re.I):
            return "3D Animated Parametric"
        return "3D Animated Surface"

    if "surface" in q or re.search(r"\bz\s*=", " ".join(equations), re.I):
        return "3D Surface"

    if "implicit" in q:
        return "2D Implicit"

    return "Auto"


# ---------------------------------------------------------
# Preset curves
# ---------------------------------------------------------
def plot_special(name, ax, resolution):
    if name == "Butterfly":
        tt = np.linspace(0, 12 * np.pi, resolution)
        r = (
            np.exp(np.sin(tt))
            - 2 * np.cos(4 * tt)
            + np.sin((2 * tt - np.pi) / 24) ** 5
        )
        xx = r * np.cos(tt)
        yy = r * np.sin(tt)
        ax.plot(xx, yy)
        ax.set_title("Butterfly Curve")

    elif name == "Heart":
        tt = np.linspace(0, 2 * np.pi, resolution)
        xx = 16 * np.sin(tt) ** 3
        yy = (
            13 * np.cos(tt)
            - 5 * np.cos(2 * tt)
            - 2 * np.cos(3 * tt)
            - np.cos(4 * tt)
        )
        ax.plot(xx, yy)
        ax.set_title("❤️ Heart Curve")

    ax.set_aspect("equal", adjustable="box")


# ---------------------------------------------------------
# 3D Animation helper
# ---------------------------------------------------------
def animate_3d_surface(expr, xmin, xmax, ymin, ymax, resolution=180):
    """Create an animated rotating 3D surface and return the HTML."""
    from matplotlib.animation import FuncAnimation
    from matplotlib import cm
    from matplotlib import pyplot as plt

    n = min(int(resolution), 220)
    X = np.linspace(xmin, xmax, n)
    Y = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(X, Y)

    fn = sp.lambdify((x, y), expr, "numpy")
    ZZ = np.asarray(fn(XX, YY), dtype=float)
    ZZ[~np.isfinite(ZZ)] = np.nan

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    finite = ZZ[np.isfinite(ZZ)]
    if finite.size:
        zmin, zmax = np.nanpercentile(finite, [2, 98])
        if zmin == zmax:
            zmin, zmax = np.nanmin(finite), np.nanmax(finite)
        ax.set_zlim(zmin, zmax)

    surf = [None]

    def update(frame):
        ax.clear()
        surf[0] = ax.plot_surface(
            XX, YY, ZZ,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True
        )
        ax.view_init(elev=28, azim=frame)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("🌌 Animated 3D Surface")
        return (surf[0],)

    animation = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 360, 4),
        interval=60,
        blit=False
    )

    from matplotlib.animation import PillowWriter
    import tempfile
    import base64

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = tmp.name

    animation.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)

    data = Path(gif_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def animate_3d_parametric(ex, ey, ez, resolution=260):
    """Create a rotating 3D parametric surface/shape."""
    from matplotlib.animation import FuncAnimation
    from matplotlib import cm
    import tempfile
    import base64

    n = min(int(resolution), 260)
    U = np.linspace(0, np.pi, n)
    V = np.linspace(0, 2 * np.pi, n)
    UU, VV = np.meshgrid(U, V)

    fx = sp.lambdify((u, v), ex, "numpy")
    fy = sp.lambdify((u, v), ey, "numpy")
    fz = sp.lambdify((u, v), ez, "numpy")

    XX = np.asarray(fx(UU, VV), dtype=float)
    YY = np.asarray(fy(UU, VV), dtype=float)
    ZZ = np.asarray(fz(UU, VV), dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        ax.plot_surface(
            XX, YY, ZZ,
            cmap=cm.plasma,
            linewidth=0,
            antialiased=True
        )
        ax.view_init(elev=25 + 8*np.sin(np.radians(frame)), azim=frame)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("✨ Animated 3D Parametric Surface")

    animation = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 360, 5),
        interval=70,
        blit=False
    )

    from matplotlib.animation import PillowWriter
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = tmp.name

    animation.save(gif_path, writer=PillowWriter(fps=14))
    plt.close(fig)

    return base64.b64encode(Path(gif_path).read_bytes()).decode("utf-8")


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("📈 Universal Graph Question Solver")
st.caption(
    "Type an equation OR an exam-style graph question. "
    "The app detects the graph family and plots it."
)

question = st.text_area(
    "📝 Enter your question",
    height=130,
    placeholder=(
        "Examples:\n"
        "Plot y = x^2\n"
        "Draw x = -2 and y = x^2\n"
        "Plot the butterfly curve\n"
        "Plot the heart curve\n"
        "Plot the parametric curve x(t)=cos(t), y(t)=sin(t)\n"
        "Plot the polar curve r = 1 + cos(t)\n"
        "Plot the 3D surface z = sin(sqrt(x^2+y^2))"
    )
)

with st.sidebar:
    st.header("⚙️ Graph Settings")

    graph_override = st.selectbox(
        "Graph type",
        [
            "Auto",
            "2D Function",
            "2D Implicit",
            "2D Parametric",
            "Polar",
            "3D Surface",
            "3D Animated Surface",
            "3D Animated Parametric",
            "Butterfly",
            "Heart"
        ]
    )

    col1, col2 = st.columns(2)
    xmin = col1.number_input("X min", value=-10.0)
    xmax = col2.number_input("X max", value=10.0)

    col3, col4 = st.columns(2)
    ymin = col3.number_input("Y min", value=-10.0)
    ymax = col4.number_input("Y max", value=10.0)

    resolution = st.slider("Resolution", 300, 1800, 800, 100)
    grid = st.checkbox("Show grid", True)
    show_axes = st.checkbox("Show axes", True)

    st.divider()
    st.subheader("✨ Quick examples")
    example = st.selectbox(
        "Choose one",
        [
            "None",
            "y = x^2",
            "x = -2",
            "x^2 + y^2 = 25",
            "Butterfly",
            "Heart",
            "Parametric circle",
            "Polar cardioid",
            "3D paraboloid",
            "Animated 3D paraboloid",
            "Animated 3D sphere"
        ]
    )

if example != "None" and st.button("Use Example"):
    examples = {
        "y = x^2": "Plot y = x^2",
        "x = -2": "Draw x = -2",
        "x^2 + y^2 = 25": "Plot x^2 + y^2 = 25",
        "Butterfly": "Plot the butterfly curve",
        "Heart": "Plot the heart curve",
        "Parametric circle": "Plot parametric x(t)=cos(t), y(t)=sin(t)",
        "Polar cardioid": "Plot polar r = 1 + cos(t)",
        "3D paraboloid": "Plot 3D surface z = x^2 + y^2",
        "Animated 3D paraboloid": "Animate 3D surface z = sin(sqrt(x^2+y^2))",
        "Animated 3D sphere": (
            "Animate 3D parametric "
            "x(u,v)=sin(u)*cos(v) "
            "y(u,v)=sin(u)*sin(v) "
            "z(u,v)=cos(u)"
        )
    }
    st.session_state["question"] = examples[example]
    st.rerun()

question = st.session_state.get("question", question)

if st.button("🚀 SOLVE & GRAPH", type="primary"):
    if not question.strip():
        st.warning("Please enter a graph question.")
        st.stop()

    equations = extract_equations(question)
    detected = detect_graph_type(question, equations)

    graph_type = detected if graph_override == "Auto" else graph_override

    st.info(
        f"Detected graph type: **{graph_type}**"
        + (f"  |  Equations found: `{len(equations)}`" if equations else "")
    )

    if equations:
        with st.expander("🔎 Parsed input"):
            for eq in equations:
                st.code(clean_math(eq), language="text")

    # Special curves
    if graph_type in ["Butterfly", "Heart"]:
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_special(graph_type, ax, resolution)
        ax.grid(grid)
        if show_axes:
            ax.axhline(0, linewidth=0.8)
            ax.axvline(0, linewidth=0.8)

        if graph_type == "Heart":
            ax.text(
                0, -1.5,
                "I LOVE YOU ❤️",
                ha="center",
                va="center",
                fontsize=18
            )

        st.pyplot(fig)
        plt.close(fig)
        st.success("Graph generated successfully.")
        st.stop()

    if not equations:
        st.error(
            "I couldn't find an equation. Try something like "
            "`y = x^2`, `x^2+y^2=25`, or `r = 1+cos(t)`."
        )
        st.stop()

    # -----------------------------------------------------
    # 2D Function
    # -----------------------------------------------------
    if graph_type == "2D Function":
        fig, ax = plt.subplots(figsize=(10, 7))
        X = np.linspace(xmin, xmax, resolution)

        plotted = 0
        for eq in equations:
            try:
                expr = equation_to_expression(eq)

                # y = f(x)
                if "y" in expr.free_symbols and "x" not in expr.free_symbols:
                    # Unusual case; solve y if possible
                    sols = sp.solve(expr, y)
                    for sol in sols:
                        fn = sp.lambdify(x, sol, "numpy")
                        ax.plot(X, fn(X), label=eq)
                        plotted += 1
                else:
                    fn = sp.lambdify(x, expr, "numpy")
                    ax.plot(X, fn(X), label=eq)
                    plotted += 1

            except Exception as e:
                st.warning(f"Could not plot `{eq}`: {e}")

        if plotted == 0:
            st.error("No valid function could be plotted.")
            st.stop()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2D Function Graph")
        ax.grid(grid)
        if show_axes:
            ax.axhline(0, linewidth=0.8)
            ax.axvline(0, linewidth=0.8)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------
    # 2D Implicit
    # -----------------------------------------------------
    elif graph_type == "2D Implicit":
        fig, ax = plt.subplots(figsize=(9, 7))

        X = np.linspace(xmin, xmax, resolution)
        Y = np.linspace(ymin, ymax, resolution)
        XX, YY = np.meshgrid(X, Y)

        plotted = 0

        for eq in equations:
            try:
                expr = equation_to_expression(eq)
                fn = sp.lambdify((x, y), expr, "numpy")
                ZZ = np.asarray(fn(XX, YY), dtype=float)

                if ZZ.shape != XX.shape:
                    ZZ = np.broadcast_to(ZZ, XX.shape)

                ZZ[~np.isfinite(ZZ)] = np.nan
                ax.contour(XX, YY, ZZ, levels=[0])
                plotted += 1

            except Exception as e:
                st.warning(f"Could not plot `{eq}`: {e}")

        if plotted == 0:
            st.error("No valid implicit equation could be plotted.")
            st.stop()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("2D Implicit Graph")
        ax.grid(grid)
        if show_axes:
            ax.axhline(0, linewidth=0.8)
            ax.axvline(0, linewidth=0.8)

        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------
    # 2D Parametric
    # -----------------------------------------------------
    elif graph_type == "2D Parametric":
        if len(equations) < 2:
            st.error("Parametric graph needs both x(t) and y(t).")
            st.stop()

        try:
            ex = None
            ey = None

            for eq in equations:
                s_eq = clean_math(eq)

                m = re.match(r"\s*x\s*\(\s*t\s*\)\s*=\s*(.*)", s_eq, re.I)
                if m:
                    ex = parse_expr(m.group(1))
                    continue

                m = re.match(r"\s*y\s*\(\s*t\s*\)\s*=\s*(.*)", s_eq, re.I)
                if m:
                    ey = parse_expr(m.group(1))
                    continue

                # Also support x(t)=... / y(t)=... without requiring
                # exact spacing or with uppercase letters.

            # Also accept x=..., y=... if user selected parametric.
            if ex is None:
                for eq in equations:
                    if re.match(r"\s*x\s*=", clean_math(eq), re.I):
                        ex = parse_expr(clean_math(eq).split("=", 1)[1])
            if ey is None:
                for eq in equations:
                    if re.match(r"\s*y\s*=", clean_math(eq), re.I):
                        ey = parse_expr(clean_math(eq).split("=", 1)[1])

            if ex is None or ey is None:
                raise ValueError("Need x(t)=... and y(t)=...")

            T = np.linspace(0, 2 * np.pi, resolution)
            fx = sp.lambdify(t, ex, "numpy")
            fy = sp.lambdify(t, ey, "numpy")

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.plot(fx(T), fy(T))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("2D Parametric Graph")
            ax.grid(grid)
            if show_axes:
                ax.axhline(0, linewidth=0.8)
                ax.axvline(0, linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Parametric error: {e}")

    # -----------------------------------------------------
    # Polar
    # -----------------------------------------------------
    elif graph_type == "Polar":
        fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "polar"})

        plotted = 0
        T = np.linspace(0, 2 * np.pi, resolution)

        for eq in equations:
            try:
                s = clean_math(eq)
                if "=" in s:
                    left, right = s.split("=", 1)
                    if left.strip().lower() != "r":
                        raise ValueError("Polar equation must use r = ...")
                    expr = parse_expr(right)
                else:
                    expr = parse_expr(s)

                fn = sp.lambdify(t, expr, "numpy")
                R = fn(T)
                ax.plot(T, R, label=eq)
                plotted += 1

            except Exception as e:
                st.warning(f"Could not plot `{eq}`: {e}")

        if plotted == 0:
            st.error("No valid polar equation.")
            st.stop()

        ax.set_title("Polar Graph")
        if plotted > 1:
            ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------
    # 3D Animated Surface
    # -----------------------------------------------------
    elif graph_type == "3D Animated Surface":
        expr_text = None

        for eq in equations:
            if re.match(r"\s*z\s*=", clean_math(eq), re.I):
                expr_text = clean_math(eq).split("=", 1)[1]
                break

        if expr_text is None:
            expr_text = equations[0]

        try:
            expr = parse_expr(expr_text)

            with st.spinner("🎬 Creating 3D animation..."):
                b64 = animate_3d_surface(
                    expr, xmin, xmax, ymin, ymax,
                    resolution=min(resolution, 220)
                )

            st.image(
                f"data:image/gif;base64,{b64}",
                caption="Animated 3D surface"
            )
            st.success("3D animation generated! 🎬")

        except Exception as e:
            st.error(f"3D animation error: {e}")

    # -----------------------------------------------------
    # 3D Animated Parametric
    # -----------------------------------------------------
    elif graph_type == "3D Animated Parametric":
        try:
            ex = ey = ez = None

            for eq in equations:
                s_eq = clean_math(eq)

                mx = re.match(r"\s*x\s*\(\s*u\s*,\s*v\s*\)\s*=\s*(.*)", s_eq, re.I)
                my = re.match(r"\s*y\s*\(\s*u\s*,\s*v\s*\)\s*=\s*(.*)", s_eq, re.I)
                mz = re.match(r"\s*z\s*\(\s*u\s*,\s*v\s*\)\s*=\s*(.*)", s_eq, re.I)

                if mx:
                    ex = parse_expr(mx.group(1))
                elif my:
                    ey = parse_expr(my.group(1))
                elif mz:
                    ez = parse_expr(mz.group(1))

            if ex is None or ey is None or ez is None:
                raise ValueError(
                    "Use x(u,v)=..., y(u,v)=..., z(u,v)=..."
                )

            with st.spinner("🎬 Creating 3D parametric animation..."):
                b64 = animate_3d_parametric(
                    ex, ey, ez,
                    resolution=min(resolution, 180)
                )

            st.image(
                f"data:image/gif;base64,{b64}",
                caption="Animated 3D parametric surface"
            )
            st.success("3D parametric animation generated! ✨")

        except Exception as e:
            st.error(f"3D parametric animation error: {e}")

    # -----------------------------------------------------
    # 3D Surface
    # -----------------------------------------------------
    elif graph_type == "3D Surface":
        expr_text = None

        # Prefer z=...
        for eq in equations:
            if re.match(r"\s*z\s*=", clean_math(eq), re.I):
                expr_text = clean_math(eq).split("=", 1)[1]
                break

        if expr_text is None:
            # If user entered only an expression, use it as z=f(x,y)
            expr_text = equations[0]

        try:
            expr = parse_expr(expr_text)
            fn = sp.lambdify((x, y), expr, "numpy")

            n3 = min(resolution, 500)
            X = np.linspace(xmin, xmax, n3)
            Y = np.linspace(ymin, ymax, n3)
            XX, YY = np.meshgrid(X, Y)
            ZZ = np.asarray(fn(XX, YY), dtype=float)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title("3D Surface")
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"3D surface error: {e}")

    else:
        st.error("Choose a graph type or keep Auto mode.")


st.divider()
st.markdown(
    """
### 📚 Supported

**2D:** functions, lines, parabolas, circles, implicit curves, multiple equations  
**Parametric:** `x(t), y(t)`  
**Polar:** `r = f(t)`  
**3D:** `z = f(x,y)` surfaces  
**Special:** ❤️ Heart and 🦋 Butterfly curves

> Tip: For the most reliable result, put each equation on its own line.
"""
)
