import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import re

st.set_page_config(
    page_title="Universal Graph Question Solver",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------
# Math setup
# ---------------------------------------------------------
MATH = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log,
    "exp": sp.exp, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceil": sp.ceiling,
    "pi": sp.pi, "e": sp.E,
    "E": sp.E
}

x, y, t, u, v = sp.symbols("x y t u v")
MATH.update({"x": x, "y": y, "t": t, "u": u, "v": v})


def clean_math(s):
    """Make common calculator/exam notation SymPy-friendly."""
    s = s.strip()
    s = s.replace("−", "-").replace("π", "pi")
    s = s.replace("×", "*").replace("·", "*")
    s = s.replace("^", "**")
    s = re.sub(r"\bln\s*\(", "log(", s)
    return s


def parse_expr(text):
    return sp.sympify(clean_math(text), locals=MATH)


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
            found.append(s)

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
            "3D paraboloid"
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
        "3D paraboloid": "Plot 3D surface z = x^2 + y^2"
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
                m = re.match(r"\s*x\s*\(\s*t\s*\)\s*=\s*(.*)", clean_math(eq), re.I)
                if m:
                    ex = parse_expr(m.group(1))
                m = re.match(r"\s*y\s*\(\s*t\s*\)\s*=\s*(.*)", clean_math(eq), re.I)
                if m:
                    ey = parse_expr(m.group(1))

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
            
