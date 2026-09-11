import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Equation Grapher",
    page_icon="📈",
    layout="wide"
)

x, y = sp.symbols("x y")

# Supported functions
FUNCTIONS = {
    "x": x,
    "y": y,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


def make_expression(equation):
    equation = equation.strip()

    # Allow x^2 instead of x**2
    equation = equation.replace("^", "**")

    # Convert equation into F(x,y) = 0
    if "=" in equation:
        left, right = equation.split("=", 1)

        left = sp.sympify(left.strip(), locals=FUNCTIONS)
        right = sp.sympify(right.strip(), locals=FUNCTIONS)

        return left - right

    return sp.sympify(equation, locals=FUNCTIONS)


# ---------------- UI ----------------

st.title("📈 Equation Grapher")
st.caption("Enter a mathematical equation and generate its graph.")

equation = st.text_input(
    "Enter equation",
    value="y = x^2",
    placeholder="Example: y = x^2"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    xmin = st.number_input("X minimum", value=-10.0)

with col2:
    xmax = st.number_input("X maximum", value=10.0)

with col3:
    ymin = st.number_input("Y minimum", value=-10.0)

with col4:
    ymax = st.number_input("Y maximum", value=10.0)


# Examples
st.write("### Examples")

examples = [
    "y = x^2",
    "x = -2",
    "y = sin(x)",
    "x^2 + y^2 = 25",
    "(x^2+y^2-1)^3 = x^2*y^3"
]

example = st.selectbox(
    "Choose an example",
    ["Custom"] + examples
)

if example != "Custom":
    equation = example

plot_button = st.button(
    "🚀 Generate Graph",
    type="primary",
    use_container_width=True
)


# ---------------- GRAPH ----------------

if plot_button:

    if xmin >= xmax:
        st.error("X minimum must be smaller than X maximum.")
        st.stop()

    if ymin >= ymax:
        st.error("Y minimum must be smaller than Y maximum.")
        st.stop()

    try:

        expression = make_expression(equation)

        # Generate grid
        X = np.linspace(xmin, xmax, 700)
        Y = np.linspace(ymin, ymax, 700)

        XX, YY = np.meshgrid(X, Y)

        # Convert SymPy expression to NumPy function
        func = sp.lambdify(
            (x, y),
            expression,
            modules="numpy"
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore"
        ):
            Z = func(XX, YY)

        Z = np.asarray(Z, dtype=float)

        # Remove invalid values
        Z[~np.isfinite(Z)] = np.nan

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.contour(
            XX,
            YY,
            Z,
            levels=[0],
            linewidths=2
        )

        # Axes
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(
            f"Graph of {equation}"
        )

        ax.grid(True, alpha=0.3)

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        st.pyplot(fig)

        plt.close(fig)

        st.success("Graph generated successfully!")

    except Exception as error:

        st.error("Could not graph this equation.")

        st.code(str(error))

        st.info(
            "Try formats such as: "
            "y=x^2, x=-2, sin(x), "
            "x^2+y^2=25"
        )
