
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Symbols
x, y = sp.symbols("x y")

ALLOWED = {
    "x": x, "y": y,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
    "abs": sp.Abs, "pi": sp.pi, "E": sp.E
}


def parse_equation(text):
    """Convert an equation into F(x,y)=0."""
    text = text.strip().replace("^", "**")

    if "=" in text:
        left, right = text.split("=", 1)
        return sp.sympify(left.strip(), locals=ALLOWED) - \
               sp.sympify(right.strip(), locals=ALLOWED)

    # Allow input such as: y-x^2
    return sp.sympify(text, locals=ALLOWED)


class EquationGrapher:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Equation Grapher")
        self.root.geometry("1100x750")
        self.root.minsize(850, 600)

        # ---------- Top UI ----------
        top = ttk.Frame(root, padding=12)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Equation:",
            font=("Arial", 12, "bold")
        ).pack(side="left")

        self.equation = tk.StringVar(value="y = x^2")
        entry = ttk.Entry(
            top,
            textvariable=self.equation,
            font=("Consolas", 13)
        )
        entry.pack(side="left", fill="x", expand=True, padx=10)
        entry.bind("<Return>", lambda e: self.plot())

        ttk.Button(
            top,
            text="GRAPH",
            command=self.plot
        ).pack(side="left")

        ttk.Button(
            top,
            text="CLEAR",
            command=self.clear
        ).pack(side="left", padx=(7, 0))

        # ---------- Controls ----------
        controls = ttk.LabelFrame(root, text="Graph Settings", padding=10)
        controls.pack(fill="x", padx=12, pady=(0, 10))

        ttk.Label(controls, text="X min").grid(row=0, column=0, padx=5)
        self.xmin = tk.StringVar(value="-10")
        ttk.Entry(
            controls, textvariable=self.xmin, width=9
        ).grid(row=0, column=1)

        ttk.Label(controls, text="X max").grid(row=0, column=2, padx=5)
        self.xmax = tk.StringVar(value="10")
        ttk.Entry(
            controls, textvariable=self.xmax, width=9
        ).grid(row=0, column=3)

        ttk.Label(controls, text="Y min").grid(row=0, column=4, padx=5)
        self.ymin = tk.StringVar(value="-10")
        ttk.Entry(
            controls, textvariable=self.ymin, width=9
        ).grid(row=0, column=5)

        ttk.Label(controls, text="Y max").grid(row=0, column=6, padx=5)
        self.ymax = tk.StringVar(value="10")
        ttk.Entry(
            controls, textvariable=self.ymax, width=9
        ).grid(row=0, column=7)

        self.grid_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Grid",
            variable=self.grid_on,
            command=self.refresh_grid
        ).grid(row=0, column=8, padx=15)

        # ---------- Examples ----------
        examples = ttk.Frame(root, padding=(12, 0, 12, 8))
        examples.pack(fill="x")

        ttk.Label(
            examples,
            text="Examples:",
            font=("Arial", 10, "bold")
        ).pack(side="left")

        for eq in [
            "y = x^2",
            "x = -2",
            "y = sin(x)",
            "x^2 + y^2 = 25",
            "(x^2+y^2-1)^3 = x^2*y^3"
        ]:
            ttk.Button(
                examples,
                text=eq,
                command=lambda e=eq: self.set_example(e)
            ).pack(side="left", padx=3)

        # ---------- Plot ----------
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=12, pady=5
        )

        self.status = tk.StringVar(value="Enter an equation and press GRAPH.")
        ttk.Label(
            root,
            textvariable=self.status,
            relief="sunken",
            anchor="w",
            padding=5
        ).pack(fill="x", side="bottom")

        self.plot()

    def set_example(self, equation):
        self.equation.set(equation)
        self.plot()

    def get_ranges(self):
        xmin = float(self.xmin.get())
        xmax = float(self.xmax.get())
        ymin = float(self.ymin.get())
        ymax = float(self.ymax.get())

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Minimum must be smaller than maximum.")

        return xmin, xmax, ymin, ymax

    def plot(self):
        try:
            expression = parse_equation(self.equation.get())
            xmin, xmax, ymin, ymax = self.get_ranges()

            X = np.linspace(xmin, xmax, 900)
            Y = np.linspace(ymin, ymax, 900)
            XX, YY = np.meshgrid(X, Y)

            function = sp.lambdify(
                (x, y),
                expression,
                modules=["numpy"]
            )

            with np.errstate(all="ignore"):
                Z = function(XX, YY)

            Z = np.asarray(Z, dtype=float)

            # Fix scalar output for unusual equations
            if Z.shape == ():
                Z = np.full_like(XX, Z)

            Z[~np.isfinite(Z)] = np.nan

            self.ax.clear()

            # F(x,y)=0 is the actual equation curve
            self.ax.contour(
                XX,
                YY,
                Z,
                levels=[0],
                linewidths=2
            )

            self.ax.axhline(0, linewidth=1)
            self.ax.axvline(0, linewidth=1)

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_title(
                f"Graph: {self.equation.get()}",
                fontsize=13
            )
            self.ax.grid(self.grid_on.get(), alpha=0.3)

            # Equal scale when possible
            self.ax.set_aspect("equal", adjustable="box")

            self.figure.tight_layout()
            self.canvas.draw()

            self.status.set(
                f"Plotted successfully: {self.equation.get()}"
            )

        except Exception as error:
            messagebox.showerror(
                "Invalid Equation",
                f"Could not graph the equation.\n\n{error}"
            )
            self.status.set("Error: invalid equation.")

    def refresh_grid(self):
        self.ax.grid(self.grid_on.get(), alpha=0.3)
        self.canvas.draw()

    def clear(self):
        self.equation.set("")
        self.ax.clear()
        self.ax.axhline(0, linewidth=1)
        self.ax.axvline(0, linewidth=1)
        self.ax.grid(self.grid_on.get(), alpha=0.3)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Equation Grapher")
        self.canvas.draw()
        self.status.set("Graph cleared.")


if __name__ == "__main__":
    root = tk.Tk()

    # Use a modern ttk theme when available
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    EquationGrapher(root)
    root.mainloop()
