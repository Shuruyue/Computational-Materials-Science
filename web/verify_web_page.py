"""Check the JavaScript of the interactive page web/numerical-methods.html against the weekly Python modules.

The page holds JavaScript copies of week01/work/quadrature.py, week02/work/finite_diff.py and
week03/work/ode.py between the markers "// ---- physics: begin" and "// ---- physics: end". This script runs
that block in Node.js and compares it with the Python code:

- trapezoid and Simpson values for the three integrands of week 1, N = 2 ... 65536;
- the exact error formulas drawn on the page, against SciPy (zeta(-1/2), modified Bessel functions);
- every finite-difference stencil for 20 points x0, at steps 0.5 ... 1e-5 where truncation dominates
  (at smaller steps the browser's sine may round differently from NumPy's in the last bit);
- forward and backward Euler, the trapezoidal rule and RK4 on the week-3 test problem;
- the RK4 stability limit on the negative real axis.

Usage (from the repository root): python web/verify_web_page.py
Requires Node.js on the PATH and the conda environment of the repository (NumPy, SciPy).
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import iv, zeta

ROOT = Path(__file__).resolve().parent.parent
for week in ("week01", "week02", "week03"):
    sys.path.insert(0, str(ROOT / week / "work"))
import finite_diff as fd  # noqa: E402
import ode  # noqa: E402
import quadrature as qd  # noqa: E402

PAGE = ROOT / "web" / "numerical-methods.html"
REL_TOL = 1e-9
NS = [2 ** k for k in range(1, 17)]
X0 = [0.05 + i * 0.15 for i in range(20)]
HS = [0.5 * 10 ** (-k / 4) for k in range(18)]          # 0.5 down to about 1e-5
ODE_CASES = [(-10, 40), (-10, 640), (-1, 100), (-100, 300), (-2100, 2105), (-2100, 2048)]

NODE_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const page = fs.readFileSync(process.argv[1], "utf8");
const after = page.split("// ---- physics: begin")[1], block = after.slice(after.indexOf("\n")).split("// ---- physics: end")[0];
const ctx = {}; vm.createContext(ctx);
vm.runInContext(block + "\n;this.api = { trapezoid, simpson, INTEGRANDS, STENCILS, solveAll, rk4RealLimit, ZETA_M_HALF };", ctx);
const a = ctx.api, inp = JSON.parse(fs.readFileSync(0, "utf8")), out = { quad: {}, stencils: {}, ode: [], limit: a.rk4RealLimit(), zeta: a.ZETA_M_HALF };
for (const [k, it] of Object.entries(a.INTEGRANDS)) out.quad[k] = inp.Ns.map((N) => [a.trapezoid(it.f, it.a, it.b, N), a.simpson(it.f, it.a, it.b, N), it.I, it.errT(N), it.errS ? it.errS(N) : null]);
for (const [k, st] of Object.entries(a.STENCILS)) out.stencils[k] = inp.X0.map((x) => inp.HS.map((h) => st.d(Math.sin, x, h)));
for (const [lam, N] of inp.ode) { const s = a.solveAll(lam, N, 2); out.ode.push([s.fe[N], s.be[N], s.tr[N], s.rk4[N]]); }
process.stdout.write(JSON.stringify(out));
"""

INTEGRANDS = {
    "sin": (np.sin, 0.0, math.pi, 2.0),
    "sqrt": (np.sqrt, 0.0, 1.0, 2.0 / 3.0),
    "ecos": (lambda x: np.exp(np.cos(x)), 0.0, 2 * math.pi, 2 * math.pi * float(iv(0, 1))),
}


def formula(key, N):
    """The exact error formulas of the page, computed with SciPy."""
    if key == "sin":
        T = lambda n: (math.pi / n) / math.tan(math.pi / (2 * n))  # noqa: E731
        return T(N) - 2, (4 * T(N) - T(N / 2)) / 3 - 2
    if key == "sqrt":
        h = 1 / N
        return float(zeta(-0.5)) * h ** 1.5 + h ** 2 / 24, float(zeta(-0.5)) * (4 - 2 ** 1.5) / 3 * h ** 1.5
    terms = [float(iv(k * N, 1)) for k in range(1, 30)]
    return 4 * math.pi * sum(terms), None


def rel(a, b):
    if a is None or b is None:
        return 0.0 if a is None and b is None else math.inf
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


def run():
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is needed to run the page's JavaScript")
    payload = json.dumps({"Ns": NS, "X0": X0, "HS": HS, "ode": ODE_CASES})
    out = subprocess.run([node, "-e", NODE_HARNESS, str(PAGE)], input=payload, capture_output=True, text=True,
                         check=False)
    if out.returncode != 0:
        raise RuntimeError("Node.js failed:\n" + out.stderr)
    got = json.loads(out.stdout)
    worst = {}

    def note(key, value):
        worst[key] = max(worst.get(key, 0.0), value)

    for key, (f, a, b, exact) in INTEGRANDS.items():
        for N, (T, S, I, eT, eS) in zip(NS, got["quad"][key]):
            note("quadrature values", max(rel(T, qd.trapezoid(f, a, b, N)), rel(S, qd.simpson(f, a, b, N)), rel(I, exact)))
            fT, fS = formula(key, N)
            note("error formulas", max(rel(eT, fT), rel(eS, fS)))
    names = {"forward": fd.forward, "central": fd.central, "one_sided3": fd.one_sided3,
             "richardson5": fd.richardson5, "complex": fd.complex_step}
    for key, fn in names.items():
        for x, row in zip(X0, got["stencils"][key]):
            for h, value in zip(HS, row):
                note("finite differences", rel(value, float(fn(np.sin, x, h))))
    for (lam, N), js in zip(ODE_CASES, got["ode"]):
        f, g = ode.cos_problem(lam)
        h = 2 / N
        ref = [ode.forward_euler(f, 1.0, h, N)[-1], ode.backward_euler_linear(lam, g, 1.0, h, N)[-1],
               ode.trapezoid_linear(lam, g, 1.0, h, N)[-1], ode.rk4(f, 1.0, h, N)[-1]]
        note("time steppers", max(rel(a, b) for a, b in zip(js, ref)))
    note("RK4 stability limit", rel(got["limit"], ode.rk4_real_limit()))
    note("zeta(-1/2)", rel(got["zeta"], float(zeta(-0.5))))
    return worst


def main():
    worst = run()
    print("largest relative difference page vs Python:")
    for key, value in worst.items():
        print(f"  {key:22s} {value:.1e}")
    ok = max(worst.values()) <= REL_TOL
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
