import sympy as sp
import random
import math

# ------------------------------------------------------------------
# 1. Declare symbols (assume all positive where needed)
# ------------------------------------------------------------------
Km1, Km2, V1, V2 = sp.symbols('Km1 Km2 V1 V2', positive=True)
x_min, x_max = sp.symbols('x_min x_max', real=True)
y_min1, y_max1 = sp.symbols('y_min1 y_max1', real=True)
y_min2, y_max2 = sp.symbols('y_min2 y_max2', real=True)

# ------------------------------------------------------------------
# 2. Handy deltas
# ------------------------------------------------------------------
Dx  = x_max - x_min
Dy1 = y_max1 - y_min1
Dy2 = y_max2 - y_min2
sqrt_term = sp.sqrt(Dy1 * Dy2)            # common square‑root

# ------------------------------------------------------------------
# 3. New formulation  A,B,C,X_new
# ------------------------------------------------------------------
A = sp.sqrt(Km1*Km2*V1*V2) * sp.sqrt(Dy1*Dy2) * (Km1 + Km2 + 1) * Dx
B = -Km1*Km2*Dx * (V1*Dy1 + V2*Dy2)
C = Km1*V1*x_min*Dy1 - Km2*V2*x_max*Dy2
X_new = (A + B + C) / (Km1*V1*Dy1 - Km2*V2*Dy2)

# ------------------------------------------------------------------
# 4. Original long expression  X_orig
# ------------------------------------------------------------------
X_orig = (
    Km1**sp.Rational(3,2) * sp.sqrt(Km2)*sp.sqrt(V1)*sp.sqrt(V2)*x_max*sqrt_term
  - Km1**sp.Rational(3,2) * sp.sqrt(Km2)*sp.sqrt(V1)*sp.sqrt(V2)*x_min*sqrt_term
  + sp.sqrt(Km1)*Km2**sp.Rational(3,2)*sp.sqrt(V1)*sp.sqrt(V2)*x_max*sqrt_term
  - sp.sqrt(Km1)*Km2**sp.Rational(3,2)*sp.sqrt(V1)*sp.sqrt(V2)*x_min*sqrt_term
  + sp.sqrt(Km1)*sp.sqrt(Km2)*sp.sqrt(V1)*sp.sqrt(V2)*x_max*sqrt_term
  - sp.sqrt(Km1)*sp.sqrt(Km2)*sp.sqrt(V1)*sp.sqrt(V2)*x_min*sqrt_term
  - Km1*Km2*V1*x_max*y_max1 + Km1*Km2*V1*x_max*y_min1
  + Km1*Km2*V1*x_min*y_max1 - Km1*Km2*V1*x_min*y_min1
  - Km1*Km2*V2*x_max*y_max2 + Km1*Km2*V2*x_max*y_min2
  + Km1*Km2*V2*x_min*y_max2 - Km1*Km2*V2*x_min*y_min2
  + Km1*V1*x_min*y_max1 - Km1*V1*x_min*y_min1
  - Km2*V2*x_max*y_max2 + Km2*V2*x_max*y_min2
) / (Km1*V1*y_max1 - Km1*V1*y_min1 - Km2*V2*y_max2 + Km2*V2*y_min2)

# ------------------------------------------------------------------
# 5. Symbolic check
# ------------------------------------------------------------------
symbolic_diff = sp.simplify(X_new - X_orig)

print("Symbolic simplification gives:", symbolic_diff)

# ------------------------------------------------------------------
# 6. Numeric sanity check with random positive values
# ------------------------------------------------------------------
def random_positive(lo=1.0, hi=10.0):
    return random.uniform(lo, hi)

for i in range(5):
    # generate values ensuring y_max > y_min and x_max > x_min
    xm = random_positive()
    xp = xm + random_positive()
    ym1 = random_positive()
    yp1 = ym1 + random_positive()
    ym2 = random_positive()
    yp2 = ym2 + random_positive()
    subs = {
        Km1: random_positive(),
        Km2: random_positive(),
        V1:  random_positive(),
        V2:  random_positive(),
        x_min: xm,
        x_max: xp,
        y_min1: ym1,
        y_max1: yp1,
        y_min2: ym2,
        y_max2: yp2
    }
    num_diff = float(X_new.subs(subs) - X_orig.subs(subs))
    print(f"Test {i+1}: numeric diff = {num_diff:.3e}")


