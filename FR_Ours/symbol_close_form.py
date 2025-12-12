import sympy

# 定義參數符號
V1, Km1 = sympy.symbols('V1 Km1', positive=True, real=True)
V2, Km2 = sympy.symbols('V2 Km2', positive=True, real=True)
x_min, x_max = sympy.symbols('x_min x_max', real=True, positive=True)
y_min1, y_max1 = sympy.symbols('y_min1 y_max1', real=True, positive=True)
y_min2, y_max2 = sympy.symbols('y_min2 y_max2', real=True, positive=True)
x = sympy.Symbol('x', real=True, positive=True)

# 1) x_norm1 (翻轉 x)
x_norm_1 = (x_max - x)/(x_max - x_min)
# y_norm1
y_norm_1 = (V1 * x_norm_1)/(Km1 + x_norm_1)
# F1 (翻轉 y)
F1_expr = y_min1 + (1 - y_norm_1)*(y_max1 - y_min1)

# 2) x_norm2 (不翻轉 x)
x_norm_2 = (x - x_min)/(x_max - x_min)
# y_norm2
y_norm_2 = (V2 * x_norm_2)/(Km2 + x_norm_2)
# F2 (翻轉 y)
F2_expr = y_min2 + (1 - y_norm_2)*(y_max2 - y_min2)

# 3) G(x) = F1 + F2
G_expr = F1_expr + F2_expr

# 4) dG/dx
dGdx_expr = sympy.diff(G_expr, x)

# 5) 求解 dG/dx = 0
solutions = sympy.solve(sympy.Eq(dGdx_expr, 0), x, dict=True)

# 6) 化簡 (可選)
F1_simpl = sympy.simplify(F1_expr)
F2_simpl = sympy.simplify(F2_expr)
G_simpl  = sympy.simplify(G_expr)
dGdx_expr_simpl = sympy.simplify(dGdx_expr)

print("=== Symbolic expressions (possibly simplified) ===")
print("F1(x) =", F1_simpl)
print("F2(x) =", F2_simpl)
print("G(x)  =", G_simpl)
print("dG/dx =", dGdx_expr_simpl)
print()
print("=== dG/dx = 0 => solutions ===")

# print all solutions
for i, sol_dict in enumerate(solutions):
    print(f"Solution {i+1}:", sol_dict)

# sol_str = str(solutions[1][x])  # or sympy.srepr(sol_expr)
with open("my_symbolic_solution.txt", "w", encoding="utf-8") as f:
    for i, sol_dict in enumerate(solutions):
        f.write(str(solutions[i][x]))
        # add a newline after each solution
        f.write("\n")