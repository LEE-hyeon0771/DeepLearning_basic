import sympy as sp
sp.init_printing()
x,y,z = sp.var('x, y, z')

h = x**2 + y**2 + z**2
g = x+y+z-12
f = x**2 + y**2

lamda, mu = sp.symbols('lamda, mu')
L = h - lamda * g - mu * f

gradL = [sp.diff(L, var) for var in [x,y,z]]
eqs = gradL + [g] + [f]

solution = sp.solve(eqs, [x,y,z, lamda, mu], dict = True)
print([h.subs(p) for p in solution])






