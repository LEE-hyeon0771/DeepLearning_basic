import sympy as sp

sp.init_printing()
x,y,z = sp.var('x, y, z')

f = x**2 + y**2 + z**2
g = x+y+z-12
h = x**2 + y**2 - z

lamda, mu = sp.symbols('lamda, mu')
L = f - lamda * g - mu * h

gradL = [sp.diff(L, var) for var in [x,y,z]]
eqs = gradL + [g] + [h]

solution = sp.solve(eqs, [x, y, z, lamda, mu], dict=True)
print(solution)

for p in solution:
    print(f.subs(p))







