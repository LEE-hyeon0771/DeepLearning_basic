import sympy as sp

sp.init_printing()
x, y, z = sp.var('x, y, z')

f = x ** 2 + y ** 2 + z ** 2
g = x + y + z - 12
h = x ** 2 + y ** 2 - z

lamda, mu = sp.symbols('lamda, mu')
L = f - lamda * g - mu * h

gradL = [sp.diff(L, var) for var in [x, y, z]]
eqs = gradL + [g] + [h]

solution = sp.solve(eqs, [x, y, z, lamda, mu], dict=True)
print(solution)

rational = lambda x: all(i.exp.is_Integer for i in x.atoms(sp.Pow))

save = []
for p in solution:
    if (rational(p[x]) and rational(p[y]) and rational(p[z])):
        save.append([f.subs(p), p[x], p[y], p[z]])

save.sort(key=lambda x: x[0])
print(save[0])