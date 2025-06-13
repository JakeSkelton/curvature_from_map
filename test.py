'''
File: test.py
Created Date: 13 Jun 2025
Author: Jake Skelton
Date Modified: Fri Jun 13 2025
Copyright (c): 2025 Jake Skelton
'''

import sympy as sp
import einsteinpy.symbolic as ep

# Symbols
a11, a12, a21, a22, b1, b2, x1, x2 = sp.symbols(
    ['a11', 'a12', 'a21', 'a22', 'b1', 'b2', 'x1', 'x2'], real=True)
x = sp.Array([x1, x2])
# Coordinate map
map = sp.Array([a11*x1 + a12*x2 + b1*x1*x2,
                a21*x1 + a22*x2 + b2*x1*x2])
jac = sp.Matrix(map.diff(x)).transpose()
jact = jac.transpose()
# Pull-back metric
metric = sp.simplify(jact * jac)
emetric = ep.MetricTensor(metric, [x1, x2])
# Curvature
R = ep.RicciScalar.from_metric(emetric)
print("Curvature of pull-back metric: %s"%R.tensor())
