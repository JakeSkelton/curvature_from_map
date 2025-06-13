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
a1, b1, a2, b2, x1, x2 = sp.symbols(['a1', 'b1', 'a2', 'b2', 'x1', 'x2'])
x = sp.Array([x1, x2])
# Coordinate map
map = sp.Array([a1*x2 + b1*x1*x2, a2*x1 + b2*x1*x2])
jac = sp.Matrix(map.diff(x)).transpose()
jact = jac.transpose()
# Pull-back metric
metric = sp.simplify(jact * jac)
emetric = ep.MetricTensor(metric, [x1, x2])
# Curvature
R = ep.RicciScalar.from_metric(emetric)
print("Curvature of pull-back metric: %s"%R.tensor())
