'''
File: ttmap.py
Created Date: 13 Jun 2025
Author: Jake Skelton
Date Modified: Fri Jun 13 2025
Copyright (c): 2025 Jake Skelton
'''

import sympy as sp
from main import curvature_from_map

sp.init_printing()

x1, x2 = sp.symbols(['x1', 'x2'], real=True)
d = sp.Function('d', real=True)(x1)
e = sp.Function('e', real=True)(x1)


am = sp.asinh((1-d)/e)
ap = sp.asinh((1+d)/e)
g = d + e * sp.sinh((am + ap)*(x2-1)/2 + am)

map = sp.Array([x1, g])
variables = [x1, x2]
curvature = curvature_from_map(map, variables, simplify=False)
