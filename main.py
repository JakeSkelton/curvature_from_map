'''
File: main.py
Created Date: 13 Jun 2025
Author: Jake Skelton
Date Modified: Fri Jun 13 2025
Copyright (c): 2025 Jake Skelton
'''

import sympy as sp
import einsteinpy.symbolic as ep
from typing import TypedDict


class CurvatureInfo(TypedDict):
    Riemann: ep.RiemannCurvatureTensor
    Ricci: ep.RicciTensor
    Scalar: ep.RicciScalar


def curvature_from_map(map: sp.Array, variables: list, simplify: bool = True
                       ) -> CurvatureInfo:
    """
    Calculate the curvature of a pull-back metric from a coordinate map.

    Parameters:
    - map: sp.Array representing the coordinate map, as 1D vector.
    - variables: list of variable names, as sympy symbols, used in the map.

    Returns:
    - sp.Array representing the Riemann curvature of the pull-back metric.
    """
    x = sp.Array(variables)
    jac = sp.Matrix(map.diff(x)).transpose()
    jact = jac.transpose()
    metric = jact * jac
    if simplify:
        metric = sp.simplify(metric, inverse=True)
    emetric = ep.MetricTensor(metric, variables)
    riemann = ep.RiemannCurvatureTensor.from_metric(emetric)
    ricci = ep.RicciTensor.from_riemann(riemann)

    return {"Riemann": riemann,
            "Ricci": ricci,
            "Scalar": ep.RicciScalar.from_riccitensor(ricci)}
