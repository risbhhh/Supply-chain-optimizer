"""A simple inventory ordering optimizer using OR-Tools linear solver.
Given forecasted demand for a horizon, decide order quantity at t=0 to minimize expected cost.
This is a simplified demo: single period or limited horizon; extend for multi-period decisions.
"""
from ortools.linear_solver import pywraplp
import numpy as np


def optimize_orders(forecasts, on_hand, lead_time=0, max_order=1000, holding_cost=0.5, order_cost=10, stockout_cost=5):
    # forecasts: array of demand for next H periods
    H = len(forecasts)
    solver = pywraplp.Solver.CreateSolver('GLOP') or pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        raise RuntimeError("Could not create OR-Tools solver")

    # decision: order quantity at t=0
    q = solver.NumVar(0.0, max_order, 'q')

    # expected inventory level after demand (simplified single period): inventory = on_hand + q - demand_sum
    expected_inv = on_hand + q - sum(forecasts)

    # costs
    holding = holding_cost * solver.Max(expected_inv, 0)
    # OR-Tools GLOP doesn't support Max directly; approximate by introducing variable
    inv_pos = solver.NumVar(0.0, solver.infinity(), 'inv_pos')
    # constraints: inv_pos >= expected_inv and inv_pos >= 0
    solver.Add(inv_pos >= expected_inv)
    solver.Add(inv_pos >= 0)

    holding_cost_expr = holding_cost * inv_pos
    order_cost_expr = order_cost * solver.Min(q, 1)  # if q>0 then incur fixed order cost (approx)
    # Min not supported: approximate with binary would be needed; for demo use proportional
    order_cost_expr = order_cost * (q / (q + 1e-6))

    stockout = stockout_cost * solver.Max(-expected_inv, 0)
    short_pos = solver.NumVar(0.0, solver.infinity(), 'short_pos')
    solver.Add(short_pos >= -expected_inv)
    solver.Add(short_pos >= 0)
    stockout_cost_expr = stockout_cost * short_pos

    total_cost = holding_cost_expr + order_cost_expr + stockout_cost_expr

    solver.Minimize(total_cost)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Solver failed")

    return {'order_qty': q.solution_value(), 'expected_inventory': expected_inv, 'objective': solver.Objective().Value()}


if __name__ == '__main__':
    res = optimize_orders([30, 28, 25], on_hand=40)
    print(res)
