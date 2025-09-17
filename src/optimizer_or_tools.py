"""
Supply Chain Optimization using OR-Tools.

Takes demand forecasts and decides optimal order quantities 
to minimize holding + stockout + ordering costs.
"""

from ortools.linear_solver import pywraplp


def optimize_inventory(
    forecasts,
    on_hand=100,
    holding_cost=1.0,
    stockout_cost=5.0,
    order_cost=50.0,
    max_order=1000,
):
    """
    Optimize inventory decisions for the forecast horizon.
    
    Args:
        forecasts (list): forecasted demand per period
        on_hand (int): current inventory
        holding_cost (float): per-unit holding cost
        stockout_cost (float): per-unit stockout penalty
        order_cost (float): fixed cost if an order is placed
        max_order (int): maximum order quantity allowed
    
    Returns:
        dict: optimal order plan + costs
    """
    H = len(forecasts)
    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("Could not initialize solver.")

    # Decision variables
    order = [solver.IntVar(0, max_order, f"order_{t}") for t in range(H)]
    inventory = [solver.NumVar(0, solver.infinity(), f"inv_{t}") for t in range(H)]
    shortage = [solver.NumVar(0, solver.infinity(), f"short_{t}") for t in range(H)]
    order_indicator = [solver.IntVar(0, 1, f"y_{t}") for t in range(H)]  # binary for order cost

    # Initial inventory balance
    solver.Add(inventory[0] == on_hand + order[0] - forecasts[0] + shortage[0])

    # Balance constraints for subsequent periods
    for t in range(1, H):
        solver.Add(inventory[t] == inventory[t - 1] + order[t] - forecasts[t] + shortage[t])

    # Link order quantities to indicator (big-M formulation)
    M = max_order
    for t in range(H):
        solver.Add(order[t] <= M * order_indicator[t])

    # Objective: minimize total cost
    total_cost = (
        holding_cost * sum(inventory)
        + stockout_cost * sum(shortage)
        + order_cost * sum(order_indicator)
    )
    solver.Minimize(total_cost)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found.")

    return {
        "orders": [int(order[t].solution_value()) for t in range(H)],
        "inventory": [inventory[t].solution_value() for t in range(H)],
        "shortages": [shortage[t].solution_value() for t in range(H)],
        "objective": solver.Objective().Value(),
    }


if __name__ == "__main__":
    sample_forecasts = [30, 25, 20, 35, 40]
    result = optimize_inventory(sample_forecasts, on_hand=50)
    print("Optimal Plan:", result)
