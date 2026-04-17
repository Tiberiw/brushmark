def ema_smoothing(values: list[float], beta: float = 0.9) -> list[float]:
    """Function for computing the Exponential Moving Average of a list.
        Uses the bias correction formula
    """
    results = []
    v = 0
    for t, val in enumerate(values, start=1):
        v = beta * v + (1 - beta) * val
        results.append(v / (1 - beta**t))
    return results