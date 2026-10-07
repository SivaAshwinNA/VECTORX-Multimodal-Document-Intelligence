def basic_numeric_reasoning(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "maximum": max(values),
        "minimum": min(values),
        "difference": max(values) - min(values),
        "average": sum(values) / len(values),
    }
