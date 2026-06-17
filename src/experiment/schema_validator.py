"""Deterministic schema validation for the fixed extraction schema."""

REQUIRED_STEP_FIELDS = {"operation", "machine", "duration", "precondition", "postcondition"}


def validate_extraction(data):
    """Validate that data follows the fixed schema structure.

    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Top-level must be a dict"

    if "steps" not in data:
        return False, "Missing 'steps' field"

    steps = data["steps"]
    if not isinstance(steps, list):
        return False, "'steps' must be a list"

    if len(steps) == 0:
        return False, "'steps' is empty"

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {i}: not a dict"

        missing = REQUIRED_STEP_FIELDS - set(step.keys())
        if missing:
            return False, f"Step {i}: missing fields {missing}"

        # duration must contain a number
        duration_str = str(step.get("duration", ""))
        if not any(c.isdigit() for c in duration_str):
            return False, f"Step {i}: duration '{duration_str}' has no numeric value"

        # machine must be non-empty
        if not str(step.get("machine", "")).strip():
            return False, f"Step {i}: machine is empty"

        # precondition/postcondition must be lists
        for field in ("precondition", "postcondition"):
            val = step.get(field)
            if not isinstance(val, list):
                return False, f"Step {i}: '{field}' must be a list"

    return True, ""
