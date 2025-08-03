async def exact_match(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    Returns True if the output matches the expected output exactly.
    """
    return outputs == reference_outputs

# Mapping of builtin code evaluators
CODE_EVALUATORS = {
    "ExactMatch": exact_match,
    # Add more builtins here as needed
} 