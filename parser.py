

#################################################################
# Parsing di predicato multi-argomento: "Dog(x,y,z)" -> ("Dog", ["x","y","z"])
#################################################################

def parse_predicato(value_str):
    '''
    Parses a predicate string into its name and a list of arguments.

    This function takes a string representation of a predicate, such as "Dog(x)" or "Parent(x, y)",
    and extracts the predicate's name and its arguments.

    Parameters:
    - value_str (str): The string representation of the predicate.

    Returns:
    - tuple: A tuple containing:
        - pred_name (str): The name of the predicate.
        - arg_list (list): A list of argument names (strings).

    Raises:
    - ValueError: If the input string is malformed (e.g., missing parentheses).

    Examples:
    -parse_predicato("Dog(x)")
        ("Dog", ["x"])
    -parse_predicato("Parent(x, y)")
        ("Parent", ["x", "y"])
    -parse_predicato("Likes(x, y, z)")
        ("Likes", ["x", "y", "z"])
    '''
    if not value_str.endswith(")"):
        raise ValueError(f"Predicato malformato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    args_str = value_str[idx_par+1:-1]  # es. "x,y"
    arg_list = [arg.strip() for arg in args_str.split(",")]
    # arg_list es. ["x","y"]
    return pred_name, arg_list