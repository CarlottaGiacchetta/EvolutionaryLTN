

#################################################################
# Parsing di predicato multi-argomento: "Dog(x,y,z)" -> ("Dog", ["x","y","z"])
#################################################################

def parse_predicato(value_str):
    """
    Esempi:
    "Dog(x)"               -> ("Dog", ["x"])
    "Parent(x, y)"         -> ("Parent", ["x","y"])
    "Likes(x, y, z)"       -> ("Likes", ["x","y","z"])
    """
    if not value_str.endswith(")"):
        raise ValueError(f"Predicato malformato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    args_str = value_str[idx_par+1:-1]  # es. "x,y"
    arg_list = [arg.strip() for arg in args_str.split(",")]
    # arg_list es. ["x","y"]
    return pred_name, arg_list