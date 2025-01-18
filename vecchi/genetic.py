import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gp_main import * 


ltn.device = torch.device("cpu")


population_size = 50
generations = 10
max_depth = 5

# Costanti

costanti = {
    "Fluffy": ltn.core.Constant(torch.randn(2), trainable=False),
    "Garfield": ltn.core.Constant(torch.randn(2), trainable=False),
    "Rex": ltn.core.Constant(torch.randn(2), trainable=False)
}

tmp_x = torch.stack([costanti[i].value for i in costanti.keys()], dim=0)

predicati = {
    "Cat": ltn.core.Predicate(model=make_unary_predicate()),
    "Dog": ltn.core.Predicate(model=make_unary_predicate()),
    "HasWhiskers": ltn.core.Predicate(model=make_unary_predicate()),
    "IsParent": ltn.core.Predicate(model=make_unary_predicate()),
    "DogLover": ltn.core.Predicate(model=make_unary_predicate()),
    "HateAllAnimals": ltn.core.Predicate(model=make_unary_predicate()),
}

quantificatori = {
    # quantificatori
    "FORALL": ltn.core.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier='f'),
    "EXISTS": ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')
}

operatori = {
    # operatori
    "AND": ltn.core.Connective(fuzzy_ops.AndProd()),
    "OR": ltn.core.Connective(fuzzy_ops.OrMax()),
    "IMPLIES": ltn.core.Connective(fuzzy_ops.ImpliesLuk()),
    "NOT": ltn.core.Connective(fuzzy_ops.NotStandard()),
}

# Scope vars, ad esempio:
variabili = {
    "x": ltn.core.Variable("x", tmp_x, add_batch_dim=False),
    "y": ltn.core.Variable("y", tmp_x, add_batch_dim=False)
}

# Unisco predicati, quantificatori e operatori in un unico dizionario
ltn_dict = {}
ltn_dict.update(costanti)
ltn_dict.update(predicati)
ltn_dict.update(quantificatori)

# prendo le chiavi e faccio le variabili
OPERATORS = [k for k in operatori.keys()]
QUANTIFIERS = [k for k in quantificatori.keys()]
PREDICATES = [k for k in predicati.keys()]
VARIABLES = [k for k in variabili.keys()]



matrix_size = int(np.sqrt(population_size))
popolazione = np.array([
    [
        [Albero(VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES), 0] # individuo e fitness --> massimizzare
        for _ in range(matrix_size)
    ] for _ in range(matrix_size)
])

for i in range(matrix_size):
    for j in range(matrix_size):
        print(popolazione[i, j], end="\t")
        print(get_neighbors(popolazione, i, j), end="\n\n")
        exit()