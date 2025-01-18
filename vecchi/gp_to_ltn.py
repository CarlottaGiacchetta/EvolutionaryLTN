import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gp_main import * 


torch.autograd.set_detect_anomaly(True)
ltn.device = torch.device("cpu")  # or "cuda" if available

# Create random embeddings for the 3 individuals
Fluffy_tensor = torch.randn(2)
Garfield_tensor = torch.randn(2)
Rex_tensor = torch.randn(2)

# Define LTN constants
Fluffy = ltn.core.Constant(Fluffy_tensor, trainable=False)
Garfield = ltn.core.Constant(Garfield_tensor, trainable=False)
Rex = ltn.core.Constant(Rex_tensor, trainable=False)

# Create a variable "x" enumerating the 3 individuals
all_inds = torch.stack([Fluffy.value, Garfield.value, Rex.value], dim=0)  # shape [3,2]
x = ltn.core.Variable("x", all_inds, add_batch_dim=False)

# Example of a small MLP for a unary predicate
def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
    return nn.Sequential(
        nn.Linear(in_features, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid()  # output in [0,1]
    )

Cat_model = make_unary_predicate()  # Cat(x) = [0,1]
Dog_model = make_unary_predicate()  # Dog(x) = [0,1]
HasWhiskers_model = make_unary_predicate()  # HasWhiskers(x) = [0,1]

Cat = ltn.core.Predicate(model=Cat_model)
Dog = ltn.core.Predicate(model=Dog_model)
HasWhiskers = ltn.core.Predicate(model=HasWhiskers_model)

# Costruisci il dizionario ltn_dict
ltn_dict = {
    # predicati
    "Cat": Cat,
    "Dog": Dog,
    "HasWhiskers": HasWhiskers,
    # operatori
    "AND": ltn.core.Connective(fuzzy_ops.AndProd()),
    "OR": ltn.core.Connective(fuzzy_ops.OrMax()),
    "IMPLIES": ltn.core.Connective(fuzzy_ops.ImpliesLuk()),
    "NOT": ltn.core.Connective(fuzzy_ops.NotStandard()),
    # quantificatori
    "FORALL": ltn.core.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier='f'),
    "EXISTS": ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')
}

# Scope vars, ad esempio:
scope_vars = {
    "x": x,    # x è ltn.core.Variable("x", all_inds, add_batch_dim=False)
}


a1 = Albero()
print("Albero1 iniziale:", a1, "profondità=", a1.profondita, "valido?", a1.valida_albero())
a2 = Albero()
print("Albero2 iniziale:", a2, "profondità=", a2.profondita, "valido?", a2.valida_albero())

c1, c2 = crossover(a1, a2, prob=0.9)
print("\n-- CROSSOVER --")
print("Child1:", c1, "depth=", c1.profondita, "valid?", c1.valida_albero())
print("Child2:", c2, "depth=", c2.profondita, "valid?", c2.valida_albero())

c3, c4 = crossover(c1, c2, prob=0.9)
c5, c6 = crossover(c3, c4, prob=0.9)
print("\n-- CROSSOVER --")
print("Child3:", c3, "depth=", c3.profondita, "valid?", c3.valida_albero())
print("Child4:", c4, "depth=", c4.profondita, "valid?", c4.valida_albero())
print("Child5:", c5, "depth=", c5.profondita, "valid?", c5.valida_albero())
print("Child6:", c6, "depth=", c6.profondita, "valid?", c6.valida_albero())

m1 = mutate(c1, prob=0.9)
print("\n-- MUTATION Child1 --")
print("Mutated:", m1, "depth=", m1.profondita, "valid?", m1.valida_albero())

ltn_formula = m1.to_ltn_formula(ltn_dict, scope_vars)

print(ltn_formula)
