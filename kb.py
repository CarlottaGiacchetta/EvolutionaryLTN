import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gp_main import * 

import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
from gp_main import Nodo, build_ltn_formula_node, make_unary_predicate

# Definizione della KB
def create_kb():
    kb_formulas = []

    # Formula 1: FORALL x: Cat(x) -> HasWhiskers(x)
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    has_whiskers_x = Nodo("PREDICATO", "HasWhiskers(x)")
    implies_formula = Nodo("OPERATORE", "IMPLIES", [cat_x, has_whiskers_x])
    formula1 = Nodo("QUANTIFICATORE", "FORALL", [var_x, implies_formula])
    kb_formulas.append(formula1)

    # Formula 2: FORALL x: Dog(x) -> NOT(Cat(x))
    dog_x = Nodo("PREDICATO", "Dog(x)")
    not_cat_x = Nodo("OPERATORE", "NOT", [cat_x])
    implies_formula2 = Nodo("OPERATORE", "IMPLIES", [dog_x, not_cat_x])
    formula2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, implies_formula2])
    kb_formulas.append(formula2)

    # Formula 3: EXISTS x: Cat(x) OR Dog(x)
    or_formula = Nodo("OPERATORE", "OR", [cat_x, dog_x])
    formula3 = Nodo("QUANTIFICATORE", "EXISTS", [var_x, or_formula])
    kb_formulas.append(formula3)

    return kb_formulas

# Setup LTN
def setup_ltn(kb_formulas):
    constants = {
        "Fluffy": ltn.Constant(torch.randn(2), trainable=False),
        "Garfield": ltn.Constant(torch.randn(2), trainable=False),
        "Rex": ltn.Constant(torch.randn(2), trainable=False),
    }

    tmp_x = torch.stack([constants[name].value for name in constants.keys()], dim=0)

    variables = {
        "x": ltn.Variable("x", tmp_x, add_batch_dim=False),
    }

    predicates = {
        "Cat": ltn.Predicate(make_unary_predicate()),
        "Dog": ltn.Predicate(make_unary_predicate()),
        "HasWhiskers": ltn.Predicate(make_unary_predicate()),
    }

    quantifiers = {
        "FORALL": ltn.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "EXISTS": ltn.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier="e"),
    }

    operators = {
        "AND": ltn.Connective(fuzzy_ops.AndProd()),
        "OR": ltn.Connective(fuzzy_ops.OrMax()),
        "IMPLIES": ltn.Connective(fuzzy_ops.ImpliesLuk()),
        "NOT": ltn.Connective(fuzzy_ops.NotStandard()),
    }

    ltn_dict = {**constants, **predicates, **quantifiers, **operators}
    # Ottimizzatore
    parameters = []
    for pred in predicates.values():
        parameters += list(pred.model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # Training Loop
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = kb_loss(kb_formulas, ltn_dict, variables)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Valutazione Finale
    print("\n--- Valutazione Finale ---")
    for i, formula in enumerate(kb_formulas, start=1):
        ltn_formula = build_ltn_formula_node(formula, ltn_dict, variables)
        satisfaction = ltn_formula.value.mean().item()
        print(f"Formula {i}: {formula}")
        print(f"Soddisfazione: {satisfaction:.4f}")
    
    return ltn_dict, variables


    

# Loss Function
def kb_loss(kb_formulas, ltn_dict, variables):
    total_loss = 0
    for formula in kb_formulas:
        ltn_formula = build_ltn_formula_node(formula, ltn_dict, variables)
        total_loss += (1 - ltn_formula.value.mean())
    return total_loss
