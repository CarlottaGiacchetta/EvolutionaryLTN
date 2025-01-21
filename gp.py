import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy
from kb import * 
from structure import *
from utils import *

# Setup
kb_formulas = create_kb()
ltn_dict, variables = setup_ltn(kb_formulas)
is_matrix = False
population_size = 49
generations = 100
max_depth = 5
num_offspring = 5
metodi = [fitness_proportionate_selection, fitness_proportionate_selection_modern]
metodo = fitness_proportionate_selection


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
    #"IsParent": ltn.core.Predicate(model=make_unary_predicate()),
    #"DogLover": ltn.core.Predicate(model=make_unary_predicate()),
    #"HateAllAnimals": ltn.core.Predicate(model=make_unary_predicate()),
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
    #"y": ltn.core.Variable("y", tmp_x, add_batch_dim=False)
}

# Unisco predicati, quantificatori e operatori in un unico dizionario
ltn_dict = {}
ltn_dict.update(costanti)
ltn_dict.update(predicati)
ltn_dict.update(quantificatori)
ltn_dict.update(operatori)

# prendo le chiavi e faccio le variabili
OPERATORS = [k for k in operatori.keys()]
QUANTIFIERS = [k for k in quantificatori.keys()]
PREDICATES = [k for k in predicati.keys()]
VARIABLES = [k for k in variabili.keys()]



popolazione = popolazione_init(population_size=population_size, is_matrix=is_matrix, predicati=predicati, quantificatori=quantificatori, operatori=operatori, variabili=variabili)

# Esecuzione
popolazione_finale = evolutionary_run_GP(
    popolazione,
    generations=generations,
    ltn_dict={**predicati, **quantificatori, **operatori},
    variabili=variabili,
    operatori=operatori,
    metodo=metodo,
    is_matrix=is_matrix,
    num_offspring=num_offspring,
    kb_formulas=kb_formulas
)


if is_matrix:
    # Ordinamento della popolazione in base alla fitness in ordine decrescente
    popolazione_ordinata = sorted(
        (individuo for row in popolazione_finale for individuo in row),
        key=lambda x: x[1],  # Ordina per fitness
        reverse=True  # Ordine decrescente
    )

else:
    # Ordinamento della popolazione in base alla fitness in ordine decrescente
    popolazione_ordinata = sorted(
        (individuo for individuo in popolazione_finale),
        key=lambda x: x[1],  # Ordina per fitness
        reverse=True  # Ordine decrescente
    )

# Miglior individuo finale
migliori = popolazione_ordinata[0:5]
print('popolazione finale\n', popolazione_finale)
print('popolazione ordinata\n', popolazione_ordinata)

for migliore in migliori:
    print(f"\n--- Risultati Finali ---")
    
    print(f"Miglior individuo finale: {migliore[0]}")
    print(f"Fitness: {migliore[1]}")