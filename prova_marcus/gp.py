import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy
from kb import * 
from structure import *
from utils import *



population_size = 100
generations = 100
max_depth = 5

# Costanti
costanti = {
    "Marcus": ltn.core.Constant(torch.randn(2), trainable=False),
    "Tweety": ltn.core.Constant(torch.randn(2), trainable=False),
}

tmp_x = torch.stack([costanti[i].value for i in costanti.keys()], dim=0)

predicati = {
    "Fly": ltn.core.Predicate(model=make_unary_predicate()),
    "Animal": ltn.core.Predicate(model=make_unary_predicate()),
    "Bird": ltn.core.Predicate(model=make_unary_predicate()),
    "Penguin": ltn.core.Predicate(model=make_unary_predicate()),
    "Swallow": ltn.core.Predicate(model=make_unary_predicate()),
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



# prendo le chiavi e faccio le variabili
OPERATORS = [k for k in operatori.keys()]
QUANTIFIERS = [k for k in quantificatori.keys()]
PREDICATES = [k for k in predicati.keys()]
VARIABLES = [k for k in variabili.keys()]


# Setup
kb_rules, kb_facts = create_kb(predicati, quantificatori, operatori, costanti)
optimizer, costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts = setup_ltn(costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts)

# Unisco predicati, quantificatori e operatori in un unico dizionario
ltn_dict = {}
ltn_dict.update(costanti)
ltn_dict.update(predicati)
ltn_dict.update(quantificatori)
ltn_dict.update(operatori)

matrix_size = int(np.sqrt(population_size))


print(matrix_size)
popolazione = np.array([
    [
        [Albero(VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES), 0] # individuo e fitness --> massimizzare
        for _ in range(matrix_size)
    ] for _ in range(matrix_size)
])

# Esecuzione

popolazione_finale = evolutionary_run(
    popolazione,
    generations=generations,
    ltn_dict=ltn_dict,
    variabili=variabili,
    predicati=predicati,
    costanti = costanti,
    ottimizzatore = optimizer,
    kb_rules= kb_rules, 
    kb_facts = kb_facts
)

# Ordinamento della popolazione in base alla fitness in ordine decrescente
popolazione_ordinata = sorted(
    (individuo for row in popolazione_finale for individuo in row),
    key=lambda x: x[1],  # Ordina per fitness
    reverse=True  # Ordine decrescente
)

# Miglior individuo finale
migliore = popolazione_ordinata[0]

print(f"\n--- Risultati Finali ---")
print(popolazione_ordinata)
print(f"Miglior individuo finale: {migliore[0]}")
print(f"Fitness: {migliore[1]}")