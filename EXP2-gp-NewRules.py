import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import numpy as np
from copy import deepcopy
from kb import * 
from utils import *
from evo_funct import *

population_size = 100
generations = 100
max_depth = 5
is_matrix = True
metodo = fitness_proportionate_selection

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
    "FORALL": ltn.core.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier='f'),
    "EXISTS": ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')
}

operatori = {
    "AND": ltn.core.Connective(fuzzy_ops.AndProd()),
    "OR": ltn.core.Connective(fuzzy_ops.OrMax()),
    "IMPLIES": ltn.core.Connective(fuzzy_ops.ImpliesLuk()),
    "NOT": ltn.core.Connective(fuzzy_ops.NotStandard()),
}

# Scope vars
variabili = {
    "x": ltn.core.Variable("x", tmp_x, add_batch_dim=False),
    # "y": ltn.core.Variable("y", tmp_x, add_batch_dim=False)
}

# Liste di simboli
OPERATORS = list(operatori.keys())
QUANTIFIERS = list(quantificatori.keys())
PREDICATES = list(predicati.keys())
VARIABLES = list(variabili.keys())

# Setup: crea regole e fatti, allena i predicati
kb_rules, kb_facts = create_kb(predicati, quantificatori, operatori, costanti)
optimizer, costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts = setup_ltn(
    costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts
)

# Dizionario con oggetti LTN
ltn_dict = {}
ltn_dict.update(costanti)
ltn_dict.update(predicati)
ltn_dict.update(quantificatori)
ltn_dict.update(operatori)

# Popolazione (matrice sqrt x sqrt)
matrix_size = int(np.sqrt(population_size))


quantificatori_2 = {
    "FORALL": ltn.core.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier='f'),
}
PROVA = list(quantificatori_2.keys())

popolazione = popolazione_init(population_size=population_size, 
                               is_matrix=is_matrix, 
                               PREDICATES=PREDICATES, 
                               QUANTIFIERS=PROVA, 
                               OPERATORS=OPERATORS, 
                               VARIABLES=VARIABLES, 
                               ltn_dict={**predicati, **quantificatori, **operatori}, 
                               variabili=variabili)

# Calcoliamo la soddisfazione di base (senza formule evolutive)
baseline_sat = measure_kb_sat(kb_rules, kb_facts, variabili, costanti)
print("Iniziamo con baseline SAT della KB fissa:", baseline_sat)

# Esecuzione evolutiva
popolazione_finale = evolutionary_run(
    popolazione,
    generations=generations,
    ltn_dict=ltn_dict,
    variabili=variabili,
    predicati=predicati,
    costanti=costanti,
    kb_rules=kb_rules,
    kb_facts=kb_facts,
    baseline_sat=baseline_sat,
    is_matrix=is_matrix,
    metodo=metodo
)

# Ordiniamo la popolazione in base alla fitness
popolazione_ordinata = sorted(
    (individuo for row in popolazione_finale for individuo in row),
    key=lambda x: x[1],
    reverse=True
)

# Miglior individuo finale
migliore = popolazione_ordinata[0]
print(f"\n--- Risultati Finali ---")
print(f"Miglior individuo finale: {migliore[0]}")
print(f"Fitness: {migliore[1]}")
