
import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import numpy as np
from tree import *
from utils import *
from parser import *
from kb import *
from evo_funct import *


# Define the parameters for the Genetic Algorithm
population_size = 9  # Set the population size
generations = 10     # Set the number of generations
num_offspring = 5     # Set the number of offspring per generation
is_matrix = True

# Define the selection method (you can choose from your available methods)
metodo = fitness_proportionate_selection  # Use fitness_proportionate_selection for selection


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

# Initialize Population
popolazione = popolazione_init(population_size=population_size, 
                           is_matrix=is_matrix, 
                           PREDICATES=PREDICATES, 
                           QUANTIFIERS=QUANTIFIERS, 
                           OPERATORS=OPERATORS, 
                           VARIABLES=VARIABLES, 
                           ltn_dict={**predicati, **quantificatori, **operatori}, 
                           variabili=variabili)

print(popolazione)
# Esecuzione
popolazione_finale = evolutionary_run_GA(
        popolazione,
        generations=generations,
        ltn_dict={**predicati, **quantificatori, **operatori},
        variabili=variabili,
        operatori=operatori,
        metodo=metodo,
        is_matrix=is_matrix,
        population_size=population_size,
        num_offspring=num_offspring
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
migliori = popolazione_ordinata[:5]
print('Popolazione finale:')
print(popolazione_finale)

print('Popolazione ordinata:')
print(popolazione_ordinata)

for migliore in migliori:
    print(f"\n--- Risultati Finali ---")
    print(f"Miglior individuo finale: {migliore[0]}")
    print(f"Fitness: {migliore[1]}")