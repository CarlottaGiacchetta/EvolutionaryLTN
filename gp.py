import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import numpy as np
from tree import *
from utils import *
from parser import *
from kb import *
from evo_funct import *



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


is_matrix = True
population_size = 1000
generations = 100
max_depth = 20
num_offspring = 20
metodi = [fitness_proportionate_selection, fitness_proportionate_selection_modern]
metodo = fitness_proportionate_selection



run_matrixfitprop = {
        'run1': {'is_matrix': True,  'population_size': 16, 'generations': 20, 'num_offspring': 5,   'metodo': fitness_proportionate_selection},
        'run2': {'is_matrix': True,  'population_size': 16, 'generations': 50, 'num_offspring': 5,   'metodo': fitness_proportionate_selection},
        'run3': {'is_matrix': True, 'population_size': 49, 'generations': 20, 'num_offspring': 20,   'metodo': fitness_proportionate_selection},
        'run4': {'is_matrix': True, 'population_size': 49, 'generations': 50, 'num_offspring': 20,   'metodo': fitness_proportionate_selection},
        'run5': {'is_matrix': True,  'population_size': 16, 'generations': 100, 'num_offspring': 5,  'metodo': fitness_proportionate_selection},
        'run6': {'is_matrix': True, 'population_size': 49, 'generations': 100, 'num_offspring': 20,  'metodo': fitness_proportionate_selection},
        'run7': {'is_matrix': True, 'population_size': 100, 'generations': 100, 'num_offspring': 40, 'metodo': fitness_proportionate_selection}}

run_matrixoversel = {
        'run1': {'is_matrix': True,  'population_size': 16, 'generations': 20, 'num_offspring': 5,   'metodo': fitness_proportionate_selection_modern},
        'run2': {'is_matrix': True,  'population_size': 16, 'generations': 50, 'num_offspring': 5,   'metodo': fitness_proportionate_selection_modern},
        'run3': {'is_matrix': True, 'population_size': 49, 'generations': 20, 'num_offspring': 20,   'metodo': fitness_proportionate_selection_modern},
        'run4': {'is_matrix': True, 'population_size': 49, 'generations': 50, 'num_offspring': 20,   'metodo': fitness_proportionate_selection_modern},
        'run5': {'is_matrix': True,  'population_size': 16, 'generations': 100, 'num_offspring': 5,  'metodo': fitness_proportionate_selection_modern},
        'run6': {'is_matrix': True, 'population_size': 49, 'generations': 100, 'num_offspring': 20,  'metodo': fitness_proportionate_selection_modern},
        'run7': {'is_matrix': True, 'population_size': 100, 'generations': 100, 'num_offspring': 40, 'metodo': fitness_proportionate_selection_modern}}

run_listfitprop = {
        'run1': {'is_matrix': False,  'population_size': 16, 'generations': 20, 'num_offspring': 5,   'metodo': fitness_proportionate_selection},
        'run2': {'is_matrix': False,  'population_size': 16, 'generations': 50, 'num_offspring': 5,   'metodo': fitness_proportionate_selection},
        'run3': {'is_matrix': False, 'population_size': 49, 'generations': 20, 'num_offspring': 20,   'metodo': fitness_proportionate_selection},
        'run4': {'is_matrix': False, 'population_size': 49, 'generations': 50, 'num_offspring': 20,   'metodo': fitness_proportionate_selection},
        'run5': {'is_matrix': False,  'population_size': 16, 'generations': 100, 'num_offspring': 5,  'metodo': fitness_proportionate_selection},
        'run6': {'is_matrix': False, 'population_size': 49, 'generations': 100, 'num_offspring': 20,  'metodo': fitness_proportionate_selection},
        'run7': {'is_matrix': False, 'population_size': 100, 'generations': 100, 'num_offspring': 40, 'metodo': fitness_proportionate_selection}}

run_listoversel = {
        'run1': {'is_matrix': False,  'population_size': 16, 'generations': 20, 'num_offspring': 5,   'metodo': fitness_proportionate_selection_modern},
        'run2': {'is_matrix': False,  'population_size': 16, 'generations': 50, 'num_offspring': 5,   'metodo': fitness_proportionate_selection_modern},
        'run3': {'is_matrix': False, 'population_size': 49, 'generations': 20, 'num_offspring': 20,   'metodo': fitness_proportionate_selection_modern},
        'run4': {'is_matrix': False, 'population_size': 49, 'generations': 50, 'num_offspring': 20,   'metodo': fitness_proportionate_selection_modern},
        'run5': {'is_matrix': False,  'population_size': 16, 'generations': 100, 'num_offspring': 5,  'metodo': fitness_proportionate_selection_modern},
        'run6': {'is_matrix': False, 'population_size': 49, 'generations': 100, 'num_offspring': 20,  'metodo': fitness_proportionate_selection_modern},
        'run7': {'is_matrix': False, 'population_size': 100, 'generations': 100, 'num_offspring': 40, 'metodo': fitness_proportionate_selection_modern}}

runs = run_listoversel

for run in runs:
    print('\n\n\n\n\n')
    print('--------------------------')
    print('\n')
    print(run, runs[run]) 
    print('\n')
    
    is_matrix = runs[run]['is_matrix']
    population_size = runs[run]['population_size']
    generations = runs[run]['generations']
    num_offspring = runs[run]['num_offspring']
    metodo = runs[run]['metodo']   

    popolazione = popolazione_init(population_size=population_size, 
                               is_matrix=is_matrix, 
                               PREDICATES=PREDICATES, 
                               QUANTIFIERS=QUANTIFIERS, 
                               OPERATORS=OPERATORS, 
                               VARIABLES=VARIABLES, 
                               ltn_dict={**predicati, **quantificatori, **operatori}, 
                               variabili=variabili)

    # Esecuzione
    popolazione_finale = evolutionary_run_GP(
        popolazione,
        generations=generations,
        ltn_dict={**predicati, **quantificatori, **operatori},
        variabili=variabili,
        operatori=operatori,
        metodo=metodo,
        is_matrix=is_matrix,
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


    migliori = []
    # Miglior individuo finale
    for pop in popolazione_ordinata:
        if pop[1]>=1:
            migliori.append(pop[0])


    migliori = list(set(migliori))
    print(len(migliori))



    for individuo in migliori:
        pred = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
        formula = individuo.to_ltn_formula(ltn_dict, variabili)
        print(individuo, formula.value.item())
