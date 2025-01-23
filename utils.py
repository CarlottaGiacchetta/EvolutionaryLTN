from kb import * 
from structure import *
import numpy as np


def compute_fitness(popolazione, ltn_dict, variabili, is_matrix):
    if is_matrix:
        # Calcola la fitness per ogni individuo
        for i in range(popolazione.shape[0]):
            for j in range(popolazione.shape[1]):
                individuo = popolazione[i, j][0]  
                predicati = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
                formula = individuo.to_ltn_formula(ltn_dict, variabili)
                fitness = formula.value.item()
            
                # Penalizza se ci sono duplicati
                if len(predicati) != len(set(predicati)):
                    fitness *= 0.6
                popolazione[i, j][1] = fitness 
               
    else:
        for i in range(len(popolazione)):
            individuo = popolazione[i][0]  # Albero
            predicati = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
            formula = individuo.to_ltn_formula(ltn_dict, variabili)
            fitness = formula.value.item()
            
            if len(predicati) != len(set(predicati)):
                fitness *= 0.6
            popolazione[i][1] = fitness  # Aggiorna fitness
    return popolazione


def popolazione_init(population_size, is_matrix, PREDICATES, QUANTIFIERS, OPERATORS, VARIABLES, ltn_dict, variabili):
    if is_matrix:
        matrix_size = int(np.sqrt(population_size))
        popolazione = compute_fitness(np.array([
            [
                [Albero(VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES), 0] # individuo e fitness --> massimizzare
                for _ in range(matrix_size)
            ] for _ in range(matrix_size)
        ]),
            ltn_dict=ltn_dict,
            variabili=variabili,
            is_matrix=is_matrix)
    else:
        # Genera la popolazione come lista di liste con individui inizializzati e fitness iniziale a 0

        popolazione = [
        [Albero(VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES), 0]
        for _ in range(population_size)]
        popolazione = compute_fitness(
        popolazione,
        ltn_dict=ltn_dict,
        variabili=variabili,
        is_matrix=is_matrix
    )
    return popolazione

def compute_fitness_singolo(individuo, ltn_dict, variabili):
    predicati = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
    formula = individuo.to_ltn_formula(ltn_dict, variabili)
    
    fitness = formula.value.item()
    # Penalizza se ci sono duplicati
    if len(predicati) != len(set(predicati)):
        fitness *= 0.6
    if individuo.profondita > 6:
        fitness *= 0.9
    return fitness 


def fitness_proportionate_selection(population, is_matrix, num_to_select=2):
    fitness_values = [individual[-1] for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]

    # Seleziona individui in base alle probabilità
    selected_individuals = random.choices(population, weights=probabilities, k=num_to_select)
    return selected_individuals

    
    

def fitness_proportionate_selection_modern(population, is_matrix, num_to_select=2, x=0.32):
    """
    Modern GP selection with "over-selection":
    - Divide the population into two groups by fitness: top x% and the rest.
    - 80% of selection operations choose from the top x%, 20% from the rest.

    Args:
        population (list): Population where each individual is [object, fitness].
        num_to_select (int): Number of individuals to select.
        x (float): Proportion of the population in the top group (default 32%).

    Returns:
        list: Selected individuals from the population.
    """
    sorted_population = sorted(population, key=lambda ind: ind[-1], reverse=True)
        

    top_group_size = max(1, int(len(sorted_population) * x))
    top_group = sorted_population[:top_group_size]
    bottom_group = sorted_population[top_group_size:]

    selected_individuals = []
    for _ in range(num_to_select):
        if random.random() < 0.8 and random.random() > 0.1:  # 80% probabilità di scegliere dal top group
            selected = random.choice(top_group)
        elif random.random() > 0.8:  # 20% probabilità di scegliere dal bottom group
            selected = random.choice(bottom_group)
        else: 
            selected = random.choice(sorted_population)
        selected_individuals.append(selected)

    return selected_individuals




def evolutionary_run_GP(popolazione, generations, ltn_dict, variabili, operatori, metodo, is_matrix, num_offspring):
    if is_matrix:
        for generation in range(generations):
            #print(f"\n--- Generazione {generation + 1}/{generations} ---")
            for i in range(popolazione.shape[0]):
                for j in range(popolazione.shape[1]):
                    vicini = get_neighbors(popolazione, i, j)
                    vicini.sort(key=lambda x: x[3], reverse=True)
                    parents = metodo(vicini, is_matrix=is_matrix, num_to_select=1)
                    parents = [individual[2] for individual in parents]
                    child1, child2 = crossover(parents[0], popolazione[i,j][0], prob=0.8)
                    child1 = mutate(child1, prob=0.2)
                    child2 = mutate(child2, prob=0.4)
                    fit_child1 = compute_fitness_singolo(child1, ltn_dict, variabili)
                    fit_child2 = compute_fitness_singolo(child2, ltn_dict, variabili)
                    if fit_child1 > fit_child2:
                        popolazione[i,j][0] = child1
                        popolazione[i,j][1] = fit_child1
                    else:
                        popolazione[i,j][0] = child2
                        popolazione[i,j][1] = fit_child2

                    
    else:
        for generation in range(generations):
            parent_list = []
            child_list = []
            #print(f"\n--- Generazione {generation + 1}/{generations} ---")
            for _ in range(num_offspring):
                parents = metodo(popolazione, is_matrix=is_matrix)
                parents = [individual[0] for individual in parents]
                child1, child2 = crossover(parents[0], parents[1], prob=0.8)
                child1 = mutate(child1, prob=0.2)
                child2 = mutate(child2, prob=0.2)
                fit_child1 = compute_fitness_singolo(child1, ltn_dict, variabili)
                fit_child2 = compute_fitness_singolo(child2, ltn_dict, variabili)

                parent_list.append(parents[0])
                child_list.append([child1, fit_child1])
                child_list.append([child2, fit_child2])

            eliminati = 0
            # Rimuovi i genitori dalla popolazione
            for parent in parent_list:
                for i, individuo in enumerate(popolazione):
                    if parent == individuo[0]:  # Confronta le formule logiche
                        del popolazione[i]
                        eliminati +=1
                        break
            child_list.sort(key=lambda x: x[1], reverse=True)
            child_list_new = child_list[:num_offspring]
            random.shuffle(child_list_new)
            for i in range(eliminati):
                popolazione.append(child_list_new[i])

    return popolazione
