from kb import *
from copy import deepcopy
import numpy as np
import random
from utils import *
from tree import Albero, Nodo


#################################################################
# COMPUTE INITIAL POPULATION
#################################################################

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




#################################################################
# FITNESS
#################################################################

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

# Compute the fitness for a string-based logical formula
def compute_fitness_string(formula_str, ltn_dict, variabili):
    """
    Compute the fitness of a logical formula string.
    """
    try:
        #print('\n\n')
        #print('formula str')
        #print(formula_str)
        formula = Nodo("PREDICATO", formula_str)  # Dummy placeholder
        #print('formula str')
        #print(formula)
        fitness = compute_fitness_singolo(formula, ltn_dict, variabili)
        return fitness
    except:
        #print('except')
        return 0.1

def compute_fitness_retrain(individuo,
                            ltn_dict,
                            variabili,
                            predicati,
                            costanti,
                            kb_rules,
                            kb_facts,
                            baseline_sat,
                            salvo_formula,
                            lambda_complexity=0.01,
                            lambda_novelty=1.0):
    """
    Calcola la fitness di un individuo (formula):
      - Delta SAT (esteso vs baseline)
      - + novelty se formula non ancora vista
      - penalità esponenziale di complessità
      - penalità se la formula ha un solo predicato
      - penalità se (euristicamente) è tautologia
      - penalità se troppi duplicati di predicati
    """

    # Costruisce la regola
    new_rule = make_new_rule(individuo, ltn_dict, variabili)
    extended_rules = kb_rules + [new_rule]

    # SAT
    extended_sat = measure_kb_sat(extended_rules, kb_facts, variabili, costanti)
    delta = extended_sat - baseline_sat

    # novelty
    novelty = 1.0 if individuo not in set(salvo_formula) else 0.0

    # penalità di complessità (esponenziale)
    nodi = get_all_nodes(individuo.radice)
    num_nodi = len(nodi)
    penalty_complex = lambda_complexity * (2 ** (0.1 * num_nodi))

    # analizza predicati
    num_predicati_tot, dict_pred_count = analizza_predicati(individuo.radice)

    # penalità single-pred
    penalty_single_pred = 0.0
    if num_predicati_tot <= 1:
        penalty_single_pred = 2.0

    # penalità tautologia
    penalty_tauto = 0.0
    if is_tautology(individuo.radice):
        penalty_tauto = 3.0

    # penalità ripetizioni
    penalty_repetition = 0.0
    for pred_name, cnt in dict_pred_count.items():
        if cnt > 1:
            penalty_repetition += (cnt - 1) * 0.5

    # fitness
    fitness = delta + (lambda_novelty * novelty)
    fitness -= penalty_complex
    fitness -= penalty_single_pred
    fitness -= penalty_tauto
    fitness -= penalty_repetition

    return fitness


#################################################################
# CROSSOVER
#################################################################

def crossover(a1: Albero, a2: Albero, prob=0.8):
    """
    Esegue crossover su nodi di tipo OPERATORE (binario o NOT) o PREDICATO,
    evitando QUANTIFICATORE e VARIABILE.
    """
    if random.random() > prob:
        return a1.copia(), a2.copia()
    
    c1 = a1.copia()
    c2 = a2.copia()

    n1_all = get_all_nodes(c1.radice)
    n2_all = get_all_nodes(c2.radice)

    def is_swappable(n):
        return n.tipo_nodo in ["OPERATORE", "PREDICATO"]

    n1 = [nd for nd in n1_all if is_swappable(nd)]
    n2 = [nd for nd in n2_all if is_swappable(nd)]

    if not n1 or not n2:
        return c1, c2

    old1 = random.choice(n1)
    old2 = random.choice(n2)

    sub1 = old1.copia()
    sub2 = old2.copia()

    inserted1 = replace_node_in_tree(c1.radice, old1, sub2)
    inserted2 = replace_node_in_tree(c2.radice, old2, sub1)

    c1.profondita = c1.calcola_profondita(c1.radice)
    c2.profondita = c2.calcola_profondita(c2.radice)

    return c1, c2


# Crossover operation for logical formula strings
def crossover_string(str1, str2):
    """
    Perform a crossover operation on two logical formula strings.
    The crossover will randomly combine parts of the two strings.
    """
    parts1 = str1.split()
    parts2 = str2.split()
    
    crossover_point1 = random.randint(0, len(parts1) - 1)
    crossover_point2 = random.randint(0, len(parts2) - 1)
    
    # Swap parts after crossover points
    new_str1 = " ".join(parts1[:crossover_point1] + parts2[crossover_point2:])
    new_str2 = " ".join(parts2[:crossover_point2] + parts1[crossover_point1:])
    
    return new_str1, new_str2

#################################################################
# MUTATE
#################################################################


def mutate(albero: Albero, prob=0.3):
    """
    Esempio di mutazione con distinzioni sensate tra operatori unari (NOT) e binari (AND, OR, IMPLIES).
    - Se il target è un OPERATORE binario, possiamo cambiare l'operatore (es. AND -> OR).
    - Se il target è un PREDICATO, possiamo:
      a) Cambiare predicato (nome e arità),
      b) Avvolgerlo in un NOT,
      c) Espanderlo in un operatore binario (es. pred -> (pred AND new_pred)).
    - Se il target è un OPERATORE (binario) con figli, potremmo (facoltativamente) avvolgerlo in un quantificatore, etc.
    """
    if random.random() > prob:
        return albero.copia()
    
    new_tree = albero.copia()
    nodes_all = get_all_nodes(new_tree.radice)

    # Filtra i nodi mutabili: PREDICATO e OPERATORE
    def is_mutable(n):
        return n.tipo_nodo in ["OPERATORE", "PREDICATO"]

    candidates = [nd for nd in nodes_all if is_mutable(nd)]
    if not candidates:
        return new_tree 

    # Scegli a caso un nodo
    target = random.choice(candidates)
    r = random.random()

    # liste di operatori unari e binari
    UNARY_OPS = ["NOT"]
    BINARY_OPS = [op for op in albero.OPERATORS if op not in UNARY_OPS]

    # 1) Se il nodo è OPERATORE e r < 0.25, cambiamo l'operatore
    #    (solo se è un operatore binario)
    if target.tipo_nodo == "OPERATORE" and target.valore in BINARY_OPS and r < 0.25:
        old_op = target.valore
        # scegli un nuovo operatore binario diverso
        possibile_ops = [op for op in BINARY_OPS if op != old_op]
        if possibile_ops:  # per sicurezza
            new_op = random.choice(possibile_ops)
            target.valore = new_op

    # 2) Se il nodo è PREDICATO e 0.25 <= r < 0.5, cambiamo il nome / arità del predicato
    elif target.tipo_nodo == "PREDICATO" and 0.25 <= r < 0.5:
        scopevars = get_scope_vars(new_tree.radice, target)
        if not scopevars:
            var_list = ["x"]  # fallback se non trovi quantificatori
        else:
            # potresti sceglierne 1 o 2
            var_list = [random.choice(scopevars)]
            if random.random() < 0.5 and len(scopevars) > 1:
                var_list.append(random.choice(scopevars))
        new_pred = random.choice(albero.PREDICATES)
        var_str = ",".join(var_list)
        target.valore = f"{new_pred}({var_str})"

    # 3) Se il nodo è un PREDICATO e 0.5 <= r < 0.75, avvolgiamo in NOT (unario) --> per forza perche altrimenti sarebbe caduto nel punto 1
    elif target.tipo_nodo == "PREDICATO" and 0.5 <= r < 0.75:
        not_node = Nodo("OPERATORE", "NOT", [ target.copia() ])
        replace_node_in_tree(new_tree.radice, target, not_node)
    # 4) Se il nodo è un PREDICATO e 0.75 <= r < 0.9, espandiamo in un operatore binario
    elif target.tipo_nodo == "PREDICATO" and 0.75 <= r < 1:
        old_pred = target.copia()
        # creiamo un nuovo predicato casuale
        new_pred_name = random.choice(albero.PREDICATES)
        new_pred_val = f"{new_pred_name}(x)"
        new_pred_nodo = Nodo("PREDICATO", new_pred_val)

        # scgli un operatore binario, es. AND
        random_op = random.choice(BINARY_OPS)
        # costruiamo l'operatore con i due predicati
        expanded_node = Nodo("OPERATORE", random_op, [old_pred, new_pred_nodo])
        replace_node_in_tree(new_tree.radice, target, expanded_node)
    else:
        pass

    # Ricalcola la profondità
    new_tree.profondita = new_tree.calcola_profondita(new_tree.radice)

    return new_tree

# Mutation operation for logical formula strings
def mutate_string(formula_str):
    """
    Perform a mutation on a logical formula string.
    This could involve changing a predicate, operator, or variable.
    """
    words = formula_str.split()
    mutation_point = random.randint(0, len(words) - 1)
    mutated_word = random.choice(["Cat", "Dog", "HasWhiskers", "AND", "OR", "NOT", "IMPLIES"])
    
    words[mutation_point] = mutated_word
    return " ".join(words)



#################################################################
# EVOLUTIONARY RUN
#################################################################



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


# Main Genetic Algorithm Loop with string-based individuals
# Main Genetic Algorithm Loop with string-based individuals
def evolutionary_run_GA(popolazione, generations, ltn_dict, variabili, operatori, metodo, is_matrix,population_size, num_offspring, first=True,):
    """
    Perform the genetic algorithm using formula strings instead of tree structures.
    """
    print(first)
    
    if is_matrix:
        if first:
            # Convert individuals to string representation for GA operations
            for i in range(popolazione.shape[0]):
                for j in range(popolazione.shape[1]):
                    individuo = popolazione[i][j][0]
                    individuo_str = Albero.albero_to_string(individuo)
                    popolazione[i][j][0] = individuo_str  # Replace with string representation
                    first = False

        for generation in range(generations):
            for i in range(popolazione.shape[0]):
                for j in range(popolazione.shape[1]):
                    vicini = get_neighbors(popolazione, i, j)
                    vicini.sort(key=lambda x: x[3], reverse=True)
                    parents = metodo(vicini, is_matrix=is_matrix, num_to_select=1)
                    parents = [individual[2] for individual in parents]
                    child_list = []

                    while len(child_list) < num_offspring:
                        child1_str, child2_str = crossover_string(parents[0], popolazione[i, j][0])
                        child1_str = mutate_string(child1_str)
                        child2_str = mutate_string(child2_str)
                        fit_child1 = compute_fitness_string(child1_str, ltn_dict, variabili)
                        fit_child2 = compute_fitness_string(child2_str, ltn_dict, variabili)

                        child_list.append((child1_str, fit_child1))
                        if len(child_list) < num_offspring:
                            child_list.append((child2_str, fit_child2))

                    best_child = max(child_list, key=lambda x: x[1])
                    popolazione[i, j][0] = best_child[0]
                    popolazione[i, j][1] = best_child[1]

    else:
        if first:
            # Convert individuals to string representation for GA operations
            for i in range(len(popolazione)):
                individuo = popolazione[i][0]
                individuo_str = Albero.albero_to_string(individuo)
                popolazione[i][0] = individuo_str  # Replace with string representation
                first = False

        for generation in range(generations):
            print(f"--- Generazione {generation + 1}/{generations} ---")

            child_list = []  # Lista per raccogliere tutti i figli generati in questa generazione

            while len(child_list) < num_offspring:
                # Selection: Select parents based on fitness
                selected_parents = metodo(popolazione, is_matrix, num_to_select=2)

                # Crossover: Produce offspring by combining parents
                child1_str, child2_str = crossover_string(selected_parents[0][0], selected_parents[1][0])

                # Mutation: Apply mutation to offspring
                child1_str = mutate_string(child1_str)
                child2_str = mutate_string(child2_str)

                # Evaluate fitness of the offspring
                fit_child1 = compute_fitness_string(child1_str, ltn_dict, variabili)
                fit_child2 = compute_fitness_string(child2_str, ltn_dict, variabili)

                # Add offspring to the child list
                child_list.append((child1_str, fit_child1))
                if len(child_list) < num_offspring:
                    child_list.append((child2_str, fit_child2))

            # Update population (elitism strategy, keeping the best individuals)
            population_with_new_offspring = sorted(popolazione + child_list, key=lambda x: x[1], reverse=True)
            popolazione = population_with_new_offspring[:population_size]  # Keep only the best individuals

            # Print current best individual in the population
            best_individual = popolazione[0]
            print(f"Best individual in generation {generation + 1}: {best_individual[0]} with fitness: {best_individual[1]}")

    return popolazione



def evolutionary_run(popolazione, generations, ltn_dict, variabili, predicati, costanti, kb_rules, kb_facts, baseline_sat, is_matrix, metodo):
    patience = 30
    tolerance = 1e-4

    best_fitness = -float('inf')
    patience_counter = 0

    baseline_sat_gugu = deepcopy(baseline_sat)
    salvo_formula = set()  # Utilizza un set per una ricerca più veloce

    for generation in range(generations):
        print(f"\n--- Generazione {generation + 1}/{generations} ---")

        max_fitness_generation = -float('inf')

        # Evolvi la popolazione
        for i in range(popolazione.shape[0]):
            for j in range(popolazione.shape[1]):
                parent1, fitness_parent1 = popolazione[i, j]
                #print(i,j,parent1, fitness_parent1)

                vicini = get_neighbors(popolazione, i, j)
                vicini.sort(key=lambda x: x[3], reverse=True)
                parents = metodo(vicini, is_matrix=is_matrix, num_to_select=1)
                parents = [individual[2] for individual in parents]
                parent2_tree = parents[0]
                # CROSSOVER
                child1, child2 = crossover(parent1, parent2_tree)
                # MUTATION
                child1 = mutate(child1)
                child2 = mutate(child2)

                # Calcola fitness
                fit_child1 = compute_fitness_retrain(
                    child1, ltn_dict, variabili, predicati, costanti,
                    kb_rules, kb_facts, baseline_sat, salvo_formula
                )
                fit_child2 = compute_fitness_retrain(
                    child2, ltn_dict, variabili, predicati, costanti,
                    kb_rules, kb_facts, baseline_sat, salvo_formula
                )

                if fit_child1 >= fit_child2:
                    best_child, best_child_fitness = child1, fit_child1
                else:
                    best_child, best_child_fitness = child2, fit_child2

                if best_child_fitness > fitness_parent1:
                    popolazione[i, j] = [best_child, best_child_fitness]

                max_fitness_generation = max(max_fitness_generation, fit_child1, fit_child2)

                # Aggiorna la liveness
                parent1, fitness_parent1 = popolazione[i, j]
                parent1.update_liveness(fitness_parent1)

        # Fine generazione
        print(f"Generazione {generation+1}, miglior fitness generazione = {max_fitness_generation:.4f}")

        # Early stopping
        if max_fitness_generation > best_fitness + tolerance:
            best_fitness = max_fitness_generation
            patience_counter = 0
            print("Miglioramento significativo, reset patience.")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento significativo. Patience = {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping per mancanza di miglioramenti.")
            break

        # *** OGNI 10 GENERAZIONI: aggiungo tutte le formule con fitness > 0.98 e uniche ***
        if (generation + 1) % 10 == 0:
            # Trova tutti gli individui con fitness > 0.98
            flat_pop = [popolazione[i, j] for i in range(popolazione.shape[0]) for j in range(popolazione.shape[1])]
            qualifying_individuals = [(ind, fit) for ind, fit in flat_pop if fit > 0.99]

            # Filtra quelli già aggiunti
            unique_individuals = []
            for ind, fit in qualifying_individuals:
                ind_str = str(ind)
                if ind_str not in salvo_formula:
                    unique_individuals.append((ind, fit))
                    salvo_formula.add(ind_str)  # Aggiungi al set

            if unique_individuals:
                print(f"\nAggiungo {len(unique_individuals)} formula/e alla KB!")
                for idx, (best_ind, best_ind_fitness) in enumerate(unique_individuals, 1):
                    print(f"Aggiunta formula {idx} con fitness={best_ind_fitness:.4f}")
                    # 1) Crea una nuova regola
                    new_rule = make_new_rule(best_ind, ltn_dict, variabili)

                    # 2) Aggiungi la regola alla KB
                    kb_rules.append(new_rule)

                # 3) (Opzionale) Fai un mini-training per incorporarle
                partial_train(predicati, kb_rules, kb_facts, variabili, costanti, steps=50, lr=0.001)

                # 4) Ricalcola la baseline_sat
                new_baseline = measure_kb_sat(kb_rules, kb_facts, variabili, costanti)
                print(f"Nuova baseline SAT dopo add formula e mini-train: {new_baseline:.4f}")
                baseline_sat = new_baseline

                # 5) Stampa lo stato aggiornato della KB
                print_kb_status(kb_rules, kb_facts, variabili, costanti)

    # Dopo tutte le generazioni
    print("Stato SAT iniziale:", baseline_sat_gugu)
    print("Stato SAT finale:", measure_kb_sat(kb_rules, kb_facts, variabili, costanti))
    print("Le nuove formule aggiunte sono:\n")
    for formula_str in salvo_formula:
        print(formula_str)
    
    return popolazione




#################################################################
# SELECTION METHODS
#################################################################

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
