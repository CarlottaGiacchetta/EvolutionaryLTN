from kb import * 
from structure import *

def compute_fitness(popolazione, ltn_dict, variabili):
    # Calcola la fitness per ogni individuo
    for i in range(popolazione.shape[0]):
        for j in range(popolazione.shape[1]):
            individuo = popolazione[i, j][0]  # Albero
            #print()
            #print(individuo)
            predicati = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
            formula = individuo.to_ltn_formula(ltn_dict, variabili)
            fitness = formula.value.item()
            #print(fitness)
            #print(predicati)
            #print(set(predicati))  # Ora funziona correttamente
            #print(len(predicati), len(set(predicati)))
            
            # Penalizza se ci sono duplicati
            if len(predicati) != len(set(predicati)):
                fitness *= 0.6
            popolazione[i, j][1] = fitness  # Aggiorna fitness
            #print(fitness)
            #print()
    return popolazione

def compute_fitness_singolo(individuo, ltn_dict, variabili):
    predicati = [nodo for nodo in get_all_nodes(individuo.radice) if nodo.tipo_nodo == "PREDICATO"]
    formula = individuo.to_ltn_formula(ltn_dict, variabili)
    
    fitness = formula.value.item()
    # Penalizza se ci sono duplicati
    if len(predicati) != len(set(predicati)):
        fitness *= 0.6

    return fitness 



def evolutionary_run(popolazione, generations, ltn_dict, variabili, operatori):
    for generation in range(generations):
        print(f"\n--- Generazione {generation + 1}/{generations} ---")

        for i in range(popolazione.shape[0]):
            for j in range(popolazione.shape[1]):
                
                # 1. Trova i vicini
                vicini = get_neighbors(popolazione, i, j)
                # 2. Ordina decrescente per fitness

                vicini.sort(key=lambda x: x[3], reverse=True)
                
                # best neighbor -> parent2
                x_best, y_best, parent2_tree, parent2_fitness = vicini[0]
                
                # parent1 è l'individuo in (i,j)
                parent1 = popolazione[i, j][0]
                
                # 3. Crossover
                child1, child2 = crossover(parent1, parent2_tree, prob=0.9)
                
                # 4. Mutazione
                child1 = mutate(child1, prob=0.2)
                child2 = mutate(child2, prob=0.2)
                
                # 5. Calcolo fitness figli
                fit_child1 = compute_fitness_singolo(child1, ltn_dict, variabili)
                fit_child2 = compute_fitness_singolo(child2, ltn_dict, variabili)
                
                # 6. Crea individuo_1 e individuo_2
                individuo_1 = [child1, fit_child1]
                individuo_2 = [child2, fit_child2]
                
                # 7. Sostituisci:
                #    - parent1 (in (i,j)) diventa individuo_1
                #    - parent2 (in (x_best, y_best)) diventa individuo_2
                popolazione[i, j] = individuo_1
                popolazione[x_best, y_best] = individuo_2

        # Se vuoi, puoi ricalcolare la fitness dell'intera popolazione solo alla fine
        popolazione = compute_fitness(popolazione, ltn_dict, variabili)

    return popolazione
