from knowledge_base import *
from utils import *
from copy import deepcopy
import numpy as np
import torch

def compute_fitness_singolo(individuo, ltn_dict, variabili, predicati, costanti,
                            kb_rules, kb_facts, baseline_sat, salvo_formula,
                            lambda_complexity=0.01, lambda_novelty=1.0):
    """
    Calcola la fitness = (sat(KB+phi) - sat(KB)) + novelty - penalty_complexity
    """
    tmp = individuo
    new_rule = make_new_rule(individuo, ltn_dict, variabili)
    extended_rules = kb_rules + [new_rule]

    # 3) satisfaction
    extended_sat = measure_kb_sat(extended_rules, kb_facts, variabili, costanti)

    # 4) delta
    delta = extended_sat - baseline_sat

    # 5) complessità
    complexity = len(get_all_nodes(individuo.radice))
    penalty = lambda_complexity * complexity

    # 6) novelty 
    novelty = 1 if tmp not in set(salvo_formula) else 0

    # 7) fitness
    fitness = delta + (lambda_novelty * novelty) #- penalty
    return fitness


def partial_train(predicati, kb_rules, kb_facts, variabili, costanti, steps=50, lr=0.001):

    parameters = []
    for pred in predicati.values():
        parameters += list(pred.model.parameters())

    opt = torch.optim.Adam(parameters, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        # Calcolo la loss come (1 - kb_loss), facendo .mean() per sicurezza
        base_loss = kb_loss(kb_rules, kb_facts, variabili, costanti)  # kb_loss -> scalar
        loss = (1 - base_loss).mean()  # .mean() riduce eventuali dimensioni residue
        loss.backward()
        opt.step()


def evolutionary_run(popolazione, generations, ltn_dict, variabili, predicati, costanti,
                     ottimizzatore, kb_rules, kb_facts, baseline_sat):
    patience = 1000
    tolerance = 1e-4

    best_fitness = -float('inf')
    patience_counter = 0

    baseline_sat_gugu = deepcopy(baseline_sat)
    salvo_formula = []
    salvo_albero = []
    for generation in range(generations):
        print(f"\n--- Generazione {generation + 1}/{generations} ---")

        max_fitness_generation = -float('inf')

        # Evolvi la popolazione
        for i in range(popolazione.shape[0]):
            for j in range(popolazione.shape[1]):
                parent1, fitness_parent1 = popolazione[i, j]

                vicini = get_neighbors(popolazione, i, j)
                fitness_values = np.array([v[3] for v in vicini])
                fitness_sum = fitness_values.sum()
                if fitness_sum > 0:
                    probabilities = fitness_values / fitness_sum
                else:
                    probabilities = np.ones_like(fitness_values) / len(fitness_values)

                sel_idx = np.random.choice(len(vicini), p=probabilities)
                _, _, parent2_tree, parent2_fitness = vicini[sel_idx]

                # CROSSOVER
                child1, child2 = crossover(parent1, parent2_tree, prob=0.9)
                # MUTATION
                child1 = mutate(child1, prob=0.9)
                child2 = mutate(child2, prob=0.9)

                # Calcola fitness
                fit_child1 = compute_fitness_singolo(
                    child1, ltn_dict, variabili, predicati, costanti,
                    kb_rules, kb_facts, baseline_sat, salvo_formula
                )
                fit_child2 = compute_fitness_singolo(
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

                # update liveness
                parent1, fitness_parent1 = popolazione[i, j]
                parent1.update_liveness(fitness_parent1)

        # Fine generazione
        print(f"Generazione {generation+1}, miglior fitness generazione = {max_fitness_generation:.4f}")

        # Early stopping
        if max_fitness_generation > best_fitness + tolerance:
            best_fitness = max_fitness_generation
            patience_counter = 0
            #print("Miglioramento significativo, reset patience.")
        else:
            patience_counter += 1
            #print(f"Nessun miglioramento significativo. Patience = {patience_counter}/{patience}")

        if patience_counter >= patience:
            #print("Early stopping per mancanza di miglioramenti.")
            break

        # *** OGNI 10 GENERAZIONI: aggiungo la formula migliore alla KB se "abbastanza buona" ***
        if (generation + 1) % 10 == 0:
            # Trova miglior individuo in questa generazione
            # (abbiamo già i fitness in popolazione)
            flat_pop = [popolazione[i, j] for i in range(popolazione.shape[0]) for j in range(popolazione.shape[1])]
            flat_pop_sorted = sorted(flat_pop, key=lambda x: x[1], reverse=True)
            best_ind, best_ind_fitness = flat_pop_sorted[0]

            # Se la fitness > 0, consideriamola "abbastanza sat" (soglia personalizzabile)
            if best_ind_fitness > 0.98 and best_ind not in set(salvo_formula):
                #print(f"\nAggiungo la formula migliore della gen {generation+1} alla KB! Fitness={best_ind_fitness:.4f}")
                # 1) Converto in LTN
                salvo_formula.append(best_ind)
                new_rule = make_new_rule(best_ind, ltn_dict, variabili)
                
                kb_rules.append(new_rule)

                # 3) (Opzionale) Faccio un mini-training per incorporarla
                partial_train(predicati, kb_rules, kb_facts, variabili, costanti, steps=50, lr=0.001)

                # 4) Ricalcolo la baseline_sat
                new_baseline = measure_kb_sat(kb_rules, kb_facts, variabili, costanti)
                #print(f"Nuova baseline SAT dopo add formula e mini-train: {new_baseline:.4f}")
                baseline_sat = new_baseline

                print_kb_status(kb_rules, kb_facts, variabili, costanti)

    print("stato SAT iniziale: ", baseline_sat_gugu)
    print("stato SAT finale: ", measure_kb_sat(kb_rules, kb_facts, variabili, costanti))
    print("Le nuove formule sono: \n")
    for i in salvo_formula:
        print(i)
    return popolazione
