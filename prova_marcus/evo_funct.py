from knowledge_base import *
from copy import deepcopy
import numpy as np
import random
from utils import make_new_rule, measure_kb_sat, get_all_nodes, get_scope_vars, replace_node_in_tree, get_neighbors, print_kb_status, partial_train,is_tautology, analizza_predicati
from tree import Albero, Nodo

def compute_fitness_singolo(individuo,
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

def crossover(a1: Albero, a2: Albero):
    """
    Esegue crossover su nodi di tipo OPERATORE (binario o NOT) o PREDICATO,
    evitando QUANTIFICATORE e VARIABILE.
    """

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

#################################################################
# MUTATE
#################################################################


def mutate(albero: Albero):
    """
    Esempio di mutazione con distinzioni sensate tra operatori unari (NOT) e binari (AND, OR, IMPLIES).
    - Se il target è un OPERATORE binario, possiamo cambiare l'operatore (es. AND -> OR).
    - Se il target è un PREDICATO, possiamo:
      a) Cambiare predicato (nome e arità),
      b) Avvolgerlo in un NOT,
      c) Espanderlo in un operatore binario (es. pred -> (pred AND new_pred)).
    - Se il target è un OPERATORE (binario) con figli, potremmo (facoltativamente) avvolgerlo in un quantificatore, etc.
    """

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

def evolutionary_run(popolazione, generations, ltn_dict, variabili, predicati, costanti, kb_rules, kb_facts, baseline_sat):
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
                fitness_values = np.array([v[3] for v in vicini])
                fitness_sum = fitness_values.sum()
                if fitness_sum > 0:
                    probabilities = fitness_values / fitness_sum
                else:
                    probabilities = np.ones_like(fitness_values) / len(fitness_values)

                sel_idx = np.random.choice(len(vicini), p=probabilities)
                _, _, parent2_tree, parent2_fitness = vicini[sel_idx]

                # CROSSOVER
                child1, child2 = crossover(parent1, parent2_tree)
                # MUTATION
                child1 = mutate(child1)
                child2 = mutate(child2)

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

