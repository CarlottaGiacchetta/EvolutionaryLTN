from kb import * 
from structure import *
from copy import deepcopy
import numpy as np


def compute_fitness_singolo(individuo, ltn_dict, variabili, predicati, costanti, kb_rules, kb_facts):
    
    steps = 50
    parameters = []

    for pred in predicati.values():
        parameters += list(pred.model.parameters())
    
    parametri_da_usare = deepcopy(parameters)

    nuova_formula = lambda x: individuo.to_ltn_formula(ltn_dict, variabili)
    
    tmp_kb_rules = kb_rules + [nuova_formula]
    
    optimizer_retrain = torch.optim.Adam(parametri_da_usare, lr=0.01)
    
    for _ in range(steps):
        optimizer_retrain.zero_grad()
        loss = 1 - kb_loss(tmp_kb_rules, kb_facts, variabili, costanti)
        loss.backward()
        optimizer_retrain.step()

    numero_di_formule_unsat = 0 # da massimmizzare
    avg_sat = 0 #[0,1]

    threshold = 0.7
    for chiave in variabili:
        for i, formula in enumerate(tmp_kb_rules, start=1):
            satisfaction = formula(variabili[chiave]).value
            avg_sat += satisfaction
            if satisfaction < threshold:
                numero_di_formule_unsat +=1
    
    for costante in costanti:
        for i, formula in enumerate(kb_facts, start=1):
            satisfaction = formula(costanti[costante]).value
            avg_sat += satisfaction
            if satisfaction < threshold:
                numero_di_formule_unsat +=1

    indice_di_sat = 1 - avg_sat/((len(tmp_kb_rules) * len(variabili)) + (len(kb_facts) * len(costanti))) #(da massimizzare)
    lunghezza_formula = len([nodo.tipo_nodo for nodo in get_all_nodes(individuo.radice)]) # da minimizzare
    
    
    fitness = torch.detach(np.exp(numero_di_formule_unsat) + indice_di_sat - lunghezza_formula).item()
    #fitness = torch.detach(indice_di_sat - lunghezza_formula).item()
    #fitness = np.exp(numero_di_formule_unsat)

    return fitness, numero_di_formule_unsat


import numpy as np

def evolutionary_run(popolazione, generations, ltn_dict, variabili, predicati, costanti, ottimizzatore, kb_rules, kb_facts):
    patience = 10
    tolerance = 1e-4
    # Inizializza il valore migliore di fitness e il contatore di pazienza
    best_fitness = -float('inf')
    patience_counter = 0

    for generation in range(generations):
        print(f"\n--- Generazione {generation + 1}/{generations} ---")

        numero_unsat = 0
        max_fitness_generation = -float('inf')  # Per tracciare il miglioramento in questa generazione

        for i in range(popolazione.shape[0]):
            for j in range(popolazione.shape[1]):

                # Ottieni il parent1
                parent1, fitness_parent1 = popolazione[i, j]

                # Trova i vicini
                vicini = get_neighbors(popolazione, i, j)

                # Estrai fitness dei vicini e normalizzala per la selezione proporzionale
                fitness_values = np.array([neighbor[3] for neighbor in vicini])
                fitness_sum = fitness_values.sum()
                if fitness_sum > 0:
                    probabilities = fitness_values / fitness_sum
                else:
                    probabilities = np.ones_like(fitness_values) / len(fitness_values)  # Uniforme se fitness negativa o zero

                # Seleziona il parent2 proporzionalmente alla fitness
                selected_index = np.random.choice(len(vicini), p=probabilities)
                _, _, parent2_tree, parent2_fitness = vicini[selected_index]

                # Esegui crossover
                child1, child2 = crossover(parent1, parent2_tree, prob=0.9)

                # Esegui mutazione
                child1 = mutate(child1, prob=0.9)
                child2 = mutate(child2, prob=0.9)

                # Calcola fitness figli
                fit_child1, num_unsat1 = compute_fitness_singolo(
                    child1, ltn_dict, variabili, predicati, costanti, kb_rules, kb_facts
                )
                fit_child2, num_unsat2 = compute_fitness_singolo(
                    child2, ltn_dict, variabili, predicati, costanti, kb_rules, kb_facts
                )

                # Trova il miglior figlio
                if fit_child1 >= fit_child2:
                    best_child, best_child_fitness = child1, fit_child1
                else:
                    best_child, best_child_fitness = child2, fit_child2

                # Sostituisci il parent in (i, j) solo se il miglior figlio è migliore
                if best_child_fitness > fitness_parent1:
                    popolazione[i, j] = [best_child, best_child_fitness]

                # Aggiorna il numero di formule UNSAT
                numero_unsat = max(numero_unsat, num_unsat1, num_unsat2)

                # Aggiorna la fitness massima per questa generazione
                max_fitness_generation = max(max_fitness_generation, fit_child1, fit_child2)

        print(f"\nNumero di Formule diventate UNSAT: {numero_unsat}")
        print(f"Miglior Fitness Generazione: {max_fitness_generation:.4f}")

        # Controllo early stopping
        if max_fitness_generation > best_fitness + tolerance:
            best_fitness = max_fitness_generation
            patience_counter = 0  # Reset patience
            print("Miglioramento significativo trovato. Reset patience counter.")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento significativo. Patience counter: {patience_counter}/{patience}")

        # Interruzione anticipata se la pazienza è esaurita
        if patience_counter >= patience:
            print("Early stopping attivato per mancanza di miglioramenti significativi.")
            break

    return popolazione

