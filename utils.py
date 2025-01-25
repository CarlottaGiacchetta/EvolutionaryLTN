import torch.nn as nn
import torch
from kb import kb_loss
from tree import Nodo, Albero
from parser import *

#################################################################
# get_all_nodes e replace_node_in_tree
#################################################################

def get_all_nodes(nodo):
    nodes = [nodo]
    for f in nodo.figli:
        nodes.extend(get_all_nodes(f))
    return nodes

def replace_node_in_tree(tree: Nodo, old_node: Nodo, new_subtree: Nodo):
    """
    Se trova old_node in `tree` (per reference), lo sostituisce con new_subtree (deepcopy).
    Restituisce il nodo sostituito (nuovo) oppure None se non trovato.
    """
    if tree is old_node:
        tree.tipo_nodo = new_subtree.tipo_nodo
        tree.valore = new_subtree.valore
        tree.figli = [c.copia() for c in new_subtree.figli]
        return tree

    for i, child in enumerate(tree.figli):
        if child is old_node:
            inserted = new_subtree.copia()
            tree.figli[i] = inserted
            return inserted
        else:
            replaced = replace_node_in_tree(child, old_node, new_subtree)
            if replaced is not None:
                return replaced
    return None

def find_path(root, target):
    if root is target:
        return [root]
    for child in root.figli:
        subpath = find_path(child, target)
        if subpath:
            return [root] + subpath
    return []

def get_scope_vars(root, target):
    path = find_path(root, target)
    if not path:
        return []
    scope_vars = []
    for node in path:
        if node.tipo_nodo == "QUANTIFICATORE":
            scope_vars.append(node.figli[0].valore)
    return scope_vars

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


# Piccolo mlp per un predicato
def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
    return nn.Sequential(
        nn.Linear(in_features, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid()  # output in [0,1]
    )


def get_neighbors(popolazione, i, j):
    neighbors = []
    # elenco delle possibili direzioni
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for di, dj in directions:
        x = i + di
        y = j + dj
        # verifica che x,y siano entro i limiti di popolazione
        if 0 <= x < popolazione.shape[0] and 0 <= y < popolazione.shape[1]:
            # Salvo TUTTO ciò che mi serve: (i_vicino, j_vicino, albero, fitness)
            neighbors.append((x, y, popolazione[x, y][0], popolazione[x, y][1]))
    return neighbors


def measure_kb_sat(kb_rules, kb_facts, variabili, costanti):
    total = 0.0
    count = 0
    for rule in kb_rules:
        for var_name in variabili:
            val = rule(variabili[var_name]).value
            total += val.mean().item()
            count += 1
    #for c_name in costanti:
    #    for fact in kb_facts:
    #        val = fact(costanti[c_name]).value
    #        total += val.mean().item()
    #        count += 1
    return total / max(count, 1)

def make_new_rule(albero, ltn_dict, variabili):
    """
    Ritorna una funzione regola: data una x, ricostruisce la formula dall'albero
    e la valuta, restituendo un LTNObject fresco ogni volta.
    """
    def rule_function(x):
        # Ricostruisce la formula dal tuo albero
        # in questo modo ogni volta che 'rule_function' è chiamata,
        # si rifà il forward pass e crea un nuovo grafo PyTorch.
        ltn_formula = albero.to_ltn_formula(ltn_dict, variabili)
        return ltn_formula  # LTNObject "nuovo"
    return rule_function


def print_kb_status(kb_rules, kb_facts, variabili, costanti):
    print("\n--- Stato della Knowledge Base ---")
    print("\n**Regole:**")
    for i, rule in enumerate(kb_rules, 1):
        for var_name in variabili:
            val = rule(variabili[var_name]).value
            print(f"Regola {i}, Variabile '{var_name}': {val.item():.4f}, {rule}")



def analizza_predicati(nodo: Nodo):
    """Ritorna (num_predicati, dict{pred_name:count})."""
    from collections import defaultdict
    dict_count = defaultdict(int)

    def dfs(n):
        if n.tipo_nodo == "PREDICATO":
            pred_name, _ = parse_predicato(n.valore)
            dict_count[pred_name] += 1
        for c in n.figli:
            dfs(c)

    dfs(nodo)
    num_pred = sum(dict_count.values())
    return num_pred, dict_count

def is_tautology(nodo: Nodo):
    """
    Check basic tautologies, such as:
    - pred(x) OR NOT pred(x)
    - pred(x) => pred(x)
    Returns True if a basic tautology is detected, else False.
    """
    # Caso base: se il nodo è un quantificatore, controlla il corpo
    if nodo.tipo_nodo == "QUANTIFICATORE":
        # Controlla se il corpo è una tautologia
        return is_tautology(nodo.figli[1])
    
    # Se il nodo è un operatore, verifica i pattern di tautologia
    if nodo.tipo_nodo == "OPERATORE":
        op = nodo.valore.upper()
        
        # Tautologia: Pred(x) OR NOT Pred(x)
        if op == "OR":
            if len(nodo.figli) != 2:
                return False
            left, right = nodo.figli
            if (left.tipo_nodo == "PREDICATO" and
                right.tipo_nodo == "OPERATORE" and right.valore.upper() == "NOT" and
                len(right.figli) == 1 and right.figli[0].tipo_nodo == "PREDICATO"):
                
                pred_left = left.valore.split('(')[0].strip()
                pred_right = right.figli[0].valore.split('(')[0].strip()
                
                if pred_left == pred_right:
                    return True
            
            # Controllo inverso: NOT Pred(x) OR Pred(x)
            if (right.tipo_nodo == "PREDICATO" and
                left.tipo_nodo == "OPERATORE" and left.valore.upper() == "NOT" and
                len(left.figli) == 1 and left.figli[0].tipo_nodo == "PREDICATO"):
                
                pred_right = right.valore.split('(')[0].strip()
                pred_left = left.figli[0].valore.split('(')[0].strip()
                
                if pred_left == pred_right:
                    return True
        
        # Tautologia: Pred(x) => Pred(x)
        elif op == "IMPLIES":
            if len(nodo.figli) != 2:
                return False
            antecedent, consequent = nodo.figli
            if (antecedent.tipo_nodo == "PREDICATO" and
                consequent.tipo_nodo == "PREDICATO"):
                
                pred_antecedent = antecedent.valore.split('(')[0].strip()
                pred_consequent = consequent.valore.split('(')[0].strip()
                
                if pred_antecedent == pred_consequent:
                    return True
    
    # Altri casi non riconosciuti come tautologie
    return False
