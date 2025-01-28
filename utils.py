import torch.nn as nn
import torch
from kb import kb_loss
from tree import Nodo, Albero
from parser import *

#################################################################
# get_all_nodes e replace_node_in_tree
#################################################################

def get_all_nodes(nodo):
    '''
    Recursively retrieves all nodes in a tree starting from the given root node.

    Parameters:
    - nodo (Nodo): The root node of the tree.

    Returns:
    - list: A list containing all nodes in the tree (including the root node).
    '''
    nodes = [nodo]
    for f in nodo.figli:
        nodes.extend(get_all_nodes(f))
    return nodes

def replace_node_in_tree(tree: Nodo, old_node: Nodo, new_subtree: Nodo):
    '''
    Replaces a specific node in the tree with a new subtree.

    This function searches for the `old_node` in the given tree. If found, it replaces the node
    with `new_subtree` (a deep copy of it) and returns the replaced node. If the node is not
    found, it returns `None`.

    Parameters:
    - tree (Nodo): The root of the tree where the replacement will take place.
    - old_node (Nodo): The node to be replaced (must match by reference, not by value).
    - new_subtree (Nodo): The new subtree that will replace the old node.

    Returns:
    - Nodo: The newly inserted node if the replacement was successful, or `None` if the old node
      was not found.
    '''
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
    '''
    Finds the path from the root node to a target node in a tree.

    Parameters:
    - root (Nodo): The root of the tree.
    - target (Nodo): The target node to find.

    Returns:
    - list: A list of nodes representing the path from the root to the target.
      If the target is not found, returns an empty list.
    '''
    if root is target:
        return [root]
    for child in root.figli:
        subpath = find_path(child, target)
        if subpath:
            return [root] + subpath
    return []

def get_scope_vars(root, target):
    '''
    Retrieves the list of variables in the scope of a target node, 
    determined by quantifiers in the path from the root to the target.

    Parameters:
    - root (Nodo): The root of the tree.
    - target (Nodo): The target node for which the scope variables are retrieved.

    Returns:
    - list: A list of variable names (strings) introduced by quantifiers along the path.
      Returns an empty list if the target is not found.
    '''
    path = find_path(root, target)
    if not path:
        return []
    scope_vars = []
    for node in path:
        if node.tipo_nodo == "QUANTIFICATORE":
            scope_vars.append(node.figli[0].valore)
    return scope_vars

def partial_train(predicati, kb_rules, kb_facts, variabili, costanti, steps=50, lr=0.001):
    '''
    Performs a partial training of the predicates in the knowledge base (KB) 
    to better align them with the rules and facts.

    Parameters:
    - predicati (dict): Dictionary of predicates, where each predicate has trainable parameters.
    - kb_rules (list): List of rules in the KB.
    - kb_facts (list): List of facts in the KB.
    - variabili (dict): Dictionary of variables used in the KB.
    - costanti (dict): Dictionary of constants used in the KB.
    - steps (int): Number of optimization steps (default: 50).
    - lr (float): Learning rate for the optimizer (default: 0.001).

    Returns:
    - None
    '''

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
    '''
    Creates a simple neural network for modeling unary predicates.

    Parameters:
    - in_features (int): Number of input features (default: 2).
    - hidden1 (int): Number of neurons in the first hidden layer (default: 8).
    - hidden2 (int): Number of neurons in the second hidden layer (default: 4).

    Returns:
    - nn.Sequential: A PyTorch sequential model with ReLU activations and Sigmoid output.
    '''
    return nn.Sequential(
        nn.Linear(in_features, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid()  # output in [0,1]
    )


def get_neighbors(popolazione, i, j):
    '''
    Retrieves all neighbors of a cell (i, j) in a 2D population matrix.

    Parameters:
    - popolazione (numpy.ndarray): The population matrix.
    - i (int): Row index of the target cell.
    - j (int): Column index of the target cell.

    Returns:
    - list: A list of tuples (neighbor_i, neighbor_j, tree, fitness) for each neighbor.
    '''
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
    '''
    Measures the satisfaction of a knowledge base (KB).

    Parameters:
    - kb_rules (list): List of rules in the KB.
    - kb_facts (list): List of facts in the KB (currently unused).
    - variabili (dict): Dictionary of variables used in the KB.
    - costanti (dict): Dictionary of constants (currently unused).

    Returns:
    - float: The average satisfaction value across all rules and variables.
    '''
    total = 0.0
    count = 0
    for rule in kb_rules:
        for var_name in variabili:
            val = rule(variabili[var_name]).value
            total += val.mean().item()
            count += 1
    
    return total / max(count, 1)

def make_new_rule(albero, ltn_dict, variabili):
    '''
    Creates a new rule function from a tree.

    Parameters:
    - albero (Albero): The logical tree representing the rule.
    - ltn_dict (dict): Dictionary of LTN-compatible predicates, operators, and quantifiers.
    - variabili (dict): Dictionary of variables.

    Returns:
    - function: A callable function representing the rule.
    '''
    def rule_function(x):
        ltn_formula = albero.to_ltn_formula(ltn_dict, variabili)
        return ltn_formula  # LTNObject "nuovo"
    return rule_function


def print_kb_status(kb_rules, kb_facts, variabili, costanti):
    '''
    Prints the current status of the knowledge base, including satisfaction levels
    for rules and facts.

    Parameters:
    - kb_rules (list): List of rules in the KB.
    - kb_facts (list): List of facts in the KB.
    - variabili (dict): Dictionary of variables.
    - costanti (dict): Dictionary of constants.
    '''
    print("\n--- Stato della Knowledge Base ---")
    print("\n**Regole:**")
    for i, rule in enumerate(kb_rules, 1):
        for var_name in variabili:
            val = rule(variabili[var_name]).value
            print(f"Regola {i}, Variabile '{var_name}': {val.item():.4f}, {rule}")



def analizza_predicati(nodo: Nodo):
    '''
    Analyzes a tree to count the total number of predicates and their occurrences.

    Parameters:
    - nodo (Nodo): The root of the logical tree.

    Returns:
    - tuple: (total_predicates, dict{name: count}), where `name` is the predicate name.
    '''
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
    '''
    Checks if a given tree represents a basic tautology.

    Recognized patterns include:
    - pred(x) OR NOT pred(x)
    - pred(x) => pred(x)

    Parameters:
    - nodo (Nodo): The root of the logical tree.

    Returns:
    - bool: True if the tree represents a tautology, False otherwise.
    '''
    
    if nodo.tipo_nodo == "QUANTIFICATORE":
        # Controlla se il corpo è una tautologia
        return is_tautology(nodo.figli[1])
    
    if nodo.tipo_nodo == "OPERATORE":
        op = nodo.valore.upper()
        
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
                
            if (right.tipo_nodo == "PREDICATO" and
                left.tipo_nodo == "OPERATORE" and left.valore.upper() == "NOT" and
                len(left.figli) == 1 and left.figli[0].tipo_nodo == "PREDICATO"):
                
                pred_right = right.valore.split('(')[0].strip()
                pred_left = left.figli[0].valore.split('(')[0].strip()
                
                if pred_left == pred_right:
                    return True
                
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
                
    return False
