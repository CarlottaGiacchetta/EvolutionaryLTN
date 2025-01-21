import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
from structure import Nodo, build_ltn_formula_node, make_unary_predicate

# Definizione della KB
def create_kb():
    kb_formulas = []

    # Formula 1: FORALL x: Cat(x) -> HasWhiskers(x)
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    has_whiskers_x = Nodo("PREDICATO", "HasWhiskers(x)")
    implies_formula = Nodo("OPERATORE", "IMPLIES", [cat_x, has_whiskers_x])
    formula1 = Nodo("QUANTIFICATORE", "FORALL", [var_x, implies_formula])
    kb_formulas.append(formula1)

    # Formula2: FORALL x: NOT(Cat(x) AND Dog(x))
    cat_x = Nodo("PREDICATO", "Cat(x)")
    dog_x = Nodo("PREDICATO", "Dog(x)")
    cat_and_dog = Nodo("OPERATORE", "AND", [cat_x, dog_x])
    not_cat_and_dog = Nodo("OPERATORE", "NOT", [cat_and_dog])
    formula2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, not_cat_and_dog])
    kb_formulas.append(formula2)

    # Formula 3: EXISTS x: Cat(x) OR Dog(x)
    or_formula = Nodo("OPERATORE", "OR", [cat_x, dog_x])
    formula3 = Nodo("QUANTIFICATORE", "EXISTS", [var_x, or_formula])
    kb_formulas.append(formula3)

    return kb_formulas

# Setup LTN
def setup_ltn(kb_formulas):
    constants = {
        "Fluffy": ltn.Constant(torch.randn(2), trainable=False),
        "Garfield": ltn.Constant(torch.randn(2), trainable=False),
        "Rex": ltn.Constant(torch.randn(2), trainable=False),
    }

    tmp_x = torch.stack([constants[name].value for name in constants.keys()], dim=0)

    variables = {
        "x": ltn.Variable("x", tmp_x, add_batch_dim=False),
    }

    predicates = {
        "Cat": ltn.Predicate(make_unary_predicate()),
        "Dog": ltn.Predicate(make_unary_predicate()),
        "HasWhiskers": ltn.Predicate(make_unary_predicate()),
    }

    quantifiers = {
        "FORALL": ltn.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "EXISTS": ltn.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier="e"),
    }

    operators = {
        "AND": ltn.Connective(fuzzy_ops.AndProd()),
        "OR": ltn.Connective(fuzzy_ops.OrMax()),
        "IMPLIES": ltn.Connective(fuzzy_ops.ImpliesLuk()),
        "NOT": ltn.Connective(fuzzy_ops.NotStandard()),
    }

    ltn_dict = {**constants, **predicates, **quantifiers, **operators}
    # Ottimizzatore
    parameters = []
    for pred in predicates.values():
        parameters += list(pred.model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # Training Loop
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = kb_loss(kb_formulas, ltn_dict, variables)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Valutazione Finale
    print("\n--- Valutazione Finale ---")
    for i, formula in enumerate(kb_formulas, start=1):
        ltn_formula = build_ltn_formula_node(formula, ltn_dict, variables)
        satisfaction = ltn_formula.value.mean().item()
        print(f"Formula {i}: {formula}")
        print(f"Soddisfazione: {satisfaction:.4f}")
    
    return ltn_dict, variables


    

# Loss Function
def kb_loss(kb_formulas, ltn_dict, variables):
    total_loss = 0
    for formula in kb_formulas:
        ltn_formula = build_ltn_formula_node(formula, ltn_dict, variables)
        total_loss += (1 - ltn_formula.value.mean())
    return total_loss


def compare_nodes(node1, node2):
    print('stampo nodo 1 e nodo 2')
    print(node1)
    print(node2)
    """
    Confronta due nodi ricorsivamente (struttura, tipo e valori).
    """
    if node1.tipo_nodo != node2.tipo_nodo:
        print('node1.tipo_nodo != node2.tipo_nodo')
        print(node1.tipo_nodo, node2.tipo_nodo)
        return False
    if node1.valore != node2.valore:
        print('node1.valore != node2.valore')
        print(node1.valore, node2.valore)
        return False
    if len(node1.figli) != len(node2.figli):
        print('len(node1.figli) != len(node2.figli)')
        print(len(node1.figli), len(node2.figli))
        return False
    for child1, child2 in zip(node1.figli, node2.figli):
        print('sono nel for -> chiamata ricorsiva')
        if not compare_nodes(child1, child2):
            return False
    return True



def is_formula_in_kb(formula, kb_formulas):
    """
    Verifica se una formula semanticamente equivalente è già presente nella KB.
    """
    for kb_formula in kb_formulas:
        print()
        print('stampo kb formula')
        print(kb_formula)
        if compare_nodes(formula, kb_formula):
            return True
        print()
    return False
