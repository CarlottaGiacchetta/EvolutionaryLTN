from kb import *


# Setup
kb_formulas = create_kb()

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



# Formula2: FORALL x: NOT(Cat(x) AND Dog(x))
var_x = Nodo("VARIABILE", "x")
cat_x = Nodo("PREDICATO", "Cat(x)")
dog_x = Nodo("PREDICATO", "Dog(x)")
cat_and_dog = Nodo("OPERATORE", "AND", [cat_x, dog_x])
formula2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, cat_and_dog])
print(type(formula2))
#kb_formulas.append(formula2)

print(formula2)
print(kb_formulas)
print(formula2 in kb_formulas)
# Verifica se formula2 è nella KB
if is_formula_in_kb(formula2, kb_formulas):
    print("La formula è già nella KB.")
else:
    print("La formula NON è nella KB.")