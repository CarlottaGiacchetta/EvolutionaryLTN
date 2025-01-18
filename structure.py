import random
import copy
import torch.nn as nn

# Liste di possibili scelte
OPERATORS = ["AND", "OR", "IMPLIES"]
QUANTIFIERS = ["FORALL", "EXISTS"]
PREDICATES = ["Cat", "HasWhiskers", "Dog"]#, "Parent", "Likes", "Owner"
VARIABLES = ["x"]  # , "y" Puoi aggiungerne altre, es. "z"

#################################################################
# Definizione del Nodo
#################################################################

class Nodo:
    def __init__(self, tipo_nodo, valore=None, figli=None):
        """
        tipo_nodo: "OPERATORE", "QUANTIFICATORE", "PREDICATO", "VARIABILE"
        valore: es. "AND", "FORALL", "Dog(x,y)", ...
        figli: lista di nodi (vuota per i predicati e le variabili)
        """
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if (figli is not None) else []

    def copia(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        """
        Stampa ricorsiva del nodo.
        """
        if self.tipo_nodo == "PREDICATO":
            # es: "Dog(x,y)"
            return f"{self.valore}"
        elif self.tipo_nodo == "VARIABILE":
            # es: "x"
            return self.valore
        elif self.tipo_nodo == "OPERATORE":
            # assumiamo operatore binario o 'NOT' unario
            if self.valore == "NOT":
                return f"(NOT {self.figli[0]})"
            else:
                # es. (child0 AND child1)
                return f"({self.figli[0]} {self.valore} {self.figli[1]})"
        elif self.tipo_nodo == "QUANTIFICATORE":
            # (FORALL x: <body>)
            var_node = self.figli[0]   # nodo VARIABILE
            body = self.figli[1]       # sottoformula
            return f"({self.valore} {var_node.valore}: {body})"
        else:
            return f"UNKNOWN({self.valore})"

    def valida_nodo(self):
        """
        Controlli base sul numero di figli e coerenza con tipo_nodo.
        """
        if self.tipo_nodo == "OPERATORE":
            if self.valore == "NOT":
                if len(self.figli) != 1:
                    print("OPERATORE NOT deve avere 1 figlio, trovato", len(self.figli))
                    return False
            elif self.valore in OPERATORS:
                if len(self.figli) != 2:
                    print(f"OPERATORE {self.valore} deve avere 2 figli (binario).")
                    return False

        elif self.tipo_nodo == "QUANTIFICATORE":
            # 2 figli: [nodo VARIABILE, sottoformula]
            if len(self.figli) != 2:
                print("QUANTIFICATORE deve avere 2 figli (VARIABILE, body).")
                return False
            if self.figli[0].tipo_nodo != "VARIABILE":
                print("QUANTIFICATORE: primo figlio deve essere VARIABILE.")
                return False

        elif self.tipo_nodo == "PREDICATO":
            # In questa rappresentazione, i predicati non hanno nodi figli (gli argomenti sono nel 'valore' string)
            if len(self.figli) != 0:
                print("PREDICATO deve avere 0 figli, trovato", len(self.figli))
                return False

        elif self.tipo_nodo == "VARIABILE":
            # Nessun figlio
            if len(self.figli) != 0:
                print("VARIABILE deve avere 0 figli.")
                return False

        return True

#################################################################
# Parsing di predicato multi-argomento: "Dog(x,y,z)" -> ("Dog", ["x","y","z"])
#################################################################

def parse_predicato(value_str):
    """
    Esempi:
    "Dog(x)"               -> ("Dog", ["x"])
    "Parent(x, y)"         -> ("Parent", ["x","y"])
    "Likes(x, y, z)"       -> ("Likes", ["x","y","z"])
    """
    if not value_str.endswith(")"):
        raise ValueError(f"Predicato malformato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    args_str = value_str[idx_par+1:-1]  # es. "x,y"
    arg_list = [arg.strip() for arg in args_str.split(",")]
    # arg_list es. ["x","y"]
    return pred_name, arg_list

#################################################################
# Funzioni di building formula LTN da un albero
#################################################################

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
    """
    Dato un nodo (PREDICATO / OPERATORE / QUANTIFICATORE / VARIABILE), 
    costruisce ricorsivamente la formula LTN corrispondente.
    """
    if nodo.tipo_nodo == "PREDICATO":
        pred_name, var_names = parse_predicato(nodo.valore)
        # Recupera l'oggetto LTN corrispondente
        if pred_name not in ltn_dict:
            raise ValueError(f"Predicato {pred_name} non definito in ltn_dict!")
        ltn_pred = ltn_dict[pred_name]

        # Converti i var_names in ltn.Variable
        ltn_vars = []
        for var_name in var_names:
            if var_name not in scope_vars:
                raise ValueError(f"Variabile {var_name} non presente in scope_vars!")
            ltn_vars.append(scope_vars[var_name])

        # Applica il predicato
        return ltn_pred(*ltn_vars)

    elif nodo.tipo_nodo == "VARIABILE":
        # In logica LTN potresti semplicemente restituire scope_vars[nodo.valore]
        var_name = nodo.valore
        if var_name not in scope_vars:
            raise ValueError(f"Variabile {var_name} non presente in scope_vars!")
        return scope_vars[var_name]

    elif nodo.tipo_nodo == "OPERATORE":
        if nodo.valore == "NOT":
            child_ltn = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            return ltn_dict["NOT"](child_ltn)
        else:
            left_ltn = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            right_ltn = build_ltn_formula_node(nodo.figli[1], ltn_dict, scope_vars)
            op_obj = ltn_dict[nodo.valore]
            return op_obj(left_ltn, right_ltn)

    elif nodo.tipo_nodo == "QUANTIFICATORE":
        quant_name = nodo.valore  # "FORALL" o "EXISTS"
        var_node = nodo.figli[0]  # VARIABILE
        body_node = nodo.figli[1]
        new_var_name = var_node.valore

        # scoping
        new_scope = scope_vars.copy()
        if new_var_name not in new_scope:
            raise ValueError(f"Variabile {new_var_name} non definita in scope_vars!")
        body_ltn = build_ltn_formula_node(body_node, ltn_dict, new_scope)
        quant_op = ltn_dict[quant_name]
        return quant_op(new_scope[new_var_name], body_ltn)

    else:
        raise ValueError(f"Nodo di tipo sconosciuto: {nodo.tipo_nodo}")


#################################################################
# Classe Albero
#################################################################

class Albero:
    def __init__(self, VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES):
        """
        Esempio di costruttore semplice: 
        (QUANT var: (Pred(...var...) OP Pred(...var...)))
        """
        self.VARIABLES = VARIABLES
        self.OPERATORS = OPERATORS
        self.QUANTIFIERS = QUANTIFIERS
        self.PREDICATES = PREDICATES

        # Scegli una variabile
        var = random.choice(self.VARIABLES)
        var_node = Nodo("VARIABILE", var, [])

        # Crea un operatore binario i cui predicati usano "var"
        op = random.choice(self.OPERATORS)
        left_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        right_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        operator_node = Nodo("OPERATORE", op, [left_pred, right_pred])

        # Quantificatore
        q = random.choice(self.QUANTIFIERS)
        self.radice = Nodo("QUANTIFICATORE", q, [var_node, operator_node])

        self.profondita = self.calcola_profondita(self.radice)

    def copia(self):
        nuovo = Albero.__new__(Albero)
        nuovo.VARIABLES = self.VARIABLES
        nuovo.OPERATORS = self.OPERATORS
        nuovo.QUANTIFIERS = self.QUANTIFIERS
        nuovo.PREDICATES = self.PREDICATES
        nuovo.radice = self.radice.copia()
        nuovo.profondita = self.profondita
        return nuovo

    def calcola_profondita(self, nodo):
        if not nodo.figli:
            return 1
        return 1 + max(self.calcola_profondita(c) for c in nodo.figli)

    def valida_albero(self):
        stack = [self.radice]
        albero_valido = True
        while stack:
            nodo = stack.pop()
            if not nodo.valida_nodo():
                albero_valido = False
                break
            stack.extend(nodo.figli)
        return albero_valido
    
    def __repr__(self):
        return str(self.radice)

    def to_ltn_formula(self, ltn_dict, scope_vars):
        """
        Converte l'albero in formula LTN, 
        assumendo che scope_vars contenga le ltn.Variable pertinenti.
        """
        return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)

#################################################################
# definizione delle funzioni
#################################################################

def formula1(kb_formulas):
    # Formula 1: Cat(x) -> HasWhiskers(x)
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    has_whiskers_x = Nodo("PREDICATO", "HasWhiskers(x)")
    implies_formula = Nodo("OPERATORE", "IMPLIES", [cat_x, has_whiskers_x])
    formula1 = Nodo("QUANTIFICATORE", "FORALL", [var_x, implies_formula])
    kb_formulas.append(formula1)

def formula2(kb_formulas):
    # Formula 2: Dog(x) -> NOT(Cat(x))
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    dog_x = Nodo("PREDICATO", "Dog(x)")
    not_cat_x = Nodo("OPERATORE", "NOT", [cat_x])
    implies_formula2 = Nodo("OPERATORE", "IMPLIES", [dog_x, not_cat_x])
    formula2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, implies_formula2])
    kb_formulas.append(formula2)

def formula3(kb_formulas):
    # Formula 3: EXISTS x: Cat(x) OR Dog(x)
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    dog_x = Nodo("PREDICATO", "Dog(x)")
    or_formula = Nodo("OPERATORE", "OR", [cat_x, dog_x])
    formula3 = Nodo("QUANTIFICATORE", "EXISTS", [var_x, or_formula])
    kb_formulas.append(formula3)



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

#################################################################
# MUTATE
#################################################################

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

def mutate(albero: Albero, prob=0.3):
    """
    Esempio di mutazione:
    - Cambia operatore (binario)
    - Cambia predicato (att.ne multi-argument)
    - Aggiungi NOT a un predicato
    - Avvolgi in quantificatore
    """
    if random.random() > prob:
        return albero.copia()

    new_tree = albero.copia()
    nodes_all = get_all_nodes(new_tree.radice)

    def is_mutable(n):
        return n.tipo_nodo in ["OPERATORE", "PREDICATO"]

    candidates = [nd for nd in nodes_all if is_mutable(nd)]
    if not candidates:
        return new_tree

    target = random.choice(candidates)
    r = random.random()

    # 1) Cambiare operatore binario
    if target.tipo_nodo == "OPERATORE" and target.valore in OPERATORS and r < 0.25:
        old_op = target.valore
        new_op = random.choice([op for op in OPERATORS if op != old_op])
        target.valore = new_op

    # 2) Cambiare predicato (multi-arg)
    elif target.tipo_nodo == "PREDICATO" and r < 0.5:
        scopevars = get_scope_vars(new_tree.radice, target)
        if not scopevars:
            var_list = ["x"]  # fallback
        else:
            # potresti sceglierne 1 o 2 da scopevars
            # per semplicità scegliamo 1
            var_list = [random.choice(scopevars)]
        new_pred = random.choice(albero.PREDICATES)  # usiamo la PREDICATES da new_tree
        # se vuoi multipli argomenti: decidi random
        # es. 50% di 2 argomenti
        if random.random() < 0.5 and len(scopevars) > 1:
            var_list.append(random.choice(scopevars))
        # costruiamo "pred(var1, var2, ...)"
        var_str = ",".join(var_list)
        target.valore = f"{new_pred}({var_str})"

    # 3) Aggiungi negazione se PREDICATO
    elif target.tipo_nodo == "PREDICATO" and r < 0.75:
        not_node = Nodo("OPERATORE", "NOT", [target.copia()])
        replace_node_in_tree(new_tree.radice, target, not_node)

    # 4) Avvolgi in quantificatore
    else:
        if target.tipo_nodo != "PREDICATO" and target.figli:
            # Scegli una var a caso
            new_var = random.choice(albero.VARIABLES)
            var_node = Nodo("VARIABILE", new_var, [])
            q = random.choice(albero.QUANTIFIERS)
            qnode = Nodo("QUANTIFICATORE", q, [var_node, target.copia()])
            replace_node_in_tree(new_tree.radice, target, qnode)

    new_tree.profondita = new_tree.calcola_profondita(new_tree.radice)
    return new_tree


# Example of a small MLP for a unary predicate
def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
    return nn.Sequential(
        nn.Linear(in_features, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid()  # output in [0,1]
    )


def get_neighbors(matrix, row, col):
    """
    Restituisce gli 8 vicini di un elemento nella matrice 2D.

    Parametri:
        matrix: np.ndarray
            La matrice 2D della popolazione.
        row: int
            Indice di riga dell'individuo corrente.
        col: int
            Indice di colonna dell'individuo corrente.

    Ritorna:
        neighbors: list
            Lista dei vicini (può avere meno di 8 elementi ai bordi).
    """
    neighbors = []
    rows, cols, _ = matrix.shape

    # Coordinate relative dei vicini
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # sopra
        (0, -1),         (0, 1),      # sinistra e destra
        (1, -1), (1, 0), (1, 1)       # sotto
    ]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc

        # Controlla che le coordinate siano dentro i limiti della matrice
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append(matrix[new_row, new_col])

    return neighbors
