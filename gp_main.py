import random
import copy

OPERATORS = ["AND", "OR", "IMPLIES"]
QUANTIFIERS = ["FORALL", "EXISTS"]
PREDICATES = ["Cat", "HasWhiskers", "Dog"]
VARIABLES = ["x"]  # Se vuoi più variabili, aggiungile qui (es. ["x","y"]).


#################################################################
# Definizione del Nodo
#################################################################

class Nodo:
    def __init__(self, tipo_nodo, valore=None, figli=None):
        """
        tipo_nodo: "OPERATORE", "QUANTIFICATORE", "PREDICATO", "VARIABILE"
        valore: es. "AND", "FORALL", "Dog(x)"
        figli: lista di nodi
        """
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if (figli is not None) else []

    def copia(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        if self.tipo_nodo == "PREDICATO":
            return f"{self.valore}"
        elif self.tipo_nodo == "VARIABILE":
            return self.valore
        elif self.tipo_nodo == "OPERATORE":
            if self.valore == "NOT":
                return f"(NOT {self.figli[0]})"
            else:
                # assumiamo operatore binario
                return f"({self.figli[0]} {self.valore} {self.figli[1]})"
        elif self.tipo_nodo == "QUANTIFICATORE":
            var_node = self.figli[0]   # VARIABILE
            body = self.figli[1]       # sottoformula
            return f"({self.valore} {var_node.valore}: {body})"
        else:
            return f"UNKNOWN({self.valore})"

    def valida_nodo(self):
        """
        Controlli base sul numero di figli.
        """
        if self.tipo_nodo == "OPERATORE":
            if self.valore == "NOT":
                if len(self.figli) != 1:
                    print("OPERATORE NOT deve avere 1 figlio, trovato", len(self.figli))
                    return False
            elif self.valore in OPERATORS:
                if len(self.figli) != 2:
                    print(f"OPERATORE {self.valore} deve avere 2 figli")
                    return False
        elif self.tipo_nodo == "QUANTIFICATORE":
            # 2 figli: [VARIABILE, sottoformula]
            if len(self.figli) != 2:
                print("QUANTIFICATORE deve avere 2 figli")
                return False
            # figlio[0] deve essere VARIABILE
            if self.figli[0].tipo_nodo != "VARIABILE":
                print("QUANTIFICATORE: primo figlio deve essere VARIABILE")
                return False
        elif self.tipo_nodo == "PREDICATO":
            if len(self.figli) != 0:
                print("PREDICATO deve avere 0 figli, trovato", len(self.figli))
                return False
        elif self.tipo_nodo == "VARIABILE":
            if len(self.figli) != 0:
                print("VARIABILE deve avere 0 figli")
                return False
        return True


#################################################################
# Parsing di "Dog(x)" -> ("Dog","x")
#################################################################

def parse_predicato(value_str):
    """
    "Dog(x)" -> ("Dog","x")
    """
    if not value_str.endswith(")"):
        raise ValueError(f"Predicato malformato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    var_name = value_str[idx_par+1:-1]  # es. "x"
    return pred_name, var_name


#################################################################
# Funzioni di building formula LTN da un albero
#################################################################

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
    if nodo.tipo_nodo == "PREDICATO":
        pred_name, var_name = parse_predicato(nodo.valore)
        # ad es. "Dog","x"
        ltn_pred = ltn_dict[pred_name]
        if var_name not in scope_vars:
            raise ValueError(f"Variabile {var_name} non presente in scope_vars!")
        return ltn_pred(scope_vars[var_name])

    elif nodo.tipo_nodo == "VARIABILE":
        # In logica LTN, una variabile nuda di solito non produce formula,
        # potresti restituire scope_vars[nodo.valore] se vuoi un LTNObject
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
            op_obj = ltn_dict[nodo.valore]  # ad es. "AND", "OR", "IMPLIES" -> Connective corrispondente
            return op_obj(left_ltn, right_ltn)

    elif nodo.tipo_nodo == "QUANTIFICATORE":
        quant_name = nodo.valore  # "FORALL" o "EXISTS"
        var_node = nodo.figli[0]  # var
        body_node = nodo.figli[1]
        new_var_name = var_node.valore
        # scoping
        new_scope = scope_vars.copy()
        if new_var_name not in new_scope:
            raise ValueError(f"Variabile {new_var_name} non definita in scope_vars!")
        body_ltn = build_ltn_formula_node(body_node, ltn_dict, new_scope)
        quant_op = ltn_dict[quant_name]  # Forall or Exists
        return quant_op(new_scope[new_var_name], body_ltn)

    else:
        raise ValueError(f"Nodo di tipo sconosciuto: {nodo.tipo_nodo}")

#################################################################
# Classe Albero
#################################################################

class Albero:
    def __init__(self):
        """
        Inizialmente: (QUANT var: (pred(var) OP pred(var)))
        """
        var = random.choice(VARIABLES)
        var_node = Nodo("VARIABILE", var, [])

        op = random.choice(OPERATORS)
        left_pred = Nodo("PREDICATO", f"{random.choice(PREDICATES)}({var})")
        right_pred = Nodo("PREDICATO", f"{random.choice(PREDICATES)}({var})")
        operator_node = Nodo("OPERATORE", op, [left_pred, right_pred])

        q = random.choice(QUANTIFIERS)
        self.radice = Nodo("QUANTIFICATORE", q, [var_node, operator_node])

        self.profondita = self.calcola_profondita(self.radice)

    def copia(self):
        nuovo = Albero.__new__(Albero)
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
        assumendo che scope_vars contenga le ltn.Variable pertinenti (es. "x": x).
        """
        return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)


#################################################################
# get_all_nodes e replace_node_in_tree
#################################################################

def get_all_nodes(nodo):
    """
    Ritorna TUTTI i nodi, compreso radice e quantificatore,
    ma per il CROSSOVER vogliamo escludere QUANTIFICATORI e VARIABILI 
    per evitare sostituzioni insensate.
    """
    nodes = [nodo]
    for f in nodo.figli:
        nodes.extend(get_all_nodes(f))
    return nodes

def replace_node_in_tree(tree, old_node, new_subtree):
    """
    Se trova old_node in `tree` (per reference), lo sostituisce con new_subtree (deepcopy).
    Ritorna il nodo sostituito (nuovo) oppure None se non trovato.
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
    Facciamo crossover su nodi di tipo OPERATORE (binario o NOT) o PREDICATO,
    evitando QUANTIFICATORE e VARIABILE.
    Evitiamo anche di sostituire la radice se è QUANTIFICATORE (ha meno senso).
    """
    if random.random() > prob:
        return a1.copia(), a2.copia()

    c1 = a1.copia()
    c2 = a2.copia()

    # raccogli i nodi dai due alberi, 
    # MA escludiamo QUANTIFICATORI e VARIABILI e la radice se e' QUANT
    # (o la radice in generale se vuoi).
    n1_all = get_all_nodes(c1.radice)
    n2_all = get_all_nodes(c2.radice)

    # Filtriamo i nodi scambiabili
    def is_swappable(n):
        if n.tipo_nodo in ("QUANTIFICATORE", "VARIABILE"):
            return False
        return True

    n1 = [nd for nd in n1_all if is_swappable(nd)]
    n2 = [nd for nd in n2_all if is_swappable(nd)]

    if not n1 or not n2:
        # se non c'e' nulla da scambiare, restituiamo le copie
        return c1, c2

    old1 = random.choice(n1)
    old2 = random.choice(n2)

    # Crea copie dei sottoalberi
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

def get_scope_vars(root, target):
    """
    Ritorna la lista di variabili quantificate dal path radice->target.
    """
    path = find_path(root, target)
    if not path:
        return []
    scope_vars = []
    for node in path:
        if node.tipo_nodo == "QUANTIFICATORE":
            scope_vars.append(node.figli[0].valore)
    return scope_vars

def find_path(root, target):
    if root is target:
        return [root]
    for child in root.figli:
        subpath = find_path(child, target)
        if subpath:
            return [root] + subpath
    return []

def mutate(albero: Albero, prob=0.3):
    """
    Manteniamo la logica di mutazione:
    - Cambia operatore (binario) 
    - Cambia predicato -> riassegna la var in scope
    - Aggiungi NOT intorno a predicato
    - Avvolgi in quantificatore (se non predicato) 
    (sapendo che i quantificatori e le variabili rimangono invariati)
    """
    if random.random() > prob:
        return albero.copia()

    new_tree = albero.copia()
    nodes_all = get_all_nodes(new_tree.radice)

    # Filtriamo fuori QUANTIFICATORE e VARIABILE
    # se vogliamo non mutare i quantificatori/variabili
    def is_mutable(n):
        if n.tipo_nodo in ["QUANTIFICATORE", "VARIABILE"]:
            return False
        return True

    candidates = [nd for nd in nodes_all if is_mutable(nd)]
    if not candidates:
        return new_tree  # nulla da mutare

    target = random.choice(candidates)

    r = random.random()
    # 1) Cambiare operatore binario
    if target.tipo_nodo == "OPERATORE" and target.valore in OPERATORS and r < 0.25:
        old_op = target.valore
        new_op = random.choice([op for op in OPERATORS if op != old_op])
        target.valore = new_op

    # 2) Cambiare predicato
    elif target.tipo_nodo == "PREDICATO" and r < 0.5:
        scopevars = get_scope_vars(new_tree.radice, target)
        var = scopevars[0] if scopevars else "x"  # se non c'e', usiamo "x"
        pred_name = random.choice(PREDICATES)
        target.valore = f"{pred_name}({var})"

    # 3) Aggiungi negazione se PREDICATO
    elif target.tipo_nodo == "PREDICATO" and r < 0.75:
        not_node = Nodo("OPERATORE", "NOT", [target.copia()])
        replace_node_in_tree(new_tree.radice, target, not_node)

    # 4) Avvolgi in un quantificatore se non predicato e ha figli
    else:
        if target.tipo_nodo != "PREDICATO" and target.figli:
            # costruiamo un quant se c'e' un var
            # assumiamo che "x" e' la var
            var_node = Nodo("VARIABILE", "x", [])
            q = random.choice(QUANTIFIERS)
            qnode = Nodo("QUANTIFICATORE", q, [var_node, target.copia()])
            replace_node_in_tree(new_tree.radice, target, qnode)

    new_tree.profondita = new_tree.calcola_profondita(new_tree.radice)
    return new_tree


#########################################################################
# Esempio di test
#########################################################################
if __name__ == "__main__":
    a1 = Albero()
    print("Albero1 iniziale:", a1, "profondità=", a1.profondita, "valido?", a1.valida_albero())
    a2 = Albero()
    print("Albero2 iniziale:", a2, "profondità=", a2.profondita, "valido?", a2.valida_albero())

    print("\n-- CROSSOVER 1 --")
    child1, child2 = crossover(a1, a2, prob=0.9)
    print("Child1:", child1, "depth=", child1.profondita, "valid?", child1.valida_albero())
    print("Child2:", child2, "depth=", child2.profondita, "valid?", child2.valida_albero())

    print("\n-- CROSSOVER 2 --")
    # facciamo un secondo crossover su child1, child2
    child3, child4 = crossover(child1, child2, prob=0.9)
    print("Child3:", child3, "depth=", child3.profondita, "valid?", child3.valida_albero())
    print("Child4:", child4, "depth=", child4.profondita, "valid?", child4.valida_albero())

    # Mutazione
    print("\n-- MUTATION Child1 --")
    m1 = mutate(child1, prob=0.9)
    print("Mutated:", m1, "depth=", m1.profondita, "valid?", m1.valida_albero())
