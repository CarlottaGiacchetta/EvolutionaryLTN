import random
import copy
import numpy as np

import torch
import torch.nn as nn
import ltn
import ltn.fuzzy_ops as fuzzy_ops

# ====================================================
# 1) LISTE DI POSSIBILI TOKEN
# ====================================================
OPERATORS = ["AND", "OR", "IMPLIES"]
QUANTIFIERS = ["FORALL", "EXISTS"]
PREDICATES = ["Cat", "HasWhiskers", "Dog"]
VARIABLES = ["x"]

# ====================================================
# 2) FUNZIONI PER LA KNOWLEDGE BASE & TRAINING LTN
# ====================================================

from typing import List

class Nodo:
    """
    Struttura di nodo per costruire un albero logico:
    - tipo_nodo: "OPERATORE", "QUANTIFICATORE", "PREDICATO", "VARIABILE"
    - valore: es. "AND", "FORALL", "Cat(x)", ...
    - figli: lista di Nodi (vuota per PREDICATO, VARIABILE)
    """
    def __init__(self, tipo_nodo, valore=None, figli=None):
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if figli else []

    def __repr__(self):
        if self.tipo_nodo == "PREDICATO":
            return self.valore
        elif self.tipo_nodo == "VARIABILE":
            return self.valore
        elif self.tipo_nodo == "OPERATORE":
            if self.valore == "NOT":
                return f"NOT({self.figli[0]})"
            else:
                return f"{self.valore}({self.figli[0]},{self.figli[1]})"
        elif self.tipo_nodo == "QUANTIFICATORE":
            return f"{self.valore}({self.figli[0].valore},{self.figli[1]})"
        else:
            return f"UNKNOWN({self.valore})"

def parse_predicato(value_str):
    """
    Esempio: "Cat(x)" -> ("Cat", ["x"])
    """
    if not value_str.endswith(")"):
        raise ValueError(f"Errore parse_predicato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    args_str = value_str[idx_par+1:-1]  # "x,y"
    arg_list = [arg.strip() for arg in args_str.split(",")]
    return pred_name, arg_list

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
    """
    Ricostruisce formula LTN da un nodo.
    """
    if nodo.tipo_nodo == "PREDICATO":
        pred_name, var_names = parse_predicato(nodo.valore)
        ltn_pred = ltn_dict[pred_name]
        ltn_vars = []
        for v in var_names:
            if v not in scope_vars:
                raise ValueError(f"Variabile {v} non in scope!")
            ltn_vars.append(scope_vars[v])
        return ltn_pred(*ltn_vars)

    elif nodo.tipo_nodo == "VARIABILE":
        var_name = nodo.valore
        if var_name not in scope_vars:
            raise ValueError(f"Variabile {var_name} non in scope!")
        return scope_vars[var_name]

    elif nodo.tipo_nodo == "OPERATORE":
        if nodo.valore == "NOT":
            child_node = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            return ltn_dict["NOT"](child_node)
        else:
            left = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            right = build_ltn_formula_node(nodo.figli[1], ltn_dict, scope_vars)
            return ltn_dict[nodo.valore](left, right)

    elif nodo.tipo_nodo == "QUANTIFICATORE":
        quant_name = nodo.valore
        var_node = nodo.figli[0]  # nodo di tipo VARIABILE
        body_node = nodo.figli[1]
        new_var_name = var_node.valore
        new_scope = scope_vars.copy()
        if new_var_name not in new_scope:
            raise ValueError(f"Variabile {new_var_name} non definita!")
        body_ltn = build_ltn_formula_node(body_node, ltn_dict, new_scope)
        return ltn_dict[quant_name](new_scope[new_var_name], body_ltn)

    else:
        raise ValueError(f"Nodo di tipo sconosciuto: {nodo.tipo_nodo}")

def kb_loss(kb_formulas, ltn_dict, variables):
    total_loss = 0
    for f in kb_formulas:
        formula_ltn = build_ltn_formula_node(f, ltn_dict, variables)
        total_loss += (1 - formula_ltn.value.mean())
    return total_loss

def create_kb():
    """
    1) FORALL x: Cat(x) -> HasWhiskers(x)
    2) FORALL x: NOT(Cat(x) AND Dog(x))
    3) EXISTS x: Cat(x) OR Dog(x)
    """
    var_x = Nodo("VARIABILE", "x")
    # 1
    cat_x = Nodo("PREDICATO", "Cat(x)")
    hw_x = Nodo("PREDICATO", "HasWhiskers(x)")
    impl_node = Nodo("OPERATORE", "IMPLIES", [cat_x, hw_x])
    f1 = Nodo("QUANTIFICATORE", "FORALL", [var_x, impl_node])

    # 2
    cat2 = Nodo("PREDICATO", "Cat(x)")
    dog2 = Nodo("PREDICATO", "Dog(x)")
    and_cd = Nodo("OPERATORE", "AND", [cat2, dog2])
    not_cd = Nodo("OPERATORE", "NOT", [and_cd])
    f2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, not_cd])

    # 3
    cat3 = Nodo("PREDICATO", "Cat(x)")
    dog3 = Nodo("PREDICATO", "Dog(x)")
    or_cd = Nodo("OPERATORE", "OR", [cat3, dog3])
    f3 = Nodo("QUANTIFICATORE", "EXISTS", [var_x, or_cd])
    return [f1, f2, f3]

def setup_ltn(kb_formulas, epochs=300):
    """
    Allena i predicati su 3 formule di KB.
    """
    # costanti
    constants = {
        "Fluffy": ltn.Constant(torch.randn(2), trainable=False),
        "Garfield": ltn.Constant(torch.randn(2), trainable=False),
        "Rex": ltn.Constant(torch.randn(2), trainable=False),
    }
    tmp_x = torch.stack([constants[k].value for k in constants], dim=0)
    variables = {
        "x": ltn.Variable("x", tmp_x, add_batch_dim=False)
    }
    # predicati
    def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
        return nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    preds = {
        "Cat": ltn.Predicate(make_unary_predicate()),
        "Dog": ltn.Predicate(make_unary_predicate()),
        "HasWhiskers": ltn.Predicate(make_unary_predicate()),
    }
    quants = {
        "FORALL": ltn.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "EXISTS": ltn.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier="e"),
    }
    ops = {
        "AND": ltn.Connective(fuzzy_ops.AndProd()),
        "OR": ltn.Connective(fuzzy_ops.OrMax()),
        "IMPLIES": ltn.Connective(fuzzy_ops.ImpliesLuk()),
        "NOT": ltn.Connective(fuzzy_ops.NotStandard()),
    }

    ltn_dict = {**constants, **preds, **quants, **ops}

    params = []
    for p in preds.values():
        params += list(p.model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    for ep in range(epochs):
        optimizer.zero_grad()
        loss_val = kb_loss(kb_formulas, ltn_dict, variables)
        loss_val.backward()
        optimizer.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1}/{epochs}, loss={loss_val.item():.4f}")

    # valutazione finale
    print("\nValutazione finale KB:")
    for i, ff in enumerate(kb_formulas, 1):
        val_f = build_ltn_formula_node(ff, ltn_dict, variables).value.mean().item()
        print(f"Formula {i}: {ff} => {val_f:.4f}")

    return ltn_dict, variables

# ====================================================
# 3) PARSE E COSTRUZIONE ALBERO DA STRINGA
# ====================================================

def parse_formula(formula_str):
    """
    Trasforma una stringa tipo:
       "FORALL(x, IMPLIES(Cat(x), Dog(x)))"
    in un albero di Nodi (Nodo).
    """
    # rimuovi spazi superflui
    s = formula_str.strip()
    return parse_expr(s)

def parse_expr(s):
    """
    Parser ricorsivo "semplice" che gestisce:
      - QUANTIFICATORI: FORALL(var, sottoformula)
      - OPERATORS BINARI: AND(expr, expr), OR(expr, expr), IMPLIES(expr, expr)
      - NOT(expr)
      - PREDICATI: Cat(x), Dog(x), HasWhiskers(x)
    """
    s = s.strip()
    # Se trovo "FORALL(" o "EXISTS(" --> QUANTIFICATORE
    for q in QUANTIFIERS:
        q_pattern = q + "("
        if s.startswith(q_pattern):
            # Esempio "FORALL(x, expr)"
            # cerco la virgola che separa la variabile dal body
            inside = s[len(q_pattern):-1]  # tolgo q_pattern e ultima parentesi
            # inside es "x, IMPLIES(Cat(x),Dog(x))"
            comma_idx = find_main_comma(inside)
            var_part = inside[:comma_idx].strip()
            body_part = inside[comma_idx+1:].strip()
            var_node = Nodo("VARIABILE", var_part)
            body_node = parse_expr(body_part)
            return Nodo("QUANTIFICATORE", q, [var_node, body_node])
    
    # Se trovo "NOT("
    if s.startswith("NOT("):
        # parse figlio
        inside = s[4:-1].strip()
        child_node = parse_expr(inside)
        return Nodo("OPERATORE", "NOT", [child_node])

    # Se trovo un pattern "OP(..., ...)" => AND, OR, IMPLIES
    for op in OPERATORS:
        op_pattern = op + "("
        if s.startswith(op_pattern):
            # Esempio "AND(expr1, expr2)"
            inside = s[len(op_pattern):-1]  # tolgo "AND(" e ")" finale
            comma_idx = find_main_comma(inside)
            left_part = inside[:comma_idx].strip()
            right_part = inside[comma_idx+1:].strip()
            left_node = parse_expr(left_part)
            right_node = parse_expr(right_part)
            return Nodo("OPERATORE", op, [left_node, right_node])

    # Altrimenti, se non ho trovato operatori/quantificatori => PREDICATO?
    # Tipo "Cat(x)"
    if "(" in s:
        # predicato
        return Nodo("PREDICATO", s)
    else:
        # caso limitato in cui è solo "x"? (VARIABILE "libera"?)
        return Nodo("VARIABILE", s)

def find_main_comma(s):
    """
    Trova la posizione della virgola "principale" che separa left e right
    tenendo conto di eventuali parentesi annidate.
    """
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return i
    # se non trovata virgola "principale", errore
    raise ValueError(f"Virgola di separazione non trovata in: {s}")

# ====================================================
# 4) FUNZIONI DI GA CLASSICO SU STRINGHE
# ====================================================

def random_formula():
    """
    Genera una formula stringa casuale di profondità piccola.
    (Esempio semplificato.)
    """
    # random se quantificatore oppure espressione pura
    if random.random() < 0.5:
        # quantificatore
        q = random.choice(QUANTIFIERS)
        var = random.choice(VARIABLES)
        # formula interna
        inside = random_subexpr(2)
        return f"{q}({var},{inside})"
    else:
        # no quantificatore
        return random_subexpr(2)

def random_subexpr(depth):
    if depth <= 0:
        # predicato base
        pred = random.choice(PREDICATES)
        var = random.choice(VARIABLES)
        return f"{pred}({var})"
    # altrimenti scelgo se NOT o un op binario
    if random.random() < 0.3:
        # NOT
        child = random_subexpr(depth-1)
        return f"NOT({child})"
    else:
        op = random.choice(OPERATORS)
        left = random_subexpr(depth-1)
        right = random_subexpr(depth-1)
        return f"{op}({left},{right})"

def crossover_str(str1, str2, prob=0.8):
    """
    Esempio di crossover su stringhe:
    - Prende una sottostringa casuale da str1 e la scambia con una sottostringa di str2.
      (Molto naive!)
    """
    if random.random() > prob:
        return str1, str2
    # Trova un punto di taglio random in str1
    cut1a = random.randint(0, len(str1)-1)
    cut1b = random.randint(cut1a, len(str1))
    # Trova un punto di taglio random in str2
    cut2a = random.randint(0, len(str2)-1)
    cut2b = random.randint(cut2a, len(str2))

    substr1 = str1[cut1a:cut1b]
    substr2 = str2[cut2a:cut2b]
    child1 = str1[:cut1a] + substr2 + str1[cut1b:]
    child2 = str2[:cut2a] + substr1 + str2[cut2b:]
    return child1, child2

def mutate_str(formula_str, prob=0.3):
    """
    Esempio di mutazione su stringhe:
    - A volte rimpiazzo 'AND' con 'OR', ecc.
    - A volte inserisco "NOT(...)" intorno a un sottopezzo.
    - Molto semplificato.
    """
    if random.random() > prob:
        return formula_str  # nessuna mutazione

    # piccola probabilità di sostituire un token
    tokens = ["AND", "OR", "IMPLIES", "NOT", "FORALL", "EXISTS"]
    # In modo molto naive, cerco un token a caso da rimpiazzare
    original = formula_str
    for _ in range(10):  # max 10 tentativi
        t = random.choice(tokens)
        if t in formula_str:
            # sostituisci con un token differente
            repl = random.choice([x for x in tokens if x != t])
            formula_str = formula_str.replace(t, repl, 1)
            return formula_str

    # se non ho trovato nulla, aggiungo "NOT(" all'inizio e una ")" alla fine...
    # giusto per variare un po'.
    formula_str = f"NOT({formula_str})"
    return formula_str

# ====================================================
# 5) EVALUAZIONE (FITNESS)
# ====================================================
def evaluate_fitness(formula_str, ltn_dict, scope_vars):
    """
    1) parse la stringa in un albero
    2) build formula LTN
    3) compute mean value
    4) penalizza duplicati di predicati
    """
    try:
        root = parse_formula(formula_str)
        ltn_form = build_ltn_formula_node(root, ltn_dict, scope_vars)
        val = ltn_form.value
        fit = val.mean().item() if val.numel() > 1 else val.item()

        # penalizzazione duplicati
        preds = get_all_predicates(root)
        if len(preds) != len(set(preds)):
            fit *= 0.6

        return fit
    except Exception as e:
        # se fallisce parse o build => fitness = 0
        # (oppure un penalty)
        return 0.0

def get_all_predicates(nodo: Nodo) -> List[str]:
    """
    Ritorna la lista di stringhe di predicati (es "Cat(x)") in modo ricorsivo.
    """
    results = []
    if nodo.tipo_nodo == "PREDICATO":
        results.append(nodo.valore)
    for f in nodo.figli:
        results.extend(get_all_predicates(f))
    return results

# ====================================================
# 6) GA CLASSICO SU STRINGHE MA CON SELEZIONE A VICINI
# ====================================================

def get_neighbors(matrix, row, col):
    neighbors = []
    rows, cols, _ = matrix.shape
    directions = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),         (0,1),
        (1,-1),  (1,0),  (1,1)
    ]
    for dr,dc in directions:
        rr = row+dr
        cc = col+dc
        if 0 <= rr < rows and 0 <= cc < cols:
            neighbors.append(matrix[rr,cc])
    return neighbors

def compute_fitness(popolazione, ltn_dict, scope_vars):
    for i in range(popolazione.shape[0]):
        for j in range(popolazione.shape[1]):
            formula_str = popolazione[i,j][0]
            fit = evaluate_fitness(formula_str, ltn_dict, scope_vars)
            popolazione[i,j][1] = fit
    return popolazione

def evolutionary_run(popolazione, generations, ltn_dict, scope_vars):
    for gen in range(generations):
        print(f"\n--- Generazione {gen+1}/{generations} ---")
        rows, cols, _ = popolazione.shape

        for i in range(rows):
            for j in range(cols):
                # selezione a vicini
                vicini = get_neighbors(popolazione, i, j)
                # ordino per fitness decrescente
                vicini.sort(key=lambda x: x[1], reverse=True)
                parent1 = vicini[0][0]
                parent2 = vicini[1][0]
                # crossover
                c1, c2 = crossover_str(parent1, parent2, prob=0.9)
                # mutazione
                c1 = mutate_str(c1, prob=0.2)
                c2 = mutate_str(c2, prob=0.2)
                # valuto i 2 figli
                f1 = evaluate_fitness(c1, ltn_dict, scope_vars)
                f2 = evaluate_fitness(c2, ltn_dict, scope_vars)
                best_formula, best_fit = (c1, f1) if f1>=f2 else (c2, f2)
                # confronto con occupant
                occupant_fit = popolazione[i,j][1]
                if best_fit > occupant_fit:
                    popolazione[i,j] = [best_formula, best_fit]

        # stampo migliore in popolazione
        all_indivs = [popolazione[r,c] for r in range(rows) for c in range(cols)]
        best_local = max(all_indivs, key=lambda x: x[1])
        print(f"Miglior formula locale: {best_local[0]} (fitness={best_local[1]:.4f})")

    return popolazione

# ====================================================
# 7) MAIN
# ====================================================
if __name__=="__main__":
    # 1) COSTRUISCO E ALLENO LA KB
    kb_formulas = create_kb()
    print("==== TRAINING DELLA KB ====")
    ltn_dict_kb, vars_kb = setup_ltn(kb_formulas, epochs=200)

    # 2) COSTRUISCO DIZIONARIO PER GA (stessi predicati, operatori, quantificatori)
    #    Non servono le costanti, se non le vuoi usare in formula
    ga_ltn_dict = {
        "Cat": ltn_dict_kb["Cat"],
        "Dog": ltn_dict_kb["Dog"],
        "HasWhiskers": ltn_dict_kb["HasWhiskers"],
        "FORALL": ltn_dict_kb["FORALL"],
        "EXISTS": ltn_dict_kb["EXISTS"],
        "AND": ltn_dict_kb["AND"],
        "OR": ltn_dict_kb["OR"],
        "IMPLIES": ltn_dict_kb["IMPLIES"],
        "NOT": ltn_dict_kb["NOT"],
    }
    ga_vars = {
        "x": vars_kb["x"]
    }

    # 3) INIZIALIZZA POPOLAZIONE come NxN
    population_size = 49
    matrix_size = int(np.sqrt(population_size))
    pop_matrix = np.empty((matrix_size, matrix_size, 2), dtype=object)

    for i in range(matrix_size):
        for j in range(matrix_size):
            # formula iniziale random
            form_str = random_formula()
            pop_matrix[i,j,0] = form_str  # la stringa
            pop_matrix[i,j,1] = 0.0       # fitness

    # 4) CALCOLO FITNESS INIZIALE
    pop_matrix = compute_fitness(pop_matrix, ga_ltn_dict, ga_vars)

    # 5) RUN EVOLUTION
    generations = 5
    final_pop = evolutionary_run(pop_matrix, generations, ga_ltn_dict, ga_vars)

    # 6) ORDINAMENTO FINALE E RISULTATI
    final_list = [final_pop[r,c] for r in range(final_pop.shape[0]) for c in range(final_pop.shape[1])]
    final_sorted = sorted(final_list, key=lambda x: x[1], reverse=True)

    print("\n=== RISULTATI FINALI ===")
    for rank, (fs, fitv) in enumerate(final_sorted[:5], start=1):
        print(f"Top {rank}: {fs} => {fitv:.4f}")

    best = final_sorted[0]
    print(f"\nMiglior individuo finale: {best[0]} (fitness={best[1]:.4f})")
