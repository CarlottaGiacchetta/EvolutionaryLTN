import random
import copy
import numpy as np
import torch
import torch.nn as nn
import ltn
import ltn.fuzzy_ops as fuzzy_ops
from typing import List

# ====================================================
# 1) LISTE DI POSSIBILI TOKEN
# ====================================================
OPERATORS = ["AND", "OR", "IMPLIES"]
QUANTIFIERS = ["FORALL", "EXISTS"]
PREDICATES = ["Cat", "HasWhiskers", "Dog"]
VARIABLES = ["x"]

# ====================================================
# 2) FUNZIONI DI SUPPORTO (Parsing e pattern matching)
# ====================================================

def parse_predicato(value_str):
    """Converte una stringa come "Cat(x)" in una tuple ("Cat", ["x"])"""
    if not value_str.endswith(")"):
        raise ValueError(f"Errore parse_predicato: {value_str}")
    idx_par = value_str.index("(")
    pred_name = value_str[:idx_par]
    args_str = value_str[idx_par+1:-1]
    arg_list = [arg.strip() for arg in args_str.split(",")]
    return pred_name, arg_list

def contains_predicate(nodo: 'Nodo', pred: str) -> bool:
    """Verifica se l'albero contiene un nodo PREDICATO che inizia con pred + "("."""
    if nodo.tipo_nodo == "PREDICATO" and nodo.valore.startswith(pred + "("):
        return True
    for child in nodo.figli:
        if contains_predicate(child, pred):
            return True
    return False

def contains_implication(nodo: 'Nodo', pred_from: str, pred_to: str) -> bool:
    """Restituisce True se viene trovato un nodo IMPLIES dove il ramo sinistro contiene pred_from e il destro pred_to."""
    if nodo.tipo_nodo == "OPERATORE" and nodo.valore == "IMPLIES":
        left_contains = contains_predicate(nodo.figli[0], pred_from)
        right_contains = contains_predicate(nodo.figli[1], pred_to)
        if left_contains and right_contains:
            return True
    for child in nodo.figli:
        if contains_implication(child, pred_from, pred_to):
            return True
    return False

def contains_inconsistent_pattern(nodo: 'Nodo') -> bool:
    """
    Restituisce True se viene trovato un nodo AND in cui
    un ramo contiene "Cat(" e l'altro "Dog(".
    """
    if nodo.tipo_nodo == "OPERATORE" and nodo.valore == "AND":
        if (contains_predicate(nodo.figli[0], "Cat") and contains_predicate(nodo.figli[1], "Dog")) or \
           (contains_predicate(nodo.figli[0], "Dog") and contains_predicate(nodo.figli[1], "Cat")):
            return True
    for child in nodo.figli:
        if contains_inconsistent_pattern(child):
            return True
    return False

def penalty_coefficient(root: 'Nodo') -> float:
    """Calcola un coefficiente di penalizzazione basato sui pattern desiderati."""
    coeff = 1.0
    # Penalizza se manca il pattern "Cat(x) IMPLIES HasWhiskers(x)"
    if not contains_implication(root, "Cat", "HasWhiskers"):
        coeff *= 0.9  # Penalizzazione del 10%
    # Penalizza se è presente un pattern incoerente (es. AND(Cat(x), Dog(x)))
    if contains_inconsistent_pattern(root):
        coeff *= 0.8  # Penalizzazione del 20%
    return coeff

# ====================================================
# 3) STRUTTURE DI BASE: Nodo e Albero
# ====================================================

class Nodo:
    """Nodo per l’albero logico."""
    def __init__(self, tipo_nodo, valore=None, figli=None):
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if figli is not None else []
    
    def copia(self):
        return copy.deepcopy(self)
    
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

class Albero:
    """Albero logico: l'individuo evolutivo."""
    def __init__(self, VARIABLES=VARIABLES, OPERATORS=OPERATORS, QUANTIFIERS=QUANTIFIERS, PREDICATES=PREDICATES):
        self.VARIABLES = VARIABLES
        self.OPERATORS = OPERATORS
        self.QUANTIFIERS = QUANTIFIERS
        self.PREDICATES = PREDICATES
        
        var = random.choice(self.VARIABLES)
        var_node = Nodo("VARIABILE", var)
        op = random.choice(self.OPERATORS)
        left_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        right_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        operator_node = Nodo("OPERATORE", op, [left_pred, right_pred])
        q = random.choice(self.QUANTIFIERS)
        self.radice = Nodo("QUANTIFICATORE", q, [var_node, operator_node])
        self.profondita = self.calcola_profondita(self.radice)
    
    def calcola_profondita(self, nodo):
        if not nodo.figli:
            return 1
        return 1 + max(self.calcola_profondita(child) for child in nodo.figli)
    
    def copia(self):
        nuovo = Albero.__new__(Albero)
        nuovo.VARIABLES = self.VARIABLES
        nuovo.OPERATORS = self.OPERATORS
        nuovo.QUANTIFIERS = self.QUANTIFIERS
        nuovo.PREDICATES = self.PREDICATES
        nuovo.radice = self.radice.copia()
        nuovo.profondita = self.calcola_profondita(nuovo.radice)
        return nuovo
    
    def to_ltn_formula(self, ltn_dict, scope_vars):
        try:
            return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)
        except Exception as e:
            self.__init__(self.VARIABLES, self.OPERATORS, self.QUANTIFIERS, self.PREDICATES)
            return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)
    
    def __repr__(self):
        return str(self.radice)

# ====================================================
# 4) BUILDING FORMULA LTN DA UN NODO
# ====================================================

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
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
            child = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            return ltn_dict["NOT"](child)
        else:
            left = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            right = build_ltn_formula_node(nodo.figli[1], ltn_dict, scope_vars)
            return ltn_dict[nodo.valore](left, right)
    elif nodo.tipo_nodo == "QUANTIFICATORE":
        quant = nodo.valore
        var_node = nodo.figli[0]
        body_node = nodo.figli[1]
        new_scope = scope_vars.copy()
        body = build_ltn_formula_node(body_node, ltn_dict, new_scope)
        return ltn_dict[quant](new_scope[var_node.valore], body)
    else:
        raise ValueError(f"Nodo di tipo sconosciuto: {nodo.tipo_nodo}")

# ====================================================
# 5) TRAINING DELLA KB
# ====================================================

def kb_loss(kb_formulas, ltn_dict, variables):
    total_loss = 0
    for idx, formula in enumerate(kb_formulas):
        f_ltn = build_ltn_formula_node(formula, ltn_dict, variables)
        loss_formula = 1 - f_ltn.value.mean()
        if idx == 0:  # il vincolo principale
            loss_formula *= 1.5  # aumenta il peso del 50%
        total_loss += loss_formula
    return total_loss

def create_kb():
    kb_formulas = []
    # Formula 1: FORALL x: Cat(x) -> HasWhiskers(x)
    var_x = Nodo("VARIABILE", "x")
    cat_x = Nodo("PREDICATO", "Cat(x)")
    hw_x = Nodo("PREDICATO", "HasWhiskers(x)")
    impl_formula = Nodo("OPERATORE", "IMPLIES", [cat_x, hw_x])
    formula1 = Nodo("QUANTIFICATORE", "FORALL", [var_x, impl_formula])
    # Duplico per rafforzare questo vincolo
    kb_formulas.append(formula1)
    kb_formulas.append(formula1.copia())
    
    # Formula 2: FORALL x: Dog(x) -> NOT(Cat(x))
    dog_x = Nodo("PREDICATO", "Dog(x)")
    not_cat = Nodo("OPERATORE", "NOT", [Nodo("PREDICATO", "Cat(x)")])
    impl2 = Nodo("OPERATORE", "IMPLIES", [dog_x, not_cat])
    formula2 = Nodo("QUANTIFICATORE", "FORALL", [var_x, impl2])
    kb_formulas.append(formula2)
    
    # Formula 3: EXISTS x: OR(Cat(x), Dog(x))
    or_formula = Nodo("OPERATORE", "OR", [Nodo("PREDICATO", "Cat(x)"), dog_x])
    formula3 = Nodo("QUANTIFICATORE", "EXISTS", [var_x, or_formula])
    kb_formulas.append(formula3)
    
    return kb_formulas

def setup_ltn(kb_formulas, epochs=300):
    constants = {
        "Fluffy": ltn.Constant(torch.randn(2), trainable=False),
        "Garfield": ltn.Constant(torch.randn(2), trainable=False),
        "Rex": ltn.Constant(torch.randn(2), trainable=False)
    }
    tmp_x = torch.stack([constants[k].value for k in constants], dim=0)
    variables = {"x": ltn.Variable("x", tmp_x, add_batch_dim=False)}
    
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
        "HasWhiskers": ltn.Predicate(make_unary_predicate())
    }
    quants = {
        "FORALL": ltn.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "EXISTS": ltn.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier="e")
    }
    ops = {
        "AND": ltn.Connective(fuzzy_ops.AndProd()),
        "OR": ltn.Connective(fuzzy_ops.OrMax()),
        "IMPLIES": ltn.Connective(fuzzy_ops.ImpliesLuk()),
        "NOT": ltn.Connective(fuzzy_ops.NotStandard())
    }
    
    ltn_dict = {**constants, **preds, **quants, **ops}
    
    params = []
    for pred in preds.values():
        params += list(pred.model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    
    for ep in range(epochs):
        optimizer.zero_grad()
        loss_val = kb_loss(kb_formulas, ltn_dict, variables)
        loss_val.backward()
        optimizer.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1}/{epochs}, loss={loss_val.item():.4f}")
    
    print("\nValutazione finale KB:")
    for i, f in enumerate(kb_formulas, 1):
        val_f = build_ltn_formula_node(f, ltn_dict, variables).value.mean().item()
        print(f"Formula {i}: {f} => {val_f:.4f}")
    
    return ltn_dict, variables

# ====================================================
# 6) PARSER DA STRINGA AD ALBERO
# ====================================================

def parse_formula(formula_str):
    s = formula_str.strip()
    return parse_expr(s)

def parse_expr(s):
    s = s.strip()
    # Gestione quantificatori
    for q in QUANTIFIERS:
        q_pattern = q + "("
        if s.startswith(q_pattern):
            inside = s[len(q_pattern):-1]
            comma_idx = find_main_comma(inside)
            var_part = inside[:comma_idx].strip()
            body_part = inside[comma_idx+1:].strip()
            var_node = Nodo("VARIABILE", var_part)
            body_node = parse_expr(body_part)
            return Nodo("QUANTIFICATORE", q, [var_node, body_node])
    # Gestione NOT
    if s.startswith("NOT("):
        inside = s[4:-1].strip()
        child_node = parse_expr(inside)
        return Nodo("OPERATORE", "NOT", [child_node])
    # Gestione operatori binari
    for op in OPERATORS:
        op_pattern = op + "("
        if s.startswith(op_pattern):
            inside = s[len(op_pattern):-1]
            comma_idx = find_main_comma(inside)
            left_part = inside[:comma_idx].strip()
            right_part = inside[comma_idx+1:].strip()
            left_node = parse_expr(left_part)
            right_node = parse_expr(right_part)
            return Nodo("OPERATORE", op, [left_node, right_node])
    # Se non ci sono operatori o quantificatori, si assume sia un predicato
    if "(" in s:
        return Nodo("PREDICATO", s)
    else:
        return Nodo("VARIABILE", s)

def find_main_comma(s):
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return i
    raise ValueError(f"Virgola principale non trovata in: {s}")

# ====================================================
# 7) FUNZIONI DI GA SU STRINGHE
# ====================================================

def random_formula():
    if random.random() < 0.5:
        q = random.choice(QUANTIFIERS)
        var = random.choice(VARIABLES)
        inside = random_subexpr(2)
        return f"{q}({var},{inside})"
    else:
        return random_subexpr(2)

def random_subexpr(depth):
    if depth <= 0:
        pred = random.choice(PREDICATES)
        var = random.choice(VARIABLES)
        return f"{pred}({var})"
    if random.random() < 0.3:
        child = random_subexpr(depth-1)
        return f"NOT({child})"
    else:
        op = random.choice(OPERATORS)
        left = random_subexpr(depth-1)
        right = random_subexpr(depth-1)
        return f"{op}({left},{right})"

def crossover_str(str1, str2, prob=0.8):
    if random.random() > prob:
        return str1, str2
    cut1a = random.randint(0, len(str1)-1)
    cut1b = random.randint(cut1a, len(str1))
    cut2a = random.randint(0, len(str2)-1)
    cut2b = random.randint(cut2a, len(str2))
    substr1 = str1[cut1a:cut1b]
    substr2 = str2[cut2a:cut2b]
    child1 = str1[:cut1a] + substr2 + str1[cut1b:]
    child2 = str2[:cut2a] + substr1 + str2[cut2b:]
    return child1, child2

def mutate_str(formula_str, prob=0.3):
    if random.random() > prob:
        return formula_str
    tokens = ["AND", "OR", "IMPLIES", "NOT", "FORALL", "EXISTS"]
    for _ in range(10):
        t = random.choice(tokens)
        if t in formula_str:
            repl = random.choice([x for x in tokens if x != t])
            return formula_str.replace(t, repl, 1)
    return f"NOT({formula_str})"

def evaluate_fitness(formula_str, ltn_dict, scope_vars):
    try:
        root = parse_formula(formula_str)
        ltn_form = build_ltn_formula_node(root, ltn_dict, scope_vars)
        base_fit = ltn_form.value.mean().item() if ltn_form.value.numel() > 1 else ltn_form.value.item()
        
        # Penalizza duplicati nei predicati
        preds = get_all_predicates(root)
        if len(preds) != len(set(preds)):
            base_fit *= 0.95
        
        # Penalizza se manca il pattern "Cat(x) IMPLIES HasWhiskers(x)"
        if not contains_implication(root, "Cat", "HasWhiskers"):
            base_fit *= 0.9
        
        # Penalizza se è presente il pattern incoerente "AND(Cat(x), Dog(x))"
        if contains_inconsistent_pattern(root):
            base_fit *= 0.8
        
        # Applica il coefficiente extra di penalizzazione
        penalty = penalty_coefficient(root)
        return base_fit * penalty
    except Exception as e:
        return 0.0

def get_all_predicates(nodo: Nodo) -> List[str]:
    results = []
    if nodo.tipo_nodo == "PREDICATO":
        results.append(nodo.valore)
    for child in nodo.figli:
        results.extend(get_all_predicates(child))
    return results

def get_neighbors(matrix, row, col):
    neighbors = []
    rows, cols, _ = matrix.shape
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < rows and 0 <= c < cols:
            neighbors.append(matrix[r, c])
    return neighbors

def compute_fitness(pop_matrix, ltn_dict, scope_vars):
    for i in range(pop_matrix.shape[0]):
        for j in range(pop_matrix.shape[1]):
            form_str = pop_matrix[i, j][0]
            pop_matrix[i, j][1] = evaluate_fitness(form_str, ltn_dict, scope_vars)
    return pop_matrix

def crossover_str(str1, str2, prob=0.8):
    if random.random() > prob:
        return str1, str2
    cut1a = random.randint(0, len(str1)-1)
    cut1b = random.randint(cut1a, len(str1))
    cut2a = random.randint(0, len(str2)-1)
    cut2b = random.randint(cut2a, len(str2))
    substr1 = str1[cut1a:cut1b]
    substr2 = str2[cut2a:cut2b]
    child1 = str1[:cut1a] + substr2 + str1[cut1b:]
    child2 = str2[:cut2a] + substr1 + str2[cut2b:]
    return child1, child2

def mutate_str(formula_str, prob=0.3):
    if random.random() > prob:
        return formula_str
    tokens = ["AND", "OR", "IMPLIES", "NOT", "FORALL", "EXISTS"]
    for _ in range(10):
        t = random.choice(tokens)
        if t in formula_str:
            repl = random.choice([x for x in tokens if x != t])
            return formula_str.replace(t, repl, 1)
    return f"NOT({formula_str})"

def evolutionary_run(pop_matrix, generations, ltn_dict, scope_vars):
    for gen in range(generations):
        print(f"\n--- Generazione {gen+1}/{generations} ---")
        rows, cols, _ = pop_matrix.shape
        for i in range(rows):
            for j in range(cols):
                neighbors = get_neighbors(pop_matrix, i, j)
                neighbors.sort(key=lambda x: x[1], reverse=True)
                parent1 = pop_matrix[i, j][0]
                parent2 = neighbors[0][0]  # Il migliore vicino
                c1, c2 = crossover_str(parent1, parent2, prob=0.9)
                c1 = mutate_str(c1, prob=0.2)
                c2 = mutate_str(c2, prob=0.2)
                f1 = evaluate_fitness(c1, ltn_dict, scope_vars)
                f2 = evaluate_fitness(c2, ltn_dict, scope_vars)
                best_formula, best_fit = (c1, f1) if f1 >= f2 else (c2, f2)
                if best_fit > pop_matrix[i, j][1]:
                    pop_matrix[i, j] = [best_formula, best_fit]
        all_indivs = [pop_matrix[r, c] for r in range(rows) for c in range(cols)]
        best_local = max(all_indivs, key=lambda x: x[1])
        print(f"Miglior formula locale: {best_local[0]} (fitness={best_local[1]:.4f})")
    return pop_matrix

# ====================================================
# 8) MAIN: INIZIALIZZAZIONE, TRAINING KB, EVOLUZIONE
# ====================================================

if __name__ == "__main__":
    # Step 1: Costruisco e alleno la KB
    kb_formulas = create_kb()
    print("==== TRAINING DELLA KB ====")
    ltn_dict_kb, vars_kb = setup_ltn(kb_formulas, epochs=200)
    
    # Step 2: Costruisco il dizionario per l'evoluzione (stessi predicati, quantificatori, operatori)
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
    ga_vars = {"x": vars_kb["x"]}
    
    # Step 3: Inizializzo la popolazione come matrice NxN (qui 49 individui)
    population_size = 49
    matrix_size = int(np.sqrt(population_size))
    pop_matrix = np.empty((matrix_size, matrix_size, 2), dtype=object)
    for i in range(matrix_size):
        for j in range(matrix_size):
            form_str = random_formula()
            pop_matrix[i, j] = [form_str, 0.0]
    
    pop_matrix = compute_fitness(pop_matrix, ga_ltn_dict, ga_vars)
    
    # Step 4: Evoluzione (qui, ad esempio, 200 generazioni)
    generations = 200
    final_pop = evolutionary_run(pop_matrix, generations, ga_ltn_dict, ga_vars)
    
    final_list = [final_pop[r, c] for r in range(final_pop.shape[0]) for c in range(final_pop.shape[1])]
    final_sorted = sorted(final_list, key=lambda x: x[1], reverse=True)
    
    print("\n=== RISULTATI FINALI ===")
    for rank, (fs, fitv) in enumerate(final_sorted[:5], start=1):
        print(f"Top {rank}: {fs} => {fitv:.4f}")
    best = final_sorted[0]
    print(f"\nMiglior individuo finale: {best[0]} (fitness={best[1]:.4f})")
