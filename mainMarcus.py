import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)
ltn.device = torch.device("cpu")  # o "cuda" se disponibile

# Creiamo embedding casuali per i 3 individui
Fluffy_tensor = torch.randn(2)  
Garfield_tensor = torch.randn(2)
Rex_tensor = torch.randn(2)

# Definiamo costanti LTN:
Fluffy = ltn.core.Constant(Fluffy_tensor, trainable=False)
Garfield = ltn.core.Constant(Garfield_tensor, trainable=False)
Rex = ltn.core.Constant(Rex_tensor, trainable=False)

# Creiamo una variabile "x" che enumeri i 3 individui
# In LTNtorch, se voglio collezionare i 3 in un'unica Variabile, posso fare:
all_inds = torch.stack([Fluffy.value, Garfield.value, Rex.value], dim=0)  # shape [3,2]
x = ltn.core.Variable("x", all_inds, add_batch_dim=False)


# Esempio di MLP piccolo per un predicato unario
def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
    return nn.Sequential(
        nn.Linear(in_features, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid()  # output in [0,1]
    )

Cat_model = make_unary_predicate() # Cat(x) = [0,1]
Dog_model = make_unary_predicate()  #Dog(x) = [0,1]
HasWhiskers_model = make_unary_predicate()  #HasWhiskers(x) = [0,1]

Cat = ltn.core.Predicate(model=Cat_model)
Dog = ltn.core.Predicate(model=Dog_model)
HasWhiskers = ltn.core.Predicate(model=HasWhiskers_model)


impl = ltn.core.Connective(fuzzy_ops.ImpliesLuk())  # (p->q) = min(1, 1-p + q)

# formula 1: Cat(x) -> HasWhiskers(x)
# in LTNtorch: 
#    f1(x) = implies( Cat(x), HasWhiskers(x) )
def formula_cat_implies_whiskers(x):
    return impl(Cat(x), HasWhiskers(x))

# formula 2: Dog(x) -> not Cat(x)
not_ = ltn.core.Connective(fuzzy_ops.NotStandard())  # negazione
def formula_dog_implies_not_cat(x):
    return impl(Dog(x), not_(Cat(x)))


def fact_is_cat(c):
    return Cat(c)  # LTNObject in [0,1]

def fact_is_dog(c):
    return Dog(c)

def fact_has_whiskers(c):
    return HasWhiskers(c)


Forall = ltn.core.Quantifier(
    agg_op=fuzzy_ops.AggregPMeanError(p=2),  # universal quantifier
    quantifier='f'  # 'f' = forall
)

Exists = ltn.core.Quantifier(
    agg_op=fuzzy_ops.AggregPMean(p=2),       # existential quantifier
    quantifier='e'  # 'e' = exists
)


f1 = Forall(
    x,
    formula_cat_implies_whiskers(x)  # LTNObject result
)
print(f1.value)


params = list(Cat_model.parameters()) + list(Dog_model.parameters()) + list(HasWhiskers_model.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

def satisfaction_kb():
    # formula1 e formula2
    sat_f1 = Forall(x, formula_cat_implies_whiskers(x)).value  # scalare
    sat_f2 = Forall(x, formula_dog_implies_not_cat(x)).value   # scalare
    
    # fatti
    sat_fluffy_cat = fact_is_cat(Fluffy).value      # scalar
    sat_garfield_cat = fact_is_cat(Garfield).value
    sat_rex_dog = fact_is_dog(Rex).value
    sat_fluffy_whisk = fact_has_whiskers(Fluffy).value
    sat_garfield_whisk = fact_has_whiskers(Garfield).value
    
    all_vals = torch.stack([
        sat_f1, sat_f2,
        sat_fluffy_cat, sat_garfield_cat,
        sat_rex_dog, sat_fluffy_whisk, sat_garfield_whisk
    ])
    return torch.mean(all_vals)  # aggregator finale

# Ciclo di training
n_epochs = 200
for epoch in range(n_epochs):
    optimizer.zero_grad()
    sat = satisfaction_kb()
    loss = 1.0 - sat
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: satisfaction={sat.item():.3f} loss={loss.item():.3f}")

AND_ = ltn.core.Connective(fuzzy_ops.AndProd())     # e.g. Goguen T-norm
NOT_ = ltn.core.Connective(fuzzy_ops.NotStandard())  # negazione
Exists_ = ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')

def formula_exists_cat_no_whiskers(x):
    return AND_( Cat(x), NOT_(HasWhiskers(x)) )
    # LTNObject in [0,1]

new_formula_obj = Exists_(x, formula_exists_cat_no_whiskers(x))


with torch.no_grad():
    val_new_form = new_formula_obj.value  # se interpretato con i parametri attuali
print("Valore di verità di ∃x (Cat(x) & ¬HasWhiskers(x)):", val_new_form.item())


def check_formula_with_retraining(new_formula, steps=50):
    # Salva lo stato attuale
    old_state = [p.clone().detach() for p in params]
    
    # Ottimizziamo qualche step per massimizzare la media tra KB e new_formula
    opt = torch.optim.Adam(params, lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        sat_kb = satisfaction_kb()
        sat_new = new_formula.value  # calcolato con i parametri correnti
        total_sat = (sat_kb + sat_new) / 2.0  # aggregator semplificato
        loss = 1.0 - total_sat
        loss.backward(retain_graph=True)
        opt.step()
    
    final_sat = total_sat.item()
    # Ripristina i parametri
    with torch.no_grad():
        for p, oldp in zip(params, old_state):
            p.copy_(oldp)
    return final_sat

new_formula_sat = check_formula_with_retraining(new_formula_obj, steps=50)
print("Soddisfazione (KB + new formula) dopo re-training:", new_formula_sat)



"""
possible_formulas = [
  "exists (Cat(x) & ~HasWhiskers(x))",
  "forall (Dog(x) -> Cat(x))",
  # ... e altre ...
]

def parse_formula(f_str):
    # Dato un "f_str", restituisci la LTNObject corrispondente
    # Esempio minimal: 
    if f_str == "exists (Cat(x) & ~HasWhiskers(x))":
        return Exists_(
            x,
            AND_(Cat(x), NOT_(HasWhiskers(x)))
        )
    # ...
    # in un GP esteso, avresti un parser generico
    return None

pop_size = 5
population = [np.random.choice(possible_formulas) for _ in range(pop_size)]

for gen in range(3):
    scored_pop = []
    for f_str in population:
        form_obj = parse_formula(f_str)
        final_sat = check_formula_with_retraining(form_obj)
        # Se vogliamo "bucare" la KB => fitness = (1 - final_sat)
        fit = 1.0 - final_sat
        scored_pop.append((f_str, fit))
    
    # ordiniamo
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    best_f, best_fit = scored_pop[0]
    print(f"Gen {gen}, best formula={best_f} (fitness={best_fit:.3f})")

    # ... (crossover, mutazioni, ecc.) ...
    # per demo, sostituiamo la pop con la best formula
    for i in range(len(population)):
        population[i] = best_f


"""