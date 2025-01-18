import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
import numpy as np
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)
ltn.device = torch.device("cpu")  # or "cuda" if available

# Create random embeddings for the 3 individuals
Fluffy_tensor = torch.randn(2)
Garfield_tensor = torch.randn(2)
Rex_tensor = torch.randn(2)

print('--------------------- Initial Tensors ---------------------')
print(Fluffy_tensor)
print(Garfield_tensor)
print(Rex_tensor)
print('-------------------------------------------------------\n\n')

# Define LTN constants
Fluffy = ltn.core.Constant(Fluffy_tensor, trainable=False)
Garfield = ltn.core.Constant(Garfield_tensor, trainable=False)
Rex = ltn.core.Constant(Rex_tensor, trainable=False)
print('--------------------- Initial Constants ---------------------')
print(Fluffy)
print(Garfield)
print(Rex)
print('-------------------------------------------------------\n\n')

# Create a variable "x" enumerating the 3 individuals
all_inds = torch.stack([Fluffy.value, Garfield.value, Rex.value], dim=0)  # shape [3,2]
x = ltn.core.Variable("x", all_inds, add_batch_dim=False)
print('--------------------- x ---------------------')
print(x)
print('-------------------------------------------------------\n\n')

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

Cat_model = make_unary_predicate()  # Cat(x) = [0,1]
Dog_model = make_unary_predicate()  # Dog(x) = [0,1]
HasWhiskers_model = make_unary_predicate()  # HasWhiskers(x) = [0,1]

Cat = ltn.core.Predicate(model=Cat_model)
Dog = ltn.core.Predicate(model=Dog_model)
HasWhiskers = ltn.core.Predicate(model=HasWhiskers_model)
print('--------------------- Predicates ---------------------')
print(Cat)
print(Dog)
print(HasWhiskers)
print('-------------------------------------------------------\n\n')

impl = ltn.core.Connective(fuzzy_ops.ImpliesLuk())  # (p->q) = min(1, 1-p + q)

# Formula 1: Cat(x) -> HasWhiskers(x)
def formula_cat_implies_whiskers(x):
    return impl(Cat(x), HasWhiskers(x))

# Formula 2: Dog(x) -> not Cat(x)
not_ = ltn.core.Connective(fuzzy_ops.NotStandard())  # negation
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
print('----------------------- Formula 1 -----------------')
print(f1.value)
print('-------------------------------------------------\n\n')

# Collect parameters from all predicate models
params = list(Cat_model.parameters()) + list(Dog_model.parameters()) + list(HasWhiskers_model.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

def satisfaction_kb():
    # formula1 and formula2
    sat_f1 = Forall(x, formula_cat_implies_whiskers(x)).value  # scalar
    sat_f2 = Forall(x, formula_dog_implies_not_cat(x)).value   # scalar
    
    # facts
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
    return torch.mean(all_vals)  # final aggregator

# Training Loop
print('---------------- Training Cycle ------------------')
n_epochs = 200
for epoch in range(n_epochs):
    optimizer.zero_grad()
    sat = satisfaction_kb()
    loss = 1.0 - sat
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: satisfaction={sat.item():.3f} loss={loss.item():.3f}")
print('----------------------------------------------------\n\n')

AND_ = ltn.core.Connective(fuzzy_ops.AndProd())     # e.g., Goguen T-norm
NOT_ = ltn.core.Connective(fuzzy_ops.NotStandard())  # negation
Exists_ = ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')

def formula_exists_cat_no_whiskers(x):
    return AND_(Cat(x), NOT_(HasWhiskers(x)))
    # LTNObject in [0,1]

new_formula_obj = Exists_(x, formula_exists_cat_no_whiskers(x))

with torch.no_grad():
    val_new_form = new_formula_obj.value  # interpreted with current parameters
print("Truth value of ∃x (Cat(x) & ¬HasWhiskers(x)):", val_new_form.item())

def check_formula_with_retraining(new_formula_func, steps=50):
    """
    Retrains the model with an additional formula.

    Args:
        new_formula_func (callable): A function that creates and returns the new formula object.
        steps (int): Number of retraining steps.

    Returns:
        float: Final satisfaction value after retraining.
    """
    # Save the current state_dict
    original_state = {}
    for name, param in zip(['Cat', 'Dog', 'HasWhiskers'], [Cat_model, Dog_model, HasWhiskers_model]):
        original_state[name] = deepcopy(param.state_dict())

    # Create a new optimizer for retraining
    optimizer_retrain = torch.optim.Adam(params, lr=0.01)

    final_sat = 0.0

    try:
        for step in range(steps):
            optimizer_retrain.zero_grad()
            
            # Compute satisfaction of the knowledge base
            sat_kb = satisfaction_kb()

            # Recreate the new formula object to ensure a fresh computation graph
            new_formula_obj = new_formula_func()
            
            # Compute satisfaction of the new formula
            sat_new = new_formula_obj.value

            # Aggregate satisfactions
            total_sat = (sat_kb + sat_new) / 2.0

            # Define loss
            loss = 1.0 - total_sat

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer_retrain.step()

            final_sat = total_sat.item()

            # Optionally, print progress
            if (step + 1) % 10 == 0:
                print(f"Retraining Step {step + 1}/{steps}: satisfaction={final_sat:.3f}, loss={loss.item():.3f}")

    except RuntimeError as e:
        print(f"Error during retraining: {e}")

    # Restore the original state_dict
    for name, param in zip(['Cat', 'Dog', 'HasWhiskers'], [Cat_model, Dog_model, HasWhiskers_model]):
        param.load_state_dict(original_state[name])

    return final_sat

def create_new_formula():
    return Exists_(x, formula_exists_cat_no_whiskers(x))

new_formula_sat = check_formula_with_retraining(create_new_formula, steps=50)

print("Satisfaction (KB + new formula) after re-training:", new_formula_sat)
