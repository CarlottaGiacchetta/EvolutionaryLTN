from classe import LTNConstants, LTNVariables, LTNPredicates, LTNFormulas, LTNQuantifiers, LTNTraining
import torch
import ltn
import ltn.fuzzy_ops as fuzzy_ops

# Initialize components
constants = LTNConstants()
variables = LTNVariables(constants)
predicates = LTNPredicates()
formulas = LTNFormulas(predicates)
quantifiers = LTNQuantifiers()

# Train the model
training = LTNTraining(constants, predicates, variables, formulas, quantifiers)
training.train()

# Test temporary formula addition
def new_formula_func():
    and_ = ltn.core.Connective(fuzzy_ops.AndProd())
    not_ = ltn.core.Connective(fuzzy_ops.NotStandard())
    return quantifiers.Exists(variables.x, and_(predicates.Cat(variables.x), not_(predicates.HasWhiskers(variables.x))))

satisfaction = training.add_formula_temporarily(new_formula_func, steps=50)
print("Satisfaction (KB + new formula) after temporary addition:", satisfaction)

# Infer formula without retraining
truth_value = training.infer_formula(new_formula_func)
print("Truth value of the new formula without retraining:", truth_value)