from classe import LTNConstants, LTNVariables, LTNPredicates, LTNFormulas, LTNQuantifiers, LTNTraining
# Initialize components
constants = LTNConstants()
variables = LTNVariables(constants)
predicates = LTNPredicates()
formulas = LTNFormulas(predicates)
quantifiers = LTNQuantifiers()

# Train the model
training = LTNTraining(constants, predicates, variables, formulas, quantifiers)
training.train()