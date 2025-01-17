import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)
ltn.device = torch.device("cpu")  # or "cuda" if available

class LTNConstants:
    def __init__(self):
        self.Fluffy_tensor = torch.randn(2)
        self.Garfield_tensor = torch.randn(2)
        self.Rex_tensor = torch.randn(2)

        print('--------------------- Initial Tensors ---------------------')
        print(self.Fluffy_tensor)
        print(self.Garfield_tensor)
        print(self.Rex_tensor)
        print('-------------------------------------------------------\n\n')

        self.Fluffy = ltn.core.Constant(self.Fluffy_tensor, trainable=False)
        self.Garfield = ltn.core.Constant(self.Garfield_tensor, trainable=False)
        self.Rex = ltn.core.Constant(self.Rex_tensor, trainable=False)

        print('--------------------- Initial Constants ---------------------')
        print(self.Fluffy)
        print(self.Garfield)
        print(self.Rex)
        print('-------------------------------------------------------\n\n')

class LTNVariables:
    def __init__(self, constants):
        self.all_inds = torch.stack([
            constants.Fluffy.value,
            constants.Garfield.value,
            constants.Rex.value
        ], dim=0)  # shape [3,2]
        self.x = ltn.core.Variable("x", self.all_inds, add_batch_dim=False)
        
        print('--------------------- x ---------------------')
        print(self.x)
        print(self.all_inds)
        print('-------------------------------------------------------\n\n')

class LTNPredicates:
    def __init__(self):
        self.Cat_model = self.make_unary_predicate()
        self.Dog_model = self.make_unary_predicate()
        self.HasWhiskers_model = self.make_unary_predicate()

        self.Cat = ltn.core.Predicate(model=self.Cat_model)
        self.Dog = ltn.core.Predicate(model=self.Dog_model)
        self.HasWhiskers = ltn.core.Predicate(model=self.HasWhiskers_model)

        print('--------------------- Predicates ---------------------')
        print(self.Cat)
        print(self.Dog)
        print(self.HasWhiskers)
        print('-------------------------------------------------------\n\n')

    @staticmethod
    def make_unary_predicate(in_features=2, hidden1=8, hidden2=4):
        return nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()  # output in [0,1]
        )

class LTNFormulas:
    def __init__(self, predicates):
        self.impl = ltn.core.Connective(fuzzy_ops.ImpliesLuk())
        self.not_ = ltn.core.Connective(fuzzy_ops.NotStandard())

        self.Cat = predicates.Cat
        self.Dog = predicates.Dog
        self.HasWhiskers = predicates.HasWhiskers
        self.formulas = {
            "Cat -> HasWhiskers": self.formula_cat_implies_whiskers,
            "Dog -> ¬Cat": self.formula_dog_implies_not_cat,
        }

    def formula_cat_implies_whiskers(self, x):
        return self.impl(self.Cat(x), self.HasWhiskers(x))

    def formula_dog_implies_not_cat(self, x):
        return self.impl(self.Dog(x), self.not_(self.Cat(x)))

    def fact_is_cat(self, c):
        return self.Cat(c)

    def fact_is_dog(self, c):
        return self.Dog(c)

    def fact_has_whiskers(self, c):
        return self.HasWhiskers(c)

class LTNQuantifiers:
    def __init__(self):
        self.Forall = ltn.core.Quantifier(
            agg_op=fuzzy_ops.AggregPMeanError(p=2),  # universal quantifier
            quantifier='f'
        )

        self.Exists = ltn.core.Quantifier(
            agg_op=fuzzy_ops.AggregPMean(p=2),       # existential quantifier
            quantifier='e'
        )

class LTNTraining:
    def __init__(self, constants, predicates, variables, formulas, quantifiers):
        self.constants = constants
        self.predicates = predicates
        self.variables = variables
        self.formulas = formulas
        self.quantifiers = quantifiers

        self.params = list(predicates.Cat_model.parameters()) + \
                      list(predicates.Dog_model.parameters()) + \
                      list(predicates.HasWhiskers_model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=0.01)

    def satisfaction_kb(self):
        x = self.variables.x

        # Formulas
        sat_f1 = self.quantifiers.Forall(x, self.formulas.formula_cat_implies_whiskers(x)).value
        sat_f2 = self.quantifiers.Forall(x, self.formulas.formula_dog_implies_not_cat(x)).value

        # Facts
        sat_fluffy_cat = self.formulas.fact_is_cat(self.constants.Fluffy).value
        sat_garfield_cat = self.formulas.fact_is_cat(self.constants.Garfield).value
        sat_rex_dog = self.formulas.fact_is_dog(self.constants.Rex).value

        all_vals = torch.stack([sat_f1, sat_f2, sat_fluffy_cat, sat_garfield_cat, sat_rex_dog])
        return torch.mean(all_vals)

    def train(self, n_epochs=200):
        print('---------------- Training Cycle ------------------')
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            sat = self.satisfaction_kb()
            loss = 1.0 - sat
            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: satisfaction={sat.item():.3f} loss={loss.item():.3f}")
        print('----------------------------------------------------\n\n')


    def add_formula_temporarily(self, new_formula_func, steps=50):
        """
        Add a formula temporarily to the KB, retrain, and evaluate it.

        Args:
            new_formula_func (callable): A function that creates and returns the new formula object.
            steps (int): Number of retraining steps.

        Returns:
            float: Final satisfaction value after retraining.
        """
        # Save the current state_dict
        original_state = {}
        for name, param in zip(['Cat', 'Dog', 'HasWhiskers'], [
            self.predicates.Cat_model, 
            self.predicates.Dog_model, 
            self.predicates.HasWhiskers_model
        ]):
            original_state[name] = deepcopy(param.state_dict())

        # Create a new optimizer for retraining
        optimizer_retrain = torch.optim.Adam(self.params, lr=0.01)

        final_sat = 0.0

        try:
            for step in range(steps):
                optimizer_retrain.zero_grad()
                # Compute satisfaction of the knowledge base
                sat_kb = self.satisfaction_kb()
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
        for name, param in zip(['Cat', 'Dog', 'HasWhiskers'], [
            self.predicates.Cat_model, 
            self.predicates.Dog_model, 
            self.predicates.HasWhiskers_model
        ]):
            param.load_state_dict(original_state[name])

        return final_sat
    
    def infer_formula(self, formula_func):
        """
        Evaluate the truth value of a formula without retraining.

        Args:
            formula_func (callable): A function that creates and returns the formula object.

        Returns:
            float: Truth value of the formula.
        """
        with torch.no_grad():
            formula_obj = formula_func()
            return formula_obj.value.item()
