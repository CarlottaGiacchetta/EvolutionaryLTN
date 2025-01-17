import ltn
import ltn.fuzzy_ops as fuzzy_ops
from inspyred import ec
from random import Random
from classe import LTNConstants, LTNVariables, LTNPredicates, LTNFormulas, LTNQuantifiers, LTNTraining

# Initialize components
constants = LTNConstants()
variables = LTNVariables(constants)
predicates = LTNPredicates()
formulas = LTNFormulas(predicates)
quantifiers = LTNQuantifiers()

# ------------------- GA Functions -------------------
QUANTIFIERS = {0: "Forall", 1: "Exists"}
PREDICATES = {0: "Cat", 1: "Dog", 2: "HasWhiskers"}
OPERATORS = {0: "Implies", 1: "And", 2: "Not"}

def generate_formula(random, args):
    quantifier = random.choice([0, 1])  # 0: Forall, 1: Exists
    predicate1 = random.choice([0, 1, 2])  # 0: Cat, 1: Dog, 2: HasWhiskers
    operator = random.choice([0, 1, 2])  # 0: Implies, 1: And, 2: Not
    predicate2 = random.choice([0, 1, 2])  # 0: Cat, 1: Dog, 2: HasWhiskers
    return [quantifier, predicate1, operator, predicate2]

def evaluate_formula(candidates, args):
    training = args["training"]
    fitness = []
    for candidate in candidates:
        quantifier = getattr(args["quantifiers"], QUANTIFIERS[candidate[0]])
        pred1 = getattr(args["predicates"], PREDICATES[candidate[1]])
        operator_name = OPERATORS[candidate[2]]

        if operator_name == "Not":
            def formula_func():
                operator = ltn.core.Connective(fuzzy_ops.NotStandard())
                return quantifier(training.variables.x, operator(pred1(training.variables.x)))
        else:
            operator = ltn.core.Connective(getattr(fuzzy_ops, operator_name + "Luk")())
            pred2 = getattr(args["predicates"], PREDICATES[candidate[3]])

            def formula_func():
                return quantifier(training.variables.x, operator(pred1(training.variables.x), pred2(training.variables.x)))

        fitness.append(training.add_formula_temporarily(formula_func))
    return fitness

def safe_gaussian_mutation(random, candidates, args):
    mean = 0
    stdev = 0.1
    mutants = []
    for candidate in candidates:
        mutant = candidate[:]
        for i in range(len(mutant)):
            if isinstance(mutant[i], int):  # Solo mutazione di valori numerici
                mutant[i] += int(random.gauss(mean, stdev))
                mutant[i] = max(0, min(mutant[i], 2))  # Mantieni nei limiti validi
        mutants.append(mutant)
    return mutants


def run_ga(training, predicates, quantifiers):
    prng = Random()
    ea = ec.GA(prng)
    ea.variator = [ec.variators.uniform_crossover, safe_gaussian_mutation]
    ea.replacer = ec.replacers.steady_state_replacement
    ea.terminator = ec.terminators.evaluation_termination

    final_pop = ea.evolve(
        generator=generate_formula,
        evaluator=evaluate_formula,
        pop_size=20,
        maximize=True,
        max_evaluations=50,
        training=training,
        predicates=predicates,
        quantifiers=quantifiers
    )
    return sorted(final_pop, key=lambda ind: ind.fitness, reverse=True)

# ------------------- Main -------------------
if __name__ == "__main__":
    constants = LTNConstants()
    variables = LTNVariables(constants)
    predicates = LTNPredicates()
    formulas = LTNFormulas(predicates)
    quantifiers = LTNQuantifiers()

    training = LTNTraining(constants, predicates, variables, formulas, quantifiers)
    training.train()

    best_formulas = run_ga(training, predicates, quantifiers)
    print("Top formulas found:")
    for formula in best_formulas[:5]:
        print("Formula:", formula.candidate, "Fitness:", formula.fitness)
