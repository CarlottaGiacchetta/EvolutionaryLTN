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
OPERATORS = {0: "Implies", 1: "And", 2: "Or"}
NEGATION = {0: "Not", 1: ""}

def generate_formula(random, args):
    quantifier = random.choice([0, 1])  # 0: Forall, 1: Exists
    negation1 = random.choice([0, 1]) #0: False, 1: True
    predicate1 = random.choice([0, 1, 2])  # 0: Cat, 1: Dog, 2: HasWhiskers
    operator = random.choice([0, 1, 2])  # 0: Implies, 1: And, 2: Not
    negation2 = random.choice([0, 1]) #0: False, 1: True
    predicate2 = random.choice([0, 1, 2])  # 0: Cat, 1: Dog, 2: HasWhiskers
    return [quantifier, negation1, predicate1, operator, negation2, predicate2]

def evaluate_formula(candidates, args):
    training = args["training"]
    fitness = []
    for candidate in candidates:
        quantifier = getattr(args["quantifiers"], QUANTIFIERS[candidate[0]])
        pred1 = getattr(args["predicates"], PREDICATES[candidate[1]])
        negation = getattr(args["negations"], NEGATION[candidate[2]])
        operator_name = OPERATORS[candidate[3]]

        if negation == "Not":
            def formula_func():
                operator = ltn.core.Connective(fuzzy_ops.NotStandard())
                return quantifier(training.variables.x, negation(pred1(training.variables.x)))
        else:
            operator = ltn.core.Connective(getattr(fuzzy_ops, operator_name + "Luk")())
            pred2 = getattr(args["predicates"], PREDICATES[candidate[3]])

            def formula_func():
                return quantifier(training.variables.x, operator(pred1(training.variables.x), pred2(training.variables.x)))

        fitness.append(training.add_formula_temporarily(formula_func))
    return fitness

def evaluate_formula(candidates, args):
    training = args["training"]
    fitness = []
    for candidate in candidates:
        # Recupero componenti della formula
        quantifier = QUANTIFIERS[candidate[0]]
        pred1_name = PREDICATES[candidate[2]]
        operator_name = OPERATORS[candidate[3]]
        pred2_name = PREDICATES[candidate[5]]
        
        negation1 = candidate[1] == 0  # True se "Not" è applicato al primo predicato
        negation2 = candidate[4] == 0  # True se "Not" è applicato al secondo predicato

        # Costruzione della rappresentazione leggibile della formula
        pred1_repr = f"¬{pred1_name}" if negation1 else pred1_name
        pred2_repr = f"¬{pred2_name}" if negation2 else pred2_name
        formula_repr = f"{quantifier} x ({pred1_repr} {operator_name} {pred2_repr})"
        
        # Stampa la formula
        print(f"Evaluating formula: {formula_repr}")

        # Creazione della funzione della formula
        def formula_func():
            operator = ltn.core.Connective(getattr(fuzzy_ops, operator_name + "Luk")())

            # Applica la negazione al primo predicato, se necessario
            pred1 = getattr(args["predicates"], pred1_name)
            pred1_value = pred1(training.variables.x)
            if negation1:
                pred1_value = ltn.core.Connective(fuzzy_ops.NotStandard())(pred1_value)

            # Applica la negazione al secondo predicato, se necessario
            pred2 = getattr(args["predicates"], pred2_name)
            pred2_value = pred2(training.variables.x)
            if negation2:
                pred2_value = ltn.core.Connective(fuzzy_ops.NotStandard())(pred2_value)

            return getattr(args["quantifiers"], quantifier)(training.variables.x, operator(pred1_value, pred2_value))

        # Valutazione della formula e aggiunta del risultato alla lista di fitness
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
        pop_size=10,
        maximize=True,
        max_evaluations=10,
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
        quantifier = QUANTIFIERS[formula.candidate[0]]
        pred1_name = PREDICATES[formula.candidate[2]]
        operator_name = OPERATORS[formula.candidate[3]]
        pred2_name = PREDICATES[formula.candidate[5]]
        
        negation1 = formula.candidate[1] == 0  # True se "Not" è applicato al primo predicato
        negation2 = formula.candidate[4] == 0  # True se "Not" è applicato al secondo predicato

        # Costruzione della rappresentazione leggibile della formula
        pred1_repr = f"¬{pred1_name}" if negation1 else pred1_name
        pred2_repr = f"¬{pred2_name}" if negation2 else pred2_name
        formula_repr = f"{quantifier} x ({pred1_repr} {operator_name} {pred2_repr})"        
        print("Formula: ", formula_repr)
        print("Fitness: ", formula.fitness)
