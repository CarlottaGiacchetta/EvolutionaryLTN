import random
from kb import * 
from structure import *
from utils import *

# Convert Albero tree to string representation
def albero_to_string(albero):
    """
    Convert an Albero object (logical tree) into a string representation.
    """
    return str(albero.radice)

# Crossover operation for logical formula strings
def crossover_string(str1, str2):
    """
    Perform a crossover operation on two logical formula strings.
    The crossover will randomly combine parts of the two strings.
    """
    parts1 = str1.split()
    parts2 = str2.split()
    
    crossover_point1 = random.randint(0, len(parts1) - 1)
    crossover_point2 = random.randint(0, len(parts2) - 1)
    
    # Swap parts after crossover points
    new_str1 = " ".join(parts1[:crossover_point1] + parts2[crossover_point2:])
    new_str2 = " ".join(parts2[:crossover_point2] + parts1[crossover_point1:])
    
    return new_str1, new_str2

# Mutation operation for logical formula strings
def mutate_string(formula_str):
    """
    Perform a mutation on a logical formula string.
    This could involve changing a predicate, operator, or variable.
    """
    words = formula_str.split()
    mutation_point = random.randint(0, len(words) - 1)
    mutated_word = random.choice(["Cat", "Dog", "HasWhiskers", "AND", "OR", "NOT", "IMPLIES"])
    
    words[mutation_point] = mutated_word
    return " ".join(words)

# Compute the fitness for a string-based logical formula
def compute_fitness_string(formula_str, ltn_dict, variabili, kb_formulas):
    """
    Compute the fitness of a logical formula string.
    """
    try:
        formula = Nodo("PREDICATO", formula_str)  # Dummy placeholder
        fitness = compute_fitness_singolo(formula, ltn_dict, variabili, kb_formulas)
        return fitness
    except:
        return 0

# Main Genetic Algorithm Loop with string-based individuals
def genetic_algorithm_with_strings(population_size, generations, num_offspring, ltn_dict, variabili, predicati, quantificatori, operatori, kb_formulas, metodo):
    """
    Perform the genetic algorithm using formula strings instead of tree structures.
    """
    # Initialize Population
    popolazione = popolazione_init(
        population_size=population_size, 
        is_matrix=False, 
        predicati=predicati, 
        quantificatori=quantificatori, 
        operatori=operatori, 
        variabili=variabili
    )

    # Convert individuals to string representation for GA operations
    for i in range(len(popolazione)):
        individuo = popolazione[i][0]
        individuo_str = albero_to_string(individuo)
        popolazione[i][0] = individuo_str  # Replace with string representation

    for generation in range(generations):
        print(f"--- Generazione {generation + 1}/{generations} ---")
        
        # Selection: Select parents based on fitness
        selected_parents = metodo(popolazione, num_to_select=2)

        # Crossover: Produce offspring by combining parents
        child1_str, child2_str = crossover_string(selected_parents[0][0], selected_parents[1][0])

        # Mutation: Apply mutation to offspring
        child1_str = mutate_string(child1_str)
        child2_str = mutate_string(child2_str)

        # Evaluate fitness of the offspring
        fit_child1 = compute_fitness_string(child1_str, ltn_dict, variabili, kb_formulas)
        fit_child2 = compute_fitness_string(child2_str, ltn_dict, variabili, kb_formulas)

        # Select the best offspring to replace old population
        new_individuals = [(child1_str, fit_child1), (child2_str, fit_child2)]

        # Update population (elitism strategy, keeping the best individuals)
        population_with_new_offspring = sorted(popolazione + new_individuals, key=lambda x: x[1], reverse=True)
        popolazione = population_with_new_offspring[:population_size]  # Keep only the best individuals

        # Print current best individual in the population
        best_individual = popolazione[0]
        print(f"Best individual in generation {generation + 1}: {best_individual[0]} with fitness: {best_individual[1]}")

    return popolazione


# Define the parameters for the Genetic Algorithm
population_size = 50  # Set the population size
generations = 100     # Set the number of generations
num_offspring = 5     # Set the number of offspring per generation

# Define the selection method (you can choose from your available methods)
metodo = fitness_proportionate_selection  # Use fitness_proportionate_selection for selection

# Initialize constants, predicates, quantifiers, and operators
costanti = {
    "Fluffy": ltn.core.Constant(torch.randn(2), trainable=False),
    "Garfield": ltn.core.Constant(torch.randn(2), trainable=False),
    "Rex": ltn.core.Constant(torch.randn(2), trainable=False)
}

tmp_x = torch.stack([costanti[i].value for i in costanti.keys()], dim=0)

# Predicates
predicati = {
    "Cat": ltn.core.Predicate(model=make_unary_predicate()),
    "Dog": ltn.core.Predicate(model=make_unary_predicate()),
    "HasWhiskers": ltn.core.Predicate(model=make_unary_predicate()),
}

# Quantifiers
quantificatori = {
    "FORALL": ltn.core.Quantifier(fuzzy_ops.AggregPMeanError(p=2), quantifier='f'),
    "EXISTS": ltn.core.Quantifier(fuzzy_ops.AggregPMean(p=2), quantifier='e')
}

# Operators
operatori = {
    "AND": ltn.core.Connective(fuzzy_ops.AndProd()),
    "OR": ltn.core.Connective(fuzzy_ops.OrMax()),
    "IMPLIES": ltn.core.Connective(fuzzy_ops.ImpliesLuk()),
    "NOT": ltn.core.Connective(fuzzy_ops.NotStandard()),
}

# Variables
variabili = {
    "x": ltn.core.Variable("x", tmp_x, add_batch_dim=False),
}

# Knowledge Base formulas
kb_formulas = create_kb()  # This loads the KB from the provided 'kb.py'

# Esecuzione
popolazione_finale = genetic_algorithm_with_strings(
    population_size=population_size,
    generations=generations,
    num_offspring=num_offspring,
    ltn_dict={**predicati, **quantificatori, **operatori},
    variabili=variabili,
    predicati=predicati,
    quantificatori=quantificatori,
    operatori=operatori,
    kb_formulas=kb_formulas,
    metodo=metodo
)

# Ordinamento della popolazione in base alla fitness in ordine decrescente
popolazione_ordinata = sorted(
    popolazione_finale,
    key=lambda x: x[1],  # Ordina per fitness
    reverse=True  # Ordine decrescente
)

# Miglior individuo finale
migliori = popolazione_ordinata[:5]
print('Popolazione finale:')
print(popolazione_finale)

print('Popolazione ordinata:')
print(popolazione_ordinata)

for migliore in migliori:
    print(f"\n--- Risultati Finali ---")
    print(f"Miglior individuo finale: {migliore[0]}")
    print(f"Fitness: {migliore[1]}")