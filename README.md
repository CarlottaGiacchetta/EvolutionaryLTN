# Evolutionary Logic Tensor Network (LTN) Framework

This project implements a system based on **Genetic Algorithms (GA)** and **Genetic Programming (GP)** to optimize logical rules represented through **Logic Tensor Networks (LTN)**. The system combines machine learning methods with evolutionary techniques to improve the logical consistency of a knowledge base (KB).

---

## Project Structure

The project is organized into multiple files, each with a clear responsibility:

### 1. **`evo_funct.py`**
This file contains the core evolutionary functions:
- **Initial Population Generation**: Creates random logical trees representing logical formulas.
  - Main function: `popolazione_init`.
- **Fitness Calculation**: Evaluates the logical consistency of each individual.
  - Key functions: `compute_fitness`, `compute_fitness_singolo`, `compute_fitness_retrain`.
- **Selection**: Mechanisms to select the best individuals based on their fitness.
  - Main functions: `fitness_proportionate_selection`, `fitness_proportionate_selection_modern`.
- **Crossover and Mutation**: Genetic operations to combine and modify logical trees.
  - Main functions: `crossover`, `mutate`.
- **Evolutionary Run**: Implements the full evolutionary loop for GP and GA.
  - Main functions: `evolutionary_run_GP`, `evolutionary_run_GA`, `evolutionary_run`.

### 2. **`kb.py`**
Handles the knowledge base (KB):
- **KB Creation**: Defines logical rules and facts.
  - Main function: `create_kb`.
- **Integration with LTN**: Trains the KB with logical rules and evaluates satisfaction.
  - Key functions: `setup_ltn`, `kb_loss`.
- **KB Satisfaction Measurement**: Evaluates the logical consistency of the rules.
  - Main function: `measure_kb_sat`.
- **Support for New Rules**: Checks and adds unique formulas to the KB.
  - Main functions: `make_new_rule`, `is_formula_in_kb`.

### 3. **`tree.py`**
Implements the structure of logical trees:
- **Node**: Represents a single logical unit (operator, predicate, variable, etc.).
  - Main class: `Nodo`.
- **Tree**: Represents a logical formula as a binary tree.
  - Main class: `Albero`.
- **LTN Conversion**: Transforms a logical tree into an LTN-compatible formula.
  - Main function: `to_ltn_formula`.
- **Support for Operations**: Includes operations such as copying, node replacement, and depth calculation.

### 4. **`parser.py`**
Parses logical predicates:
- Converts strings like `Dog(x, y)` into structured representations.
  - Main function: `parse_predicato`.

### 5. **`utils.py`**
Utility functions for processing logical trees and optimization:
- **Node Management**: Retrieval, replacement, and analysis of nodes in trees.
  - Key functions: `get_all_nodes`, `replace_node_in_tree`, `analizza_predicati`.
- **KB Evaluation**: Functions for partial training and satisfaction measurement.
  - Key functions: `partial_train`, `measure_kb_sat`.
- **Operators and Predicates**: Creates unary predicates and manages logical operators.

---

## How It Works

### 1. **Initial Population**
- The initial population is created as a list (or matrix) of random logical trees.
- Each tree represents a logical formula with:
  - **Quantifiers**: `FORALL`, `EXISTS`.
  - **Operators**: `AND`, `OR`, `NOT`, `IMPLIES`.
  - **Predicates**: e.g., `Dog(x)`, `Bird(x)`.

### 2. **Fitness Calculation**
- The fitness of each formula is calculated based on:
  - **Logical Consistency**: How well the formula aligns with the KB's rules and facts.
  - **Penalties**: Deductions for complexity, tautologies, or predicate repetition.

### 3. **Crossover and Mutation**
- **Crossover**: Combines two logical trees by swapping substructures to generate new individuals.
- **Mutation**: Randomly modifies nodes (e.g., changing predicates, operators, or adding new nodes).

### 4. **Evolutionary Cycle**
- The evolutionary cycle iteratively applies selection, crossover, mutation, and fitness calculation.
- The best individuals are selected for the next generation.

### 5. **Integration with KB**
- Each generation is evaluated against the knowledge base.
- High-fitness formulas can be added to the KB to improve overall logical consistency.

---

## Usage

1. **Setup the KB**
   - Define predicates, operators, and constants in `kb.py`.

2. **Run the Evolutionary Algorithm**
   - Start the evolutionary cycle using the `evolutionary_run` or `evolutionary_run_GP` functions in `evo_funct.py`.

3. **Evaluate Results**
   - Check KB satisfaction using `measure_kb_sat` and review the new discovered rules.

---

## Technologies Used
- **Python**: Main programming language.
- **PyTorch**: Used for modeling predicates and optimizing the KB.
- **Genetic Algorithms and Genetic Programming**: To explore the space of logical formulas.

---

## Contributors
- This framework is designed for researchers and developers interested in combining machine learning with formal logic.

Feel free to reach out with questions or suggestions! 🚀

