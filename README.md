# Evolutionary Logic Tensor Network (LTN) Framework

This project implements a system based on **Genetic Algorithms (GA)** and **Genetic Programming (GP)** to optimize logical rules represented through **Logic Tensor Networks (LTN)**. The system throught evolutionary techniques improves the logical consistency of a knowledge base (KB).

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


