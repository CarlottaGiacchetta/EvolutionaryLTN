import random
import copy
import torch.nn as nn
from parser import *

#################################################################
# Definizione del Nodo
#################################################################

class Nodo:
    def __init__(self, tipo_nodo, valore=None, figli=None):
        '''
        Represents a node in a logical tree.

        Parameters:
        - tipo_nodo (str): Type of the node. Possible values:
          - "OPERATORE" (e.g., "AND", "OR", "NOT")
          - "QUANTIFICATORE" (e.g., "FORALL", "EXISTS")
          - "PREDICATO" (e.g., "Dog(x, y)", "Parent(x)")
          - "VARIABILE" (e.g., "x", "y")
        - valore (str): The value associated with the node (e.g., "AND", "Dog(x)").
        - figli (list): A list of child nodes (empty for predicates and variables).
        '''
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if (figli is not None) else []

    def copia(self):
        '''
        Creates a deep copy of the node and its children.

        Returns:
        - Nodo: A deep copy of the node.
        '''
        return copy.deepcopy(self)
    
    def __repr__(self):
        '''
        Provides a recursive string representation of the node.

        Returns:
        - str: The textual representation of the node.
        '''
        if self.tipo_nodo == "PREDICATO":
            # es: "Dog(x,y)"
            return f"{self.valore}"
        elif self.tipo_nodo == "VARIABILE":
            # es: "x"
            return self.valore
        elif self.tipo_nodo == "OPERATORE":
    
            if self.valore == "NOT":
                return f"(NOT {self.figli[0]})"
            else:
                return f"({self.figli[0]} {self.valore} {self.figli[1]})"
        elif self.tipo_nodo == "QUANTIFICATORE":
            var_node = self.figli[0]   # nodo VARIABILE
            body = self.figli[1]       # sottoformula
            return f"({self.valore} {var_node.valore}: {body})"
        else:
            return f"UNKNOWN({self.valore})"

    def valida_nodo(self):
        '''
        Validates the node based on its type and the number of children.

        Returns:
        - bool: True if the node is valid, False otherwise.
        '''
        if self.tipo_nodo == "OPERATORE":
            if self.valore == "NOT":
                if len(self.figli) != 1:
                    print("OPERATORE NOT deve avere 1 figlio, trovato", len(self.figli))
                    return False
            elif self.valore in self.OPERATORS:
                if len(self.figli) != 2:
                    print(f"OPERATORE {self.valore} deve avere 2 figli (binario).")
                    return False

        elif self.tipo_nodo == "QUANTIFICATORE":
            if len(self.figli) != 2:
                print("QUANTIFICATORE deve avere 2 figli (VARIABILE, body).")
                return False
            if self.figli[0].tipo_nodo != "VARIABILE":
                print("QUANTIFICATORE: primo figlio deve essere VARIABILE.")
                return False

        elif self.tipo_nodo == "PREDICATO":
            if len(self.figli) != 0:
                print("PREDICATO deve avere 0 figli, trovato", len(self.figli))
                return False

        elif self.tipo_nodo == "VARIABILE":
            if len(self.figli) != 0:
                print("VARIABILE deve avere 0 figli.")
                return False

        return True
    
    def __eq__(self, other):
        '''
        Checks equality between two nodes.

        Parameters:
        - other (Nodo): The other node to compare.

        Returns:
        - bool: True if the nodes are equivalent, False otherwise.
        '''
        if not isinstance(other, Nodo):
            return False
        return (
            self.tipo_nodo == other.tipo_nodo and
            self.valore == other.valore and
            self.figli == other.figli
        )

    def __hash__(self):
        '''
        Generates a hash for the node, useful for usage in sets or dictionaries.

        Returns:
        - int: The hash of the node.
        '''
        return hash((self.tipo_nodo, self.valore, tuple(self.figli)))
    

#################################################################
# Classe Albero
#################################################################

class Albero:
    def __init__(self, VARIABLES, OPERATORS, QUANTIFIERS, PREDICATES):
        '''
        Constructs a random logical tree (Albero) using the provided variables, operators,
        quantifiers, and predicates.

        The tree structure follows the pattern:
        (QUANT var: (Pred(...var...) OP Pred(...var...)))

        Parameters:
        - VARIABLES (list): A list of variable names (e.g., ["x", "y"]).
        - OPERATORS (list): A list of logical operators (e.g., ["AND", "OR", "IMPLIES"]).
        - QUANTIFIERS (list): A list of quantifiers (e.g., ["FORALL", "EXISTS"]).
        - PREDICATES (list): A list of predicate names (e.g., ["Dog", "Cat"]).
        '''
        self.VARIABLES = VARIABLES
        self.OPERATORS = OPERATORS
        self.QUANTIFIERS = QUANTIFIERS
        self.PREDICATES = PREDICATES

        self.ultima_fitness = 0
        self.stagnazione = 0

        # Scegli una variabile
        var = random.choice(self.VARIABLES)
        var_node = Nodo("VARIABILE", var, [])

        # Crea un operatore binario i cui predicati usano "var"
        op = random.choice(self.OPERATORS)
        left_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        right_pred = Nodo("PREDICATO", f"{random.choice(self.PREDICATES)}({var})")
        operator_node = Nodo("OPERATORE", op, [left_pred, right_pred])

        # Quantificatore
        q = random.choice(self.QUANTIFIERS)
        self.radice = Nodo("QUANTIFICATORE", q, [var_node, operator_node])

        self.profondita = self.calcola_profondita(self.radice)

    def copia(self):
        '''
        Creates a deep copy of the Albero object.

        Returns:
        - Albero: A new Albero object identical to the original.
        '''
        nuovo = Albero.__new__(Albero)
        nuovo.VARIABLES = self.VARIABLES
        nuovo.OPERATORS = self.OPERATORS
        nuovo.QUANTIFIERS = self.QUANTIFIERS
        nuovo.PREDICATES = self.PREDICATES
        nuovo.radice = self.radice.copia()
        nuovo.profondita = self.profondita
        nuovo.ultima_fitness = self.ultima_fitness
        nuovo.stagnazione = self.stagnazione
        return nuovo

    def calcola_profondita(self, nodo):
        '''
        Recursively calculates the depth of the tree.

        Parameters:
        - nodo (Nodo): The root node of the tree or subtree.

        Returns:
        - int: The depth of the tree.
        '''
        if not nodo.figli:
            return 1
        return 1 + max(self.calcola_profondita(c) for c in nodo.figli)

    def valida_albero(self):
        '''
        Validates the entire tree by checking the validity of each node.

        Returns:
        - bool: True if the tree is valid, False otherwise.
        '''
        stack = [self.radice]
        albero_valido = True
        while stack:
            nodo = stack.pop()
            if not nodo.valida_nodo():
                albero_valido = False
                break
            stack.extend(nodo.figli)
        return albero_valido
    
    def __repr__(self):
        '''
        Provides a string representation of the logical tree.

        Returns:
        - str: A string representation of the tree.
        '''
        return str(self.radice)
    
    def __eq__(self, other):
        '''
        Checks equality between two Albero objects based on their root nodes.

        Parameters:
        - other (Albero): The other tree to compare.

        Returns:
        - bool: True if the trees are equal, False otherwise.
        '''
        if not isinstance(other, Albero):
            return False
        return self.radice == other.radice

    def __hash__(self):
        '''
        Generates a hash value for the tree, based on its root node.

        Returns:
        - int: The hash of the tree.
        '''
        return hash(self.radice)

    def to_ltn_formula(self, ltn_dict, scope_vars):
        '''
        Converts the logical tree into a Logic Tensor Network (LTN) formula.

        Parameters:
        - ltn_dict (dict): A dictionary of LTN predicates and operators.
        - scope_vars (list): A list of LTN variables used in the formula.

        Returns:
        - LTN formula: The LTN-compatible representation of the tree.
        '''
        return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)
        
    def update_liveness(self, fitness):
        '''
        Updates the fitness of the tree and checks for stagnation.

        If the fitness does not change for more than 5 consecutive updates,
        the tree is re-initialized.

        Parameters:
        - fitness (float): The new fitness value for the tree.
        '''
        if fitness == self.ultima_fitness:
            self.stagnazione += 1
            self.ultima_fitness = fitness
        
        if self.stagnazione > 5:
            print("---> restart")
            self.__init__(self.VARIABLES, self.OPERATORS, self.QUANTIFIERS, self.PREDICATES)
    
    def albero_to_string(albero):
        '''
        Converts an Albero object (logical tree) into its string representation.

        Parameters:
        - albero (Albero): The Albero object to convert.

        Returns:
        - str: The string representation of the tree.
        '''
        return str(albero.radice)


#################################################################
# Funzioni di building formula LTN da un albero
#################################################################

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
    '''
    Recursively builds an LTN (Logic Tensor Network) formula from a logical tree node.

    The function traverses the tree structure represented by `Nodo` and constructs
    the corresponding LTN formula based on the type of node (predicate, operator, quantifier, or variable).

    Parameters:
    - nodo (Nodo): The root node of the logical tree or subtree.
    - ltn_dict (dict): A dictionary mapping predicates, operators, and quantifiers
      to their corresponding LTN implementations.
    - scope_vars (dict): A dictionary mapping variable names to their LTN.Variable objects.

    Returns:
    - LTN formula: The constructed LTN-compatible formula for the given tree.

    Raises:
    - ValueError: If a predicate, variable, or quantifier is undefined in the given dictionaries.
    '''
    if nodo.tipo_nodo == "PREDICATO":
        pred_name, var_names = parse_predicato(nodo.valore)
        # Recupera l'oggetto LTN corrispondente
        if pred_name not in ltn_dict:
            raise ValueError(f"Predicato {pred_name} non definito in ltn_dict!")
        ltn_pred = ltn_dict[pred_name]

        # Converti i var_names in ltn.Variable
        ltn_vars = []
        for var_name in var_names:
            if var_name not in scope_vars:
                raise ValueError(f"Variabile {var_name} non presente in scope_vars!")
            ltn_vars.append(scope_vars[var_name])

        # Applica il predicato
        return ltn_pred(*ltn_vars)

    elif nodo.tipo_nodo == "VARIABILE":
        # In logica LTN potresti semplicemente restituire scope_vars[nodo.valore]
        var_name = nodo.valore
        if var_name not in scope_vars:
            raise ValueError(f"Variabile {var_name} non presente in scope_vars!")
        return scope_vars[var_name]

    elif nodo.tipo_nodo == "OPERATORE":
        if nodo.valore == "NOT":
            child_ltn = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            return ltn_dict["NOT"](child_ltn)
        else:
            left_ltn = build_ltn_formula_node(nodo.figli[0], ltn_dict, scope_vars)
            right_ltn = build_ltn_formula_node(nodo.figli[1], ltn_dict, scope_vars)
            op_obj = ltn_dict[nodo.valore]
            return op_obj(left_ltn, right_ltn)

    elif nodo.tipo_nodo == "QUANTIFICATORE":
        quant_name = nodo.valore  # "FORALL" o "EXISTS"
        var_node = nodo.figli[0]  # VARIABILE
        body_node = nodo.figli[1]
        new_var_name = var_node.valore

        # scoping
        new_scope = scope_vars.copy()
        if new_var_name not in new_scope:
            raise ValueError(f"Variabile {new_var_name} non definita in scope_vars!")
        body_ltn = build_ltn_formula_node(body_node, ltn_dict, new_scope)
        quant_op = ltn_dict[quant_name]
        return quant_op(new_scope[new_var_name], body_ltn)

    else:
        raise ValueError(f"Nodo di tipo sconosciuto: {nodo.tipo_nodo}")