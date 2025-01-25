import random
import copy
import torch.nn as nn
from parser import *

#################################################################
# Definizione del Nodo
#################################################################

class Nodo:
    def __init__(self, tipo_nodo, valore=None, figli=None):
        """
        tipo_nodo: "OPERATORE", "QUANTIFICATORE", "PREDICATO", "VARIABILE"
        valore: es. "AND", "FORALL", "Dog(x,y)", ...
        figli: lista di nodi (vuota per i predicati e le variabili)
        """
        self.tipo_nodo = tipo_nodo
        self.valore = valore
        self.figli = figli if (figli is not None) else []

    def copia(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        """
        Stampa ricorsiva del nodo.
        """
        if self.tipo_nodo == "PREDICATO":
            # es: "Dog(x,y)"
            return f"{self.valore}"
        elif self.tipo_nodo == "VARIABILE":
            # es: "x"
            return self.valore
        elif self.tipo_nodo == "OPERATORE":
            # assumiamo operatore binario o 'NOT' unario
            #print(self.figli)
            #print(self.valore)
            if self.valore == "NOT":
                return f"(NOT {self.figli[0]})"
            else:
                # es. (child0 AND child1)
                return f"({self.figli[0]} {self.valore} {self.figli[1]})"
        elif self.tipo_nodo == "QUANTIFICATORE":
            # (FORALL x: <body>)
            var_node = self.figli[0]   # nodo VARIABILE
            body = self.figli[1]       # sottoformula
            return f"({self.valore} {var_node.valore}: {body})"
        else:
            return f"UNKNOWN({self.valore})"

    def valida_nodo(self):
        """
        Controlli base sul numero di figli e coerenza con tipo_nodo.
        """
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
            # 2 figli: [nodo VARIABILE, sottoformula]
            if len(self.figli) != 2:
                print("QUANTIFICATORE deve avere 2 figli (VARIABILE, body).")
                return False
            if self.figli[0].tipo_nodo != "VARIABILE":
                print("QUANTIFICATORE: primo figlio deve essere VARIABILE.")
                return False

        elif self.tipo_nodo == "PREDICATO":
            # In questa rappresentazione, i predicati non hanno nodi figli (gli argomenti sono nel 'valore' string)
            if len(self.figli) != 0:
                print("PREDICATO deve avere 0 figli, trovato", len(self.figli))
                return False

        elif self.tipo_nodo == "VARIABILE":
            # Nessun figlio
            if len(self.figli) != 0:
                print("VARIABILE deve avere 0 figli.")
                return False

        return True
    
    def __eq__(self, other):
        if not isinstance(other, Nodo):
            return False
        return (
            self.tipo_nodo == other.tipo_nodo and
            self.valore == other.valore and
            self.figli == other.figli
        )

    def __hash__(self):
        return hash((self.tipo_nodo, self.valore, tuple(self.figli)))
    

#################################################################
# Classe Albero
#################################################################

class Albero:
    def __init__(self, VARIABLES, OPERATORS, QUANTIFIERS, PREDICATES):
        """
        Esempio di costruttore semplice: 
        (QUANT var: (Pred(...var...) OP Pred(...var...)))
        """
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
        if not nodo.figli:
            return 1
        return 1 + max(self.calcola_profondita(c) for c in nodo.figli)

    def valida_albero(self):
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
        return str(self.radice)
    
    def __eq__(self, other):
        if not isinstance(other, Albero):
            return False
        return self.radice == other.radice

    def __hash__(self):
        # Hash basato sulla radice dell'albero
        return hash(self.radice)

    def to_ltn_formula(self, ltn_dict, scope_vars):
        """
        Converte l'albero in formula LTN, 
        assumendo che scope_vars contenga le ltn.Variable pertinenti.
        """
        return build_ltn_formula_node(self.radice, ltn_dict, scope_vars)
        
    def update_liveness(self, fitness):
        if fitness == self.ultima_fitness:
            self.stagnazione += 1
            self.ultima_fitness = fitness
        
        if self.stagnazione > 5:
            print("---> restart")
            self.__init__(self.VARIABLES, self.OPERATORS, self.QUANTIFIERS, self.PREDICATES)
    
    def albero_to_string(albero):
        """
        Convert an Albero object (logical tree) into a string representation.
        """
        return str(albero.radice)


#################################################################
# Funzioni di building formula LTN da un albero
#################################################################

def build_ltn_formula_node(nodo: Nodo, ltn_dict, scope_vars):
    """
    Dato un nodo (PREDICATO / OPERATORE / QUANTIFICATORE / VARIABILE), 
    costruisce ricorsivamente la formula LTN corrispondente.
    """
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