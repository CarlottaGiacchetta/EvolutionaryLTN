import torch

# Definizione della KB
def create_kb(predicati, quantificatori, operatori, costanti):
    '''
    Creates a knowledge base (KB) with rules and facts based on the logical relationships 
    defined in the problem domain.

    The KB consists of:
    1. **Rules**: Logical formulas that define relationships between predicates.
    2. **Facts**: Specific instances that instantiate certain predicates for constants.

    Parameters:
    - predicati (dict): Dictionary mapping predicate names to their logical functions.
    - quantificatori (dict): Dictionary mapping quantifier names ("FORALL", "EXISTS") to their logical functions.
    - operatori (dict): Dictionary mapping operator names ("AND", "OR", "IMPLIES", "NOT") to their logical functions.
    - costanti (dict): Dictionary mapping constant names (e.g., "Marcus", "Tweety") to their corresponding objects.

    Returns:
    - kb_rules (list): List of lambda functions representing the logical rules in the KB.
    - kb_facts (list): List of lambda functions representing the instantiated facts in the KB.
    '''
    
    # Definizione delle formule secondo le regole del paper
    
    # Regola 2: ∀x (Animal(x) → ¬Fly(x)) – "Gli animali non possono di solito volare"
    #def formula_animal_implies_not_fly(x):
    #    return operatori["IMPLIES"](predicati["Animal"](x), operatori["NOT"](predicati["Fly"](x)))
    
    # Regola 3: ∀x (Bird(x) → Fly(x)) – "Gli uccelli possono volare"
    def formula_bird_implies_fly(x):
        return operatori["IMPLIES"](predicati["Bird"](x), predicati["Fly"](x))
    
    # Regola 4: ∀x (Penguin(x) → ¬Fly(x)) – "I pinguini non possono volare"
    def formula_penguin_implies_not_fly(x):
        return operatori["IMPLIES"](predicati["Penguin"](x), operatori["NOT"](predicati["Fly"](x)))
    
    # Regola 5: ∀x (Bird(x) → Animal(x)) – "Gli uccelli sono animali"
    def formula_bird_implies_animal(x):
        return operatori["IMPLIES"](predicati["Bird"](x), predicati["Animal"](x))
    
    # Regola 6: ∀x (Penguin(x) → Bird(x)) – "I pinguini sono uccelli"
    def formula_penguin_implies_bird(x):
        return operatori["IMPLIES"](predicati["Penguin"](x), predicati["Bird"](x))
    
    # Regola 7: ∀x (Swallow(x) → Bird(x)) – "Le rondini sono uccelli"
    def formula_swallow_implies_bird(x):
        return operatori["IMPLIES"](predicati["Swallow"](x), predicati["Bird"](x))
  
    kb_rules = [
        #lambda x: quantificatori["FORALL"](x, formula_animal_implies_not_fly(x)),
        lambda x: quantificatori["FORALL"](x, formula_bird_implies_fly(x)),        
        lambda x: quantificatori["FORALL"](x, formula_penguin_implies_not_fly(x)),
        lambda x: quantificatori["FORALL"](x, formula_bird_implies_animal(x)),
        lambda x: quantificatori["FORALL"](x, formula_penguin_implies_bird(x)),
        lambda x: quantificatori["FORALL"](x, formula_swallow_implies_bird(x)),
        #lambda x : quantificatori["EXISTS"](x, gugu(x))
    ]
    

    kb_facts = []

    facts_mapping = {
        "Marcus": ["Swallow"],    # "Marcus è una rondine"
        "Tweety": ["Penguin"]     # "Tweety è un pinguino"
        
    }

    kb_facts = []
    for costante, predicati_veri in facts_mapping.items():
        for predicato in predicati_veri:
            kb_facts.append(lambda c=costanti[costante], p=predicato: predicati[p](c))
    
    return kb_rules, kb_facts



def setup_ltn(costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts):
    '''
    Sets up and trains a Logic Tensor Network (LTN) using the provided knowledge base rules and facts.

    The function initializes an optimizer, performs a training loop to optimize the satisfaction of the KB,
    and evaluates the satisfaction of rules and facts at the end.

    Parameters:
    - costanti (dict): Dictionary of constants used in the KB, mapping names to values.
    - predicati (dict): Dictionary of predicates used in the KB, mapping names to functions/models.
    - quantificatori (dict): Dictionary of quantifiers ("FORALL", "EXISTS") used in the KB.
    - operatori (dict): Dictionary of logical operators ("AND", "OR", "IMPLIES", "NOT") used in the KB.
    - variabili (dict): Dictionary of variables used in the KB, mapping variable names to their values.
    - kb_rules (list): List of lambda functions representing the rules in the KB.
    - kb_facts (list): List of lambda functions representing the facts in the KB.

    Returns:
    - optimizer (torch.optim.Optimizer): The optimizer used to train the LTN.
    - costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts: The input parameters (unchanged).
    '''
    # Ottimizzatore e parametri modello
    parameters = []
    for pred in predicati.values():
        parameters += list(pred.model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # Training Loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 1 - kb_loss(kb_rules, kb_facts, variabili, costanti)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item()}")

    print("\n--- Valutazione Finale Regole ---")
    for chiave in variabili:
        for i, formula in enumerate(kb_rules, start=1):
            satisfaction = formula(variabili[chiave]).value
            print(f"Formula {i}: {satisfaction:.4f}")
    
    print("\n--- Valutazione Finale Fatti ---")
    for costante in costanti:
        for i, formula in enumerate(kb_facts, start=1):
            satisfaction = formula(costanti[costante]).value
            print(f"Formula {i}: {satisfaction:.4f}")

    return optimizer, costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts



def kb_loss(kb_rules, kb_facts, variabili, costanti):
    '''
    Computes the loss for the Knowledge Base (KB) as the average dissatisfaction 
    of all rules and facts. The goal is to minimize this loss during training.

    Parameters:
    - kb_rules (list): List of lambda functions representing the rules in the KB.
    - kb_facts (list): List of lambda functions representing the facts in the KB.
    - variabili (dict): Dictionary of variables used in the KB, mapping variable names to their values.
    - costanti (dict): Dictionary of constants used in the KB, mapping names to values.

    Returns:
    - loss (float): The average dissatisfaction of the KB.
    '''
    total_loss = 0
    for chiave in variabili:
        for formula in kb_rules:
            total_loss += formula(variabili[chiave]).value
    
    for costante in costanti:
        for fact in kb_facts:
            total_loss += fact(costanti[costante]).value

    loss = total_loss/((len(kb_rules) * len(variabili)) + (len(kb_facts) * len(costanti)))
    
    return loss


def compare_nodes(node1, node2):
    '''
    Recursively compares two nodes to check if they are structurally, semantically, 
    and hierarchically equivalent.

    Parameters:
    - node1 (Nodo): The first node to compare.
    - node2 (Nodo): The second node to compare.

    Returns:
    - bool: True if the nodes are equivalent, False otherwise.
    '''
    if node1.tipo_nodo != node2.tipo_nodo:
        return False
    if node1.valore != node2.valore:
        return False
    if len(node1.figli) != len(node2.figli):
        return False
    for child1, child2 in zip(node1.figli, node2.figli):
        if not compare_nodes(child1, child2):
            return False
    return True



def is_formula_in_kb(formula, kb_formulas):
    '''
    Checks if a given formula is already present in the knowledge base (KB),
    by comparing it with all existing formulas in the KB.

    Parameters:
    - formula (Nodo): The formula node to check.
    - kb_formulas (list): A list of existing formulas in the KB (each represented as a Nodo).

    Returns:
    - bool: True if a semantically equivalent formula is found, False otherwise.
    '''
    for kb_formula in kb_formulas:
        if compare_nodes(formula, kb_formula):
            return True
        
    return False
