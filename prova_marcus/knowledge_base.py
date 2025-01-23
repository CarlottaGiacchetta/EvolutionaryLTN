import ltn
import torch
import ltn.fuzzy_ops as fuzzy_ops
import torch.nn as nn

# Definizione della KB
def create_kb(predicati, quantificatori, operatori, costanti):
    
    # Definizione delle formule secondo le regole del paper
    
    # Regola 2: ∀x (Animal(x) → ¬Fly(x)) – "Gli animali non possono di solito volare"
    def formula_animal_implies_not_fly(x):
        return operatori["OR"](operatori["AND"](predicati["Animal"](x), predicati["Fly"](x)), operatori["AND"](predicati["Animal"](x), operatori["NOT"](predicati["Fly"](x))))
    
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
    

    #def gugu(x):
    #    return operatori["IMPLIES"](predicati["Penguin"](x), predicati["Fly"](x))
    
    #test_formula = lambda x : quantificatori["EXISTS"](x, gugu(x))
    # Definizione delle formule da includere nella KB (regole)
    kb_rules = [
        lambda x: quantificatori["FORALL"](x, formula_animal_implies_not_fly(x)),
        lambda x: quantificatori["FORALL"](x, formula_bird_implies_fly(x)),        
        lambda x: quantificatori["FORALL"](x, formula_penguin_implies_not_fly(x)),
        lambda x: quantificatori["FORALL"](x, formula_bird_implies_animal(x)),
        lambda x: quantificatori["FORALL"](x, formula_penguin_implies_bird(x)),
        lambda x: quantificatori["FORALL"](x, formula_swallow_implies_bird(x)),
        #lambda x : quantificatori["EXISTS"](x, gugu(x))
    ]
    
    
    # Definizione delle istanze (fatti)
    kb_facts = []
    
    #for predicato in predicati:
    #    kb_facts.append(lambda c: predicati[predicato](c))

    facts_mapping = {
        "Marcus": ["Swallow"],    # "Marcus è una rondine"
        "Pingu": ["Penguin"],    # "Tweety è un pinguino"
        #"Hitler" : ["Animal"]
    }

    kb_facts = []
    for costante, predicati_veri in facts_mapping.items():
        for predicato in predicati_veri:
            kb_facts.append(lambda c=costanti[costante], p=predicato: predicati[p](c))
    
    return kb_rules, kb_facts


# Setup LTN
def setup_ltn(costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts):

    # Ottimizzatore e parametri modello
    parameters = []
    for pred in predicati.values():
        parameters += list(pred.model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # Training Loop
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 1 - kb_loss(kb_rules, kb_facts, variabili, costanti)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item()}")

    # Valutazione Finale
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
    
    #def gugu(x):
    #    return operatori["IMPLIES"](predicati["Penguin"](x), predicati["Fly"](x))
    #
    #test_formula = lambda x : quantificatori["EXISTS"](x, gugu(x))

    #print(test_formula)
    #for chiave in variabili:
    #    print(test_formula(variabili[chiave]).value)
    #exit()

    return optimizer, costanti, predicati, quantificatori, operatori, variabili, kb_rules, kb_facts


# Loss Function
def kb_loss(kb_rules, kb_facts, variabili, costanti):
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
    print('stampo nodo 1 e nodo 2')
    print(node1)
    print(node2)
    """
    Confronta due nodi ricorsivamente (struttura, tipo e valori).
    """
    if node1.tipo_nodo != node2.tipo_nodo:
        print('node1.tipo_nodo != node2.tipo_nodo')
        print(node1.tipo_nodo, node2.tipo_nodo)
        return False
    if node1.valore != node2.valore:
        print('node1.valore != node2.valore')
        print(node1.valore, node2.valore)
        return False
    if len(node1.figli) != len(node2.figli):
        print('len(node1.figli) != len(node2.figli)')
        print(len(node1.figli), len(node2.figli))
        return False
    for child1, child2 in zip(node1.figli, node2.figli):
        print('sono nel for -> chiamata ricorsiva')
        if not compare_nodes(child1, child2):
            return False
    return True



def is_formula_in_kb(formula, kb_formulas):
    """
    Verifica se una formula semanticamente equivalente è già presente nella KB.
    """
    for kb_formula in kb_formulas:
        print()
        print('stampo kb formula')
        print(kb_formula)
        if compare_nodes(formula, kb_formula):
            return True
        print()
    return False
