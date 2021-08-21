# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

import difflib
import re
import numpy as np
import pandas as pd
import os

bank_of_expressions_path = "correction_expressions.csv"

### Automatic correction


def letter_swap_rules():
    """
    Define a subcategory of rules consisting of long-s replacement and other common mismatches.
    This rules must be absolute, meaning we should never add a mistake where there were not when we apply them.
    """
    
    letter_swap_dict = dict()
    
    f_to_s_dict = {"Préfident":"Président","Confeil":"Conseil","Juftice":"Justice","l'impreffion":"l'impression",
                   "réfolu":"résolu","meffage":"message","meflage":"message","indivifible":"indivisible","ci-deffus":
                  "ci-dessus","ci-deflus":"ci-dessus","fceau":"sceau","préfent":"présent","réfolution":"résolution",
                  "fource":"source","préfente":"présente","claffe":"classe","Laufanne":"Lausanne","Miniftre":"Ministre",
                  "befoin":"besoin","Comiffion":"Commission","ci-deflus":"ci-dessus","auffi":"aussi","prefcrite":"prescrite",
                  "boiffon":"boisson","boiffons":"boissons","néceffaire":"nécessaire","Affemblée":"Assemblée",
                   "meffages":"messages","Commiffion":"Commission","fignatures":"signatures","figné":"signé","fignés":"signés",
                  "fel":"sel","fignée":"signée","fcellé":"scellé","prifonnier":"prisonnier","prifonniers":"prisonniers",
                  "néceffité":"nécessité","Meffieurs":"Messieurs","préfervatif":"préservatif", "Confeils":"Conseils"}
    
    letter_swap_dict.update(f_to_s_dict)
    
    singleton_rules = {"€":"&","mililaires":"militaires","mililaire":"militaire"}
    letter_swap_dict.update(singleton_rules)
    
    return letter_swap_dict

def create_all_rules():
    """
    Create a dictionary with all replacement-rules for the automatic correction
    """
    
    all_rules = dict()
    
    l_to_one_rules_dict = {"l(\d+)":"1\\1","(\d+)l":"\\g<1>1","(\d+)l(\d+)":"\g<1>1\\2"}
    all_rules.update(l_to_one_rules_dict)
    
    I_to_one_rules_dict = {"I(\d+)":"1\\1","(\d+)I":"\\g<1>1","(\d+)I(\d+)":"\g<1>1\\2"}
    all_rules.update(I_to_one_rules_dict)
        
    one_to_l_rules_dict = {"([a-z]+)1([a-z]*)":"\\1l\\2","([A-Z])1([a-z]+)":"\\1l\\2"} # 1([a-z]+) excluded because eg. "1er"
    all_rules.update(one_to_l_rules_dict)
    
    one_to_I_rules_dict = {"([A-Z]+)1([A-Z]*)":"\\1I\\2"}
    all_rules.update(one_to_I_rules_dict)
    
    l_to_I_rules_dict = {"([A-Z]*)l([A-Z]+)":"\\1I\\2","([A-Z][A-Z]+)l([A-Z]*)":"\\1I\\2"}
    all_rules.update(l_to_I_rules_dict)
    
    I_to_l_rules_dict = {"([a-z]+)I([a-z]*)":"\\1l\\2"}
    all_rules.update(I_to_l_rules_dict)
    
    oh_to_zero_rules_dict = {"(\d+)o":"\g<1>0","(\d+)o(\d+)":"\g<1>0\\2","(\d+)O":"\g<1>0","(\d+)O(\d+)":"\g<1>0\\2"}
    all_rules.update(oh_to_zero_rules_dict)
    
    zero_to_oh_rules_dict = {"([A-Za-z]+)0":"\\1o","0([A-Za-z]+)":"o\\1","([A-Za-z]+)0([A-Za-z]+)":"\\1o\\2"}
    all_rules.update(zero_to_oh_rules_dict)
    
    oneth_special_dict = {"ler":"1er","Ier":"1er"}
    all_rules.update(oneth_special_dict)
    
    inverse_oneth_special_dict = {"1es":"les"}
    all_rules.update(inverse_oneth_special_dict)
    
    letter_swap_dict = letter_swap_rules()
    all_rules.update(letter_swap_dict)
    
    return all_rules


def automatic_correction(text,all_rules):
    """
    Perfom a simple automatic correction using universal regex rules.
    text : the text to be corrected
    Return : the corrected text and the indices of the places that have been corrected.
    
    Because we add \b before and after each rule, these rules can only correct words
    which have exactly one error in them. This to avoid inverted corrections if the
    order of the corrections is not the right one to treat a very degenerated case.
    
    ! Pour qu'on puisse retrouver les index des corrections faites,
    il est essentiel que les corrections proposées ne changent jamais la taille du texte
    (Ce qui devrait être le cas, comme on remplace généralement un caractère par un autre)
    """
    original_text = text    
    
    for regexpr in all_rules:
        text = re.sub("\\b"+regexpr+"\\b",all_rules.get(regexpr),text)
        
    corr_indices = np.nonzero(np.array(list(text)) != np.array(list(original_text)))[0].tolist()
    
    return text, corr_indices


### Manual correction

def find_all_corrected_exprs(raw_text,corrected_text):
    """
    Compare a raw text and its version after correction and return all words and expressions
    that have been corrected in it, in their corrected version.
    raw_text : the original text
    corrected_text : the text with corrections
    """
    
    corrected_expr = list()
    
    raw_l = raw_text.split()
    corr_l = corrected_text.split()
    
    sequence_matcher = difflib.SequenceMatcher(a=raw_l,b=corr_l)
    
    for opcode in sequence_matcher.get_opcodes():
        
        if opcode[0] == "replace" or opcode[0] == "insert":

            start = opcode[3]
            end = opcode[4]
             
            # Si un token du texte brut a été décomposé en plusieurs mots, on ajoute l'expression entière.
            # Idem si c'est une insertion ex nihilo
            if opcode[2]-opcode[1] == 1 or opcode[0] == "insert":
                raw_exp = strip_punctuation(" ".join(corr_l[start:end]))
                #print(raw_exp)
                corrected_expr.append(raw_exp)
            
            # Sinon, on ajoute les mots séparément.
            else:                
                for w in corr_l[start:end]:
                    raw_w = strip_punctuation(w)
                    #print(raw_w)
                    corrected_expr.append(raw_w)
    
    return corrected_expr


def open_expr_file(file_expr_path=bank_of_expressions_path):
    """
    Open the stored expressions file and convert its data to a set of expressions
    file_expr_path : the path to the file
    """
    
    if os.path.isfile(file_expr_path):
        expr_df = pd.read_csv(file_expr_path,sep = "\t",encoding="utf8", header=None, names = ["expr_col"])
    else:
        expr_df = pd.DataFrame([],columns = ["expr_col"])
    
    exprs = set(exp for exp in expr_df["expr_col"].unique() if not pd.isnull(exp))
    
    return exprs
    
def save_expr_file(exprs, file_expr_path=bank_of_expressions_path):
    """
    Save a set of expression in the stored expressions file
    file_expr_path : the path to the file
    exprs : the set of expressions
    """
    
    expr_df = pd.DataFrame(exprs, columns = ["expr_col"])
    
    expr_df.to_csv(file_expr_path,sep="\t",encoding="utf8",index=False, header=False)
    

def update_expr_file(new_exprs, file_expr_path=bank_of_expressions_path):
    """
    Update the stored expressions file by merging the given set of expressions with the one already in the file.
    new_exprs : the iterable of new expressions to add
    file_expr_path : the path to the file
    """
    
    previous_exprs = open_expr_file(file_expr_path)
    
    updated_exprs = previous_exprs.union(set(new_exprs))
    
    save_expr_file(updated_exprs, file_expr_path)
 

 ### Text utilities

basic_punctuation = [".",",",";",":","?","!"]

def strip_punctuation(word):
    """
    This function remove the punctuation at the beginning or end of the given word in an unified manner.
    word: a sequence of characters
    return : the sequence stripped of its basic punctuations.
    """
    
    return word.strip("".join(basic_punctuation)+" ")