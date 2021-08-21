#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Ce notebook n'a besoin d'être lancé qu'une fois. Il uniformise certaines conventions des textes corrigés.
#  Réduire les lignes vides successives à une ligne vide maximum
#  
# * Eventuellement certaines corrections sur les s longs, les t <-> l (Elat Etat ?) etc
# * Formatter les apostrophes
# * Transformer les lettres doubles ae oe en leur équivalent en lettres simples.
# * harmoniser le début et la fin des lignes (strip des espaces, des caractères impossibles, etc, pour que chaque ligne se termine par un \n juste après la dernière lettre/chiffre/&/...)
# * Collapse les retours à la ligne multiples en un retour à la ligne maximum
# * Rajouter le "v" dans les états de tous les documents de classe 1 et 2. Ne pas toucher aux documents des autres classes.

# In[1]:


import os
import re
import pandas as pd
from tqdm import tqdm
import IV_util
import Correction_util as Cutil


# In[2]:


txt_path = IV_util.four_SAVE_texts

for fname in tqdm(os.listdir(txt_path)):
    
    with open(txt_path+fname, "r", encoding = "utf8") as f:
        text = f.read()
    
    #print(text)
    # Corrections automatiques de certains mots
    text, _ = Cutil.automatic_correction(text, Cutil.letter_swap_rules())
    
    
    # Uniformisation des apostrophes
    text = re.sub("’","'",text)
    
    # Uniformisation des ligatures
    ligatures_dict = {"Œ":"OE","œ":"oe","Æ":"AE","æ":"ae"}
    for k in ligatures_dict:
        text = re.sub(k,ligatures_dict[k],text)
        
    # Supprimer les espaces et autres signes parasites au début et à la fin des lignes
    unwanted_chars_end = " |=+$£}{][*_"
    unwanted_chars_beginning = unwanted_chars_end + ":;.,"
    
    lines_l = text.split("\n")
    lines_l_strip_begin = list(map(lambda line : line.lstrip(unwanted_chars_beginning), lines_l))
    lines_l_all_strip = list(map(lambda line : line.rstrip(unwanted_chars_end), lines_l_strip_begin))
    
    text = "\n".join(lines_l_all_strip)
    
    # Effondrer les retours à la ligne multiple en une ligne vide maximum
    text = re.sub("\n\n(\n*)","\n\n",text)
       
    # Supprimer les retours de ligne à la fin du texte
    # (Ne pas oublier d'en ajouter un plus tard lors de la jointures des textes en un document)
    text = text.rstrip("\n")
        
    # Supprimer les retours de ligne au début du texte
    text = text.lstrip("\n")
        
        
    with open(txt_path+fname, "w", encoding = "utf8") as g:
        g.write(text)


# In[3]:


# Modifier le log_df pour rajouter "v" à tous les éléments de classe 1 et 2.
# On ne touche pas aux classes 3-4-5, qui ont présumément été catégorisées juste lors de la phase de sélection.
log_df = IV_util.open_log_csv()

def update_state(row):
    
    if not pd.isnull(row["interest"]):
        if row["interest"] == 1 or row["interest"] == 2:
            row["states"].update(["v"])

log_df.apply(lambda row : update_state(row), axis = 1)

log_df["selection_version"] = "prod_3"

IV_util.save_log_csv(log_df)


# In[ ]:





# In[ ]:





# In[ ]:




