#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# # Extraction automatique des métadata
# 
# Le but de ce notebook est de reprendre tous les textes corrigés et les "logs" dans leur état actuel, pour obtenir les csv finaux pouvant être convertis directement en base de données. Il génère en parallèle les fichiers textes des différentes versions du texte original.
# 
# Chaque metadata est extraite grâce à des règles basées sur les expressions régulières (regex). Ces règles sont hiérarchisées selon des niveaux de priorité : on commence par chercher avec les règles les plus restrictives, et on termine avec les règles les plus larges.
# 
# Si plusieurs matchs ont le même niveau de priorité, on choisit lequel garder en fonction de leur position dans le document. Par exemple, les signatures sont plutôt vers la fin tandis que le type de document est plutôt au début.
# 
# Chaque metadata sera extraite selon un système de priorité et des règles regex simples, à déterminer par la pratique.
# On pourra utiliser les majuscules, voire la taille présumée de chaque ligne de texte grâce à l'alto, pour affiner le traitement.
# 
# Définir à chaque fois des niveaux de priorité comme p1 = ("|".join(lieux)), le \d\d ("|".join(mois)) 1\d\d\d et les utiliser. Faire les choses petit à petit en observant à chaque fois quels noms ressortent et en quelle quantité pour établir des listes de noms. Utiliser plusieurs systèmes en parallèles, comme les listes de nom pour les noms courants et les structures avec espace libre pour les noms rares (Ces dernières seront plus basses sur l'échelle de priorité)
# 
# Généralement, pour trouver les métadata, il faut commencer le preprocessing du texte (retirer tous les accents et la ponctuation, virer les lignes vides), mais conserver les retours à la ligne car ils donnent une idée de la taille que peut prendre une métadata (Typiquement, l'intitulé prend quelques lignes au maximum, tandis que les signatures sont généralement en une ou deux lignes. Un lieu + date généralement prend une ligne et l'occupe à lui seul. C'est donc pratique pour distinguer les lieux rares des autres mots qui précéderaient une date.) En revanche, il faut absolument joindre les xxx- \n xxx
# 
# Il faut aussi potentiellement utiliser le fait que les métadata sont souvent groupées, notamment date + signature + imprimeur ou intitulé + date + émetteur
# 
# Les majusucles ou la hauteur de police sont souvent de précieuses indications.
# 
# Peut-être qu'il faudrait aussi séparer les documents par date et utiliser des pattern spécifiques en fonction de ces dates (mais seulement si vraiment nécessaire)
# 
# Pour composer les listes de mots-clefs, commencer par chercher des expression assez génériques, et affiner à partir des résultats obtenus.
# 
# Parfois, on voudra faire converger plusieurs mots vers la même métadonnée, par exemple "décrète" et "décret" vers "décret", ou "LL Roten" et "L L Roten" vers le gars en questions. Dès lors, il sera judicieux de créer une fonction et un dictionnaire permettant de mapper, si une regex est trouvée, le match à une métadata particulière.
# 
# Certaines métadata sont exclusives, dans le sens ou une date ou un destinataire ne peuvent pas être simultanément une autre métadata. En revanche, les métadata sont souvent proches les unes des autres. On pourrait donc noter la position des métadata et chercher à proximité.

# Pour ce qui est des compilations, il y en a trop pour que ce soit raisonnable de les gérer à la main. En revanche, les compilations de niveau 4, de loin les plus massives, sont des compilations de lois très structurées, ce qui devrait permettre d'isoler chaque texte assez facilement automatiquement.
# 
# Bien séparer la partie "détection" des metadata et la partie choix pour séparer entre compilations et non-compilations. La partie détection renvoie le match avec des index. La partie choix traite ces index. Pour les compilations, observer si on a une succession de groupes contenant plusieurs métadata. Ceci en se rappelant que les compilations sont généralement très régulières.

# On sauvegardera donc au moins deux versions du texte : le texte original, sans retouche à part uniformisation des apostrophes et effondrement des retours à la lignes vides multiples en un retour à la ligne vide à la fois max (On a remarqué que ces retours sont quand même souvent très justes pour délimiter les paragraphes, tout comme les accents sont souvent justes) ; et la séquence de mots pour la recherche, constituée d'un maximum de processing (regroupement des mots enjambeant deux lignes, suppression de toute la ponctuation et les accents, etc)

# In[1]:


import IV_util
import topic_modelling_util as Tutil
import re
import os
import pandas as pd
import numpy as np
import unidecode
import string
from tqdm import tqdm
import convertdate
import PIL.Image
from functools import cmp_to_key


# In[2]:


# Les longues opérations peuvent être effectuées une seule fois et sauvegardées.
# Si False, le résultat de l'opération est chargé.
PERFORM_LONG_OPERATIONS = False
PERFORM_TOPIC_MODELLING = False
PERFORM_FINAL_SAVE = False


# ## Préparation du nouveau df
# Le nouveau df est entièrement et automatiquement extrait de log_df. Il contient une ligne par cote. De ce df, on extraira directement les métadonnées nécessaires pour générer le triple IIIF (difficile de sauvegarder ce df en l'état car certaines colonne prennent énormément de place.
# 
# On a besoin à la fois d'informations relatives aux cotes et d'informations relatives aux images. Il paraît donc nécessaire d'avoir deux df. Un avec une ligne par cote et un avec une ligne par image.

# In[3]:


log_df = IV_util.open_log_csv()

useful_log_df = log_df[log_df["is_included"]==1][["cote","states","official_name", "binary_files",
                                                 "txt_name","interest","is_compil"]]

# "roman" est le meilleur qualificatif que j'ai trouvé pour regrouper latin, français et italien.
useful_log_df = useful_log_df.rename(columns = {"official_name":"raw_img_filenames",
                                                "binary_files":"roman_binary_img_filenames",
                                               "txt_name":"roman_txt_filenames"})


# Suppression des noms de fichier textes qui n'existent pas réellement,
# c'est-à-dire ceux pour lesquels il n'existe pas de binary image.
useful_log_df["roman_txt_filenames"] = useful_log_df.apply(
    lambda row : pd.NA if pd.isnull(row["roman_binary_img_filenames"]) else row["roman_txt_filenames"], axis=1)


# Fillna pour que les colonnes puissent s'aggréger facilement
#useful_log_df["roman_binary_img_filenames"] = useful_log_df["roman_binary_img_filenames"].apply(lambda x :
                                                                                               #set()
                                                                                                #if pd.isnull(x)
                                                                                               #else x)


metadata_cote_df = useful_log_df.groupby("cote").agg({"states":lambda x : set.union(*x.tolist()),
                                                 "raw_img_filenames":lambda x: x.tolist(),
                                                # "roman_binary_img_filenames":lambda x : set.union(*x.tolist()),
                                                # "roman_txt_filenames":lambda x : x.tolist(),
                                                "interest": lambda x : set(x.tolist()),
                                                 "is_compil":lambda x : any(x) if not any(pd.isnull(x)) else pd.NA})

# Quand tous les fichiers sont traités en même temps, interest ne varie pas pour une cote donnée.
# Dans le cas contraire, il faut traiter.
def harmonize_interest(interest_set,is_empty_structure):
    """
    Given the set of interests for all images, return the most probable interest for the cote
    interest_set : the set of interest
    empty_structure : boolean registering wether the cote is an empty structure or not
    Return : int
    
    We except the interest to be the same for all images, as it is a cote characteristic, but it
    may happend, after a non-standard processing, that the csv is inconsistent.
    This function allow to solve the inconsistency.
    """
    
    # NA ne nous donne aucune information
    if pd.NA in interest_set:
        interest_set.remove(pd.NA)
        
        # Si seul NA était présent, ça veut dire que la cote n'a pas été traitée
        if len(interest_set) == 0:
            return pd.NA
     
    # S'il ne reste qu'un élément dans le set, c'est forcément celui qu'on cherche
    if len(interest_set) == 1:
        return interest_set.pop()
    
    else:
        # Si une catégorie a été expressement choisie et l'autre est une catégorie de base,
        # alors celle qui a été choisie n'a juste pas été appliquée partout
        out_of_norm_decision_set = interest_set - set([2,4])
        
        if len(out_of_norm_decision_set) == 1:
            return out_of_norm_decision_set.pop()
        
        # Dans tous les autres cas, on est face à une indétermination et on retourne la catégorie par défaut
        else:
            if is_empty_structure:
                return 2
            else:
                return 4

metadata_cote_df["interest"] = metadata_cote_df.apply(lambda row : harmonize_interest(row["interest"],
                                                                            "v" in row["states"]),axis=1)


# In[4]:


metadata_img_df = useful_log_df[["cote","states","raw_img_filenames","roman_binary_img_filenames",
                                 "roman_txt_filenames"]]

metadata_img_df["has_roman_txt"] = ~pd.isnull(metadata_img_df["roman_txt_filenames"])


# In[5]:


metadata_cote_df.sample(5)


# In[6]:


metadata_img_df.sample(5)


# ## Préparation des textes
# On va sauvegarder le texte original, le texte traité intermédiairement pour la recherche des métadonnées, et la séquence de mots.
# 
# Pour le texte original, on va simplement copier dans le df img le texte de chaque fichier. En revanche, le texte intermédiaire français et la séquence de mot pour chaque langue seront stockés dans le df cote. On crée uniquement un texte intermédiaire français puisqu'on ne cherche les métadonnées qu'en français. A un moment, il faudra renommer ces textes en "roman" plutôt que "fr", comme ils contiennent les 3 langues romanes.
# 
# ! Il est essentiel de séparer les textes par langue déjà maintenant. C'est pas grave si des morceaux d'autre langue atterrissent dedans pour les pages doubles français-latin et français italien (très rares d'ailleurs, moins de 30 sur 14'000). Mais, sachant que la recherche plain text et la recherche des métadata dépend de la langue, et que dans le futur les textes allemands seront ajoutés, il est essentiel que l'on garde trace de la langue de chaque image, même dans le df par cote. Même, en fait ça me paraît assez important de garder un texte par image dans le stockage IIIF, pour qu'on sache toujours exactement comment relier chaque texte à chaque image.
# 
# On constate que stocker les textes dans le dataFrame ressemble plutôt à une mauvaise idée. Les textes mettent beaucoup de temps à charger, et les df risquent d'être rapidement inutilisables. La meilleure solution consiste donc probablement à créer des nouveaux fichiers textes dans d'autres dossiers.
# 
# Comme on va finalement se fixer sur une séquence par image, on ne va pas pouvoir distinguer les langues italiennes, latines ou françaises. On pourrait se contenter d'une seule "roman sequence" par texte. Mais vu le peu de documents doubles français+latin ou italien+latin, je pense qu'on pourrait plutôt **dupliquer les textes et les appeler par leur nom de langue**, comme ça on est vraiment propre dans l'appellation, même si les données sont un peu sales dessous (mais ça change pas beaucoup des déchets allemands disséminés un peu partout dans les textes français)
# 
# ### Post-Processing texte intermédiaire
# Le texte intermédiaire s'applique bien sûr à tout le document, puisqu'on recherche les métadonnées. On va le créer uniquement en français puisqu'on 
# * Joindre toutes les pages qui partagent la même langue
# * Retirer la ponctuation (Pour éviter les problèmes avec la ponctuation sans espace, d'abord remplacer la ponctuation par un espace, puis remplacer les espaces multiples par des espaces simples)
# * Retirer tous les accents
# * Retirer toutes les lignes vides
# * Remplacer les & par des et
# * lowercase ? -> On peut utiliser plutôt le flag re.IGNORECASE. Mais ne pas oublier de l'utiliser !
# * Virer le caractère spécial de séparation des pages (Puisque le texte intermédiaire ne sera a priori pas mis dans le IIIF)
# * Joindre les xxx-\nxxx
# 
# ### Post-Processing séquence de mots
# * Faire tout le post-processing de texte intermédiaire
# * Supprimer tous les retours à la ligne
# * Lowercase
# * Remplacer les apostrophes par des espaces ? (Si oui, penser à faire la même manipulation dans l'outil de recherche)

# In[7]:


# Création des fichiers-texte intermédiaires
def original_to_intermediate_text_processing(original_text):
    """
    Process an original text to its intermediate-processed version.
    The intermediate-processed version is used to detect many metadata, but is not saved in the final dataset
    original_text : string
    Return : the processed string
    """
    # Retirer la ponctuation. On garde les traits d'union car ils sont importants pour lier les mots.
    # On garde aussi les apostrophes parce qu'on les utilise pour chercher les métadata.
    punct_to_remove = string.punctuation.replace("-","").replace("'","").replace("&","")
    text = re.sub("[{}]".format(punct_to_remove)," ",original_text)
    
    # On a remplacé la ponctuation par des espaces. On peut maintenant effondrer les espaces multiples en un seul
    text = re.sub(" ( )*"," ",text)
    
    # Retirer tous les accents
    # (Cette opération a des effets collatéraux, comme ° -> deg, mais ça n'est pas trop grave pour nous)
    text = unidecode.unidecode(text)
    
    # Retirer les espaces en début et en fin de ligne
    text = re.sub(" *\n *","\n",text)
    
    # Retirer toutes les lignes vides
    text = re.sub("\n(\n)*","\n",text)
    
    # Remplacer les "&" par des "et"
    text = re.sub("&","et",text)
    
    # Virer le caractère spécial de fin de page (Il s'agit d'un saut de page, \f ou \x0c)
    text = re.sub("\f","",text)
    
    # Joindre les mots qui enjambent
    text = re.sub("(\w)-\n(\w)","\\1\\2",text)
    
    return text


def intermediate_text_filename_generation(cote, lang):
    """
    Generate the name of a given intermediate text
    cote : the cote
    lang : the language to look for to create the texts. Must follow the Code convention ("f","l","t" or "a")
    Return : str
    """
    lang_dict = {"f":"fr","a":"de","t":"it","l":"la"}
    
    inter_text_filename = "CH-AEV_IV_"+cote+"_interm_text_"+lang_dict[lang]+".txt"
    
    return inter_text_filename


def intermediate_text_and_tokens_creation(cote, row_cote, input_dir, output_dir, lang, metadata_img_df):
    """
    Create an intermediate text file for the given row of dataframe and the given language
    cote : the cote (index of the row of the metadata)
    row_cote : a row of the metadata
    input_dir : the input directory of the original texts
    output_dir : the output directory for the intermediate texts
    lang : the language to look for to create the texts. Must follow the Code convention ("f","l","t" or "a")
    metadata_img_df : the dataframe where each image has one row
    Return : the list of tokens for the text (extra functionality)
    Create a text file.
    """
    
    inter_text_filename = intermediate_text_filename_generation(cote, lang)

    

    # Sélectionner les lignes correspondant à la cote courante
    cote_img_df = metadata_img_df[metadata_img_df["cote"] == cote]

    # Ne garder que les pages dans la langue courante
    cote_img_lang_df = cote_img_df[cote_img_df["states"].apply(lambda s : lang in s)]

    # Ne pas garder les pages manuscrites, qui n'ont pas de texte associé.
    cote_img_no_script_lang_df = cote_img_lang_df[cote_img_lang_df["states"].apply(lambda s : "m" not in s and
                                                                                  "b" not in s)]

    if not cote_img_no_script_lang_df.empty:

        pages_to_retrieve = sorted(cote_img_no_script_lang_df["roman_txt_filenames"].tolist())

        texts_list = []

        ###
        topic_modelling_texts_list = []
        ###

        for fname in pages_to_retrieve:

            with open(input_dir+fname,"r",encoding="utf8") as f:
                current_text = f.read()
                processed_text = original_to_intermediate_text_processing(current_text)
                texts_list.append(processed_text)

                ###
                topic_modelling_texts_list.append(current_text)
                ###

        full_intermediate_text = "\n".join(texts_list)

        # Ressuprimer les lignes vides ici et rejoindre les mots qui enjambent deux pages
        full_intermediate_text = re.sub("\n(\n)*","\n",full_intermediate_text)
        full_intermediate_text = re.sub("(\w)-\n(\w)","\\1\\2",full_intermediate_text)

        if not os.path.isfile(output_dir+inter_text_filename):
            with open(output_dir+inter_text_filename,"w",encoding="utf8") as g:
                g.write(full_intermediate_text)

        ###
        full_original_text = " ".join(topic_modelling_texts_list)
        full_original_text = re.sub("\n(\n)*"," ",full_original_text)
        full_original_text = re.sub("&","et",full_original_text)
        full_original_text = re.sub("\f","",full_original_text)
        full_original_text = re.sub("(\w)-\n(\w)","\\1\\2",full_original_text)

        tokens = Tutil.topic_modelling_preprocessing(full_original_text)

        return tokens
        ###


# In[8]:


def word_sequence_filename_generation(row, lang_code):
    """
    Generate the filename for a given word sequence.
    row : the row of metadata_img_df
    lang_code : the code for the language. Must follow the Code convention ("f","l","t" or "a")
    Return : str
    """
    lang_dict = {"f":"fr","a":"de","t":"it","l":"la"}
    
    seq_name = IV_util.get_prefix(row["raw_img_filenames"])+"_word_seq_"+lang_dict[lang_code]+".txt"
    
    return seq_name
   
    
def original_to_word_sequence_text_processing(original_text):
    """
    Turn the original text into a word sequence maximally processed.
    original_text : the original text to be processed.
    Return : str
    """
    
    # On commence par toutes les étapes de traitement vers le résultat intermédiaire
    intermediate_text = original_to_intermediate_text_processing(original_text)
    
    # Suppression de toutes les ponctuations y compris les apostrophes (! à l'outil de recherche)
    text = re.sub("[{}]".format(string.punctuation)," ",intermediate_text)
    
    
    # Suppression des retours à la ligne
    text = re.sub("\n"," ",text)
    
    # Effondrement des espaces en un seul espace
    text = re.sub(" ( )*"," ",text)
    
    # Lowercase tout
    text = text.lower()
    
    return text


# In[9]:


# Création des textes intermédiaires et du dataframe de tokens
if PERFORM_LONG_OPERATIONS: # Désactivé quand on sait que tous les textes ont été transcrits
    
    topic_modelling_df = pd.DataFrame(metadata_cote_df["is_compil"])
    topic_modelling_df["tokens"] = pd.NA
    
    for c, row in tqdm(metadata_cote_df.iterrows()):
        tokens = intermediate_text_and_tokens_creation(
            c,row,IV_util.four_SAVE_texts, IV_util.four_TEMP_intermediate,"f",metadata_img_df)
        
        topic_modelling_df.at[c,"tokens"] = tokens
        
        topic_modelling_df.to_csv("topic_modelling_tokens.csv", encoding = "utf8")

else:
    topic_modelling_df = pd.read_csv("topic_modelling_tokens.csv", encoding = "utf8")
    topic_modelling_df["tokens"] = topic_modelling_df["tokens"].apply(lambda x : eval(x) if not pd.isnull(x) else x)


# In[ ]:





# In[10]:


# Création des séquences de mots
if PERFORM_LONG_OPERATIONS: # Désactivé quand on sait que tous les textes ont été transcrits
    
    for i, row in tqdm(metadata_img_df.iterrows()):

        if "m" not in row["states"] and "b" not in row["states"]:

            for lang_code in ["f","t","l"]:

                if lang_code in row["states"]:

                    seq_name = word_sequence_filename_generation(row, lang_code)

                    if not os.path.isfile(IV_util.four_SAVE_word_sequences+seq_name):

                        with open(IV_util.four_SAVE_texts+row["roman_txt_filenames"],"r",encoding="utf8") as f:
                            text = f.read()

                        word_seq = original_to_word_sequence_text_processing(text)              

                        with open(IV_util.four_SAVE_word_sequences+seq_name,"w",encoding="utf8") as g:
                            g.write(word_seq)


# In[11]:


# Génération des noms de fichier pour les séquences de mots
def cond_interm(row, lang_code):
    """
    Condition for a row to have a word sequence in a given language
    row : the current row of the dataframe
    lang_code : "f", "t", "l" or "a" the language identification
    Return : bool
    """
    return lang_code in row["states"] and row["has_roman_txt"]

#metadata_cote_df["fr_interm_txt_filenames"] = metadata_cote_df.apply(lambda row :
#    intermediate_text_filename_generation(row.name,"f") if cond_interm(row,"f") else pd.NA,
#                                                                     axis=1)
# Gros bordel car c'est impossible de maintenir l'information lors de l'aggrégation, à cause des langues.

metadata_img_df["fr_word_seq_filenames"] = metadata_img_df.apply(lambda row:
    word_sequence_filename_generation(row, "f") if cond_interm(row,"f") else pd.NA,
                                                                 axis=1)

metadata_img_df["it_word_seq_filenames"] = metadata_img_df.apply(lambda row:
    word_sequence_filename_generation(row, "t") if cond_interm(row,"t") else pd.NA,
                                                                 axis=1)

metadata_img_df["la_word_seq_filenames"] = metadata_img_df.apply(lambda row:
    word_sequence_filename_generation(row, "l") if cond_interm(row,"l") else pd.NA,
                                                                 axis=1)


# In[12]:


# Génération des noms de fichier pour les textes intermédiaires
for filename in os.listdir(IV_util.four_TEMP_intermediate):
    cote = IV_util.get_cote(filename)
    
    metadata_cote_df.at[cote,"fr_interm_txt_filenames"] = filename
    


# In[13]:


metadata_cote_df.sample(5)


# In[14]:


metadata_img_df.sample(5)


# ## Fonctions génériques pour les métadata

# In[15]:


# Fonction pour donner la position dans la string et le match pour un regex particulier
def find_pattern(pattern, text, allow_part_of_word = False, enable_case_sensitive_search = False):
    """
    Generic function to match a regex pattern in our metadata quest.
    By default, the pattern is surrounded by \\b (word boundary) and is case-insensitive.
    allow_part_of_word (default False) : allow the pattern to begin or end in the middle of a word.
    enable_case_sensitive_search(default False) : restrict the search to case-sensitive
    Return : a list of match object
    """
    if allow_part_of_word and not enable_case_sensitive_search:
        return list(re.finditer(pattern,text, re.IGNORECASE))
    
    elif not allow_part_of_word and enable_case_sensitive_search:
        return list(re.finditer("\\b"+pattern+"\\b", text))
    
    elif allow_part_of_word and enable_case_sensitive_search:
        return list(re.finditer(pattern,text))
    
    else:
        return list(re.finditer("\\b"+pattern+"\\b",text, re.IGNORECASE))

def retrieve_unique_pattern(pattern, text, retrieve_match_obj = False, allow_part_of_word = False):
    """
    Generic function to retrieve a pattern present one single time in the text.
    Raise an exception if the pattern is not present exactly one time.
    Return : a string
    """
    pattern_list = find_pattern(pattern, text, allow_part_of_word = allow_part_of_word)
    if len(pattern_list) != 1:
        raise Exception("Pattern {} did not appeared exactly one time in string {}".format(pattern, text))
        
    else:
        if retrieve_match_obj:
            return pattern_list[0]
        else:
            return pattern_list[0].group()
        
def search_for_pattern(pattern, text):
    """
    Generic function that encapsulate the search for the first occurence of a pattern that may not exist.
    (re.search). Case insensitive and pattern cannot start or end in the middle of a word.
    Return : match object or None
    """
    
    return re.search("\\b"+pattern+"\\b",text, re.IGNORECASE)


# In[16]:


def decide_begin(match_list):
    """
    Among all the match objects, return the match string the closest to the beginning of the text
    Return : a string
    """
    if len(match_list) == 0:
        return pd.NA
    
    beginnings = np.array(list(map(lambda match_object : match_object.span()[0], match_list)))

    closest_index = np.argmin(beginnings)
    
    return match_list[closest_index].group()
    

def decide_end(match_list, return_whole_match_object = False):
    """
    Among all the match objects, return the march string the closest to the end of the text
    Return : a string. If return_whole_match_object is True, return a match object
    """
    if len(match_list) == 0:
        return pd.NA
    
    ends = np.array(list(map(lambda match_object : match_object.span()[1], match_list)))
    
    closest_index = np.argmax(ends)
    
    if return_whole_match_object:
        return match_list[closest_index]
    else:
        return match_list[closest_index].group()
    

def decide_extrem(match_list, text_length):
    """
    Among all the match objects, return the match string the closest to an extremity
    (end or beginning) of a text.
    Return : a string
    
    No special action is taken when there is a tie : one of the cases of the tie is randomly chosen.
    We assume these situation are very rare, and ambiguous in any case.
    """
    if len(match_list) == 0:
        return pd.NA
    
    distances = np.array(list(map(lambda match_object : min(match_object.span()[0],
                                                       text_length - match_object.span()[1]), match_list)))
    
    closest_index = np.argmin(distances)
    
    return match_list[closest_index].group()
    
def decide_maj(match_list):
    """
    Return the match that has the largest mean uppercase-ratio per word.
    When equal, return all. Hence it must be combined with another decide function
    Return : a list of match objects.
    """
    
    def uppercase_mean_word_ratio(match_text):
        words = match_text.split()
        
        def word_uppercase_ratio(word):
            letters = word.split("")
            
            n_letters = len(letters)
            n_uppercase = np.array(map(str.isupper), letters).sum()
            
            return n_uppercase/n_letters
        
        maj_ratio_per_word = np.array(map(word_uppercase_ratio, words))
        
        return maj_ratio_per_word.mean()
    
    maj_ratio = np.array(map(lambda match : uppercase_mean_word_ratio(match.group()), match_list))
    
    max_maj_ratio = maj_ratio.max()
    
    return list(np.array(match_list)[maj_ratio == max_maj_ratio].flatten())

# Pas de fonction de décision pour les matches qui occupent une ligne seule. Ca sera plutôt un critère
# de classe de priorité, dans les quelques cas où c'est nécessaire.


# In[17]:


def parenthesise(expr):
    "Put parentheses around expr"
    return "(" + expr + ")"

def make_re_group(l):
    "Create a regex union of the element of the list l"
    return parenthesise("|".join(l))


# Fonction pour résumer les stats pour une recherche donnée

# On ne va pas réaliser une mega-fonction générique pour trouver le meilleur pattern parmi toute une liste de levels
# parce que ça rendra juste le truc moins lisible.

# Est-ce intéressant d'enregistrer le niveau de fiabilité des prédictions pour faire des stats ?
# Je pense que oui, même si on ne le sauve pas à la toute fin.

# Toujours bien séparer détection des métadonnées et choix, puisque les compilations vont être traitées différement
# des autres documents.
# C'est facile en supposant que les compilations ont toutes leurs métadata au même niveau de priorité
# (Ce qui est une supposition tout à fait valide comme ce sont généralement des documents similaires compilés)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Dates
# Particularités des dates:
# * Présence du calendrier révolutionnaire. A convertir
# * Variations orthographiques des mois : Aoust, may
# * Notations 7bre, 9bre, xbre... pour les mois. Il s'agit visiblement des abbréviations standards. xbre = décembre, 9bre = novembre, etc. Attention à les transcrire correctement !
# * présence de déchets possible entre le numéro et le mois, par exemple 1°* pour 1er ou 3" pour 3e.
# * Parfois, absence du jour. le octobre 1850. Parfois, seulement l'année voire 185 , 18  .
# * Rares dates multiples, comme "les 2 et 3 décembre 1850"
# * Dates simples entre parenthèses, juste sous l'intitulé, comme "LOI \n (5 mai 1850)" intitué contient généralement le mot "fédéral(e)", et/ou le lieu est Berne. Mais on peut utiliser la date + lieu en bas du document.
# * Très rarement, une date peut être 00 au lieu de 1 à 31. Probablement pour un projet dont la date n'est pas fixée.
# * Une date incomplète peut être plus significative qu'une date complète ! Par exemple "Ainsi adopté le novembre 1867" est plus pertinent que toute autre date complète présente dans le document.
# * Format "POUR L'ANNÉE yyyy" comme dans "ALMANACH POUR L'ANNÉE ..."
# * Contrairement à la plupart des autres métadata, les retours de ligne peuvent être problématiques pour les dates car ils peuvent couper une date en deux assez facilement. Il faudra donc penser à rechercher aussi sans les retours de ligne.
# * Quelques dates sont en toutes lettres ! Pour les dates en toute lettre, on trouve parfois "et" entre la dizaine et l'unité.
# * Parfois, des dates comme "Sion, en janvier 1875"
# * Parfois, le "le" de la formule lieu + date n'est pas présent
# * Les dates peuvent être précédées du mot "circulaire" sur la même ligne
# * Formule top-level : Donné en Conseil d'Etat à Sion, le .... (et ses variations)
# * Formule comme "le 25 Jour du mois de May" ou "le 25 jour du mois d'Aout"
# 
# La formule typique est "à Sion, le ..." mais on trouve aussi des formules comme "Loi du X novembre YYYY".
# 
# Niveaux de hiérarchie : Complétude et position uniquement. Les majuscules ne sont pas pertinentes.
# 
# Dans le cas où la date est absente, faut-il considérer celle des titres de fichier ?

# In[18]:


# Patterns de dates

## Noms de mois
classical_month_list = ["janvier","fevrier","mars","avril","mai","juin","juillet","aout","septembre","octobre",
                       "novembre","decembre"]
revolutionary_month_list = ["vendemiaire","brumaire","frimaire","nivose","pluviose","ventose",
                           "germinal","floreal","prairial","messidor","thermidor","fructidor"]
variant_month_list = ["may","aoust","decembr"]
short_month_list = ["jer","7bre","8bre","9bre","10bre","xbre"] # On omet "Fer" qui se confond avec "fer".

all_month_list = classical_month_list + revolutionary_month_list + variant_month_list + short_month_list

## Regex pour tous les noms de mois
month_group_regex = make_re_group(all_month_list)
gregorian_month_group_regex = make_re_group(classical_month_list + variant_month_list + short_month_list)
republican_month_group_regex = make_re_group(revolutionary_month_list)

## Petites formules pouvant faire la transition entre le mois et l'annéee
extra_between_month_and_year = "( l'an| de l'an)?"

## Regex comprenant le mois + l'année, pour chaque calendrier
month_and_year_group_regex = make_re_group(
    [republican_month_group_regex+extra_between_month_and_year+" \d+",
     gregorian_month_group_regex+extra_between_month_and_year+" \d\d\d\d"])

## Petites formules pouvant faire la transition entre jour et mois
extra_between_day_and_month = "(jour du mois de |jour du mois d'|de |d')?"

## Regex de dates pour les dates complètes (avec jour) ou partielles (mois et année seulement), numériques.
full_date_regex = "\d\d?'*(er|me|e|deg)? "+extra_between_day_and_month+month_and_year_group_regex
partial_date_regex = month_and_year_group_regex

## Noms de chiffres possibles
# Tous les chiffres de jours peuvent être adjectivés, comme "seizieme"
toutes_lettres_chiffres = ["deux","trois","quatre","cinq","six","sept","huit","neuf"]
toutes_lettres_chiffres_purs = ["un"] + toutes_lettres_chiffres
toutes_lettres_chiffres_jours = ["premier"] + toutes_lettres_chiffres
toutes_lettres_nombres_dix = ["onze","douze","treize","quatorze","quinze","seize"]
toutes_lettres_dizaines = ["dix","vingt","trente","quarante","cinquante","soixante","septante","soixante dix",
                          "huitante","quatre vingt","nonante","quatre vingt dix"]
toutes_lettres_dizaines_jours = ["dix","vingt","trente"]
toutes_lettres_millenaires = ["mille","mil"]
toutes_lettres_centaines = ["cent","cens"]
toutes_lettres_annee = ["l'an de grace","de l'an"]

## Formules annonçant une date
formules_preliminaires = ["Ainsi adoptee", "Donnee", "Donne", "Fait"]
chambres_pour_formules_preliminaires = ["Grand Conseil","Conseil d'Etat"]

## Lieux
vrais_lieux = ["sion", "lucerne", "lausanne", "berne", "gernsbach", "st-gall", "brigue", "zurich", "bex",
              "fribourg", "sierre", "genes", "st-maurice", "paris", "turin", "palais des tuileries",
              "palais de saint cloud", "bechenkowiski", "witepsk", "smolensk", "slawkovo", "wiazma", "ghjat",
              "mojaisk", "moscou", "borowsk", "roilskoe", "vereia", "molodetschno", "palais de l'elysee",
              "lutzen", "francfort", "loerrach", "bale", "thonon", "troyes", "geneve", "lons-le-saunier",
              "yverdon", "monthey", "bramois", "viege", "bonneville", "illarsaz", "chamoson", "martigny",
              "aigle", "verolliez", "aoste", "rome", "aarau", "posieux", "massongex", "ratisbonne", "vaumarcus",
              "copenhague", "neuchatel", "coire", "riddes", "brougg argovie", "vernayaz", "glarus"]
coquilles_lieux = {"coirele":"coire", "paixberne":"berne", "precieufesion":"sion", "cirsion":"sion",
                  "vconsion":"sion"}
uniformize_lieux = {"st maurice":"st-maurice", "st gall":"st-gall", "massongez":"massongex",
                   "arau":"aarau"}
                   
lieux = vrais_lieux + list(coquilles_lieux.keys()) + list(uniformize_lieux.keys())
correct_lieux_dict = dict()
correct_lieux_dict.update(coquilles_lieux)
correct_lieux_dict.update(uniformize_lieux)

## Déterminant introduisant la date
prefix_date = ["le ","ce "]

## Regex pour les dates avec lieux
lieu_full_date_regex = make_re_group(lieux) + " " + make_re_group(prefix_date)+"?" + full_date_regex
lieu_partial_date_regex = make_re_group(lieux) + " " + "(en )?" + partial_date_regex
# (en )? SERAIT PEUT-ETRE PLUS PERTINENT AILLEURS (dans les prefix-date par exemple ?)

## Regex pour les formules introduisant les dates, complète (participe + gouvern.) ou partielles (participe seul)
full_formule_regex = make_re_group(formules_preliminaires) + " en " + make_re_group(
    chambres_pour_formules_preliminaires) + " a " + make_re_group(lieux) + " " + make_re_group(
    prefix_date)

partial_formule_regex = make_re_group(formules_preliminaires) + " a " + make_re_group(
lieux) + " " + make_re_group(prefix_date)

# Conversion des noms de mois leurs valeurs numérales
normalize_month_dict = {"janvier":1,"fevrier":2,"mars":3,"avril":4,"mai":5,"juin":6,"juillet":7,"aout":8,
                       "septembre":9,"octobre":10,"novembre":11,"decembre":12,
                       "vendemiaire":1,"brumaire":2,"frimaire":3,"nivose":4,"pluviose":5,"ventose":6,
                       "germinal":7,"floreal":8,"prairial":9,"messidor":10,"thermidor":11,"fructidor":12,
                       "may":5,"aoust":8,"decembr":12,
                       "jer":1,"7bre":9,"8bre":10,"9bre":11,"10bre":12,"xbre":12
                       }


# In[19]:


# Fonction de chargement du texte d'un fichier et de traitement pour chercher la date (\n -> espace)

# Fonction de recherche de date
def find_most_probable_dates(text):
    """
    Find in text the patterns that are the most probable to be the date of the documents and return them as
    a list match objects, along with the code that characterize the date.
    text : the text to be analyzed
    Return : (match_list, code)
    match_list : the list of match objects that contains dates
    code : the code of the level of probability of the given date.
    """
    
    # Niveau de complexité 1 (Formule avec participe passé + entité gouvernementale + lieu + date)
    match_list = find_pattern(full_formule_regex + full_date_regex, text) 
    if len(match_list) > 0:
        return match_list, "formule_grande+date_complete"
    
    match_list = find_pattern(full_formule_regex + partial_date_regex, text)
    if len(match_list) > 0:
        return match_list, "formule_grande+date_partielle"
    
    # Niveau de complexité 2 (Formule avec participe passé + lieu + date)
    match_list = find_pattern(partial_formule_regex + full_date_regex, text)
    if len(match_list) > 0:
        return match_list, "formule_petite+date_complete"
    
    match_list = find_pattern(partial_formule_regex + partial_date_regex, text)
    if len(match_list) > 0:
        return match_list, "formule_petite+date_partielle"
    
    # Niveau de complexité 3 (Lieu + date)
    match_list = find_pattern("a " + lieu_full_date_regex, text)
    if len(match_list) > 0:
        return match_list, "a_lieu+date_complete"
    
    match_list = find_pattern(lieu_full_date_regex, text)
    if len(match_list) > 0:
        return match_list, "lieu+date_complete"
    
    match_list = find_pattern("a "+lieu_partial_date_regex, text)
    if len(match_list) > 0:
        return match_list, "a_lieu+date_partielle"
    
    match_list = find_pattern(lieu_partial_date_regex, text)
    if len(match_list) > 0:
        return match_list, "lieu+date_partielle"
    
    # Niveau de complexité 4 (Préfixe + date)
    match_list = find_pattern(make_re_group(prefix_date) + full_date_regex, text)
    if len(match_list) > 0:
        return match_list, "le+date_complete"
    
    match_list = find_pattern(make_re_group(prefix_date) + partial_date_regex, text)
    if len(match_list) > 0:
        return match_list, "le+date partielle"
    
    # Niveau de complexité 5 (Date complète en toutes lettres)
    # Actuellement pas implémenté car cela prendrait du temps pour un nombre de dates limitées,
    # Et serait difficilement implémentable vu le grand nombre de variations possibles.
    # Idée ! Pour les dates en toute lettres, pourrait-on remplacer dans le texte chaque
    # nom de chiffre par son équivalent en vrai chiffre ?
    # Il resterait des subtilités mais au moins ça serait plus simple.
    # Mieux ! Commencer par trouver le mois et ensuite chercher autour du mois les chiffres.
    #
    # Note supplémentaire : si les années en toutes lettres sont rares,
    # en revanche tous les documents liés au département du Simplon ont des jours en toute lettre
    # Ca pourrait être judicieux de les intégrer.
    
    # Niveau de complexité 6(Date complète)
    match_list = find_pattern(full_date_regex, text)
    if len(match_list) > 0:
        return match_list, "date_complete"
    
    # Niveau de complexité 7 (Formules spéciales)
    match_list = find_pattern("pour l'annee \d\d\d\d|pour l'exercice( de)? \d\d\d\d", text)
    if len(match_list) > 0:
        return match_list, "pour_annee+annee"
    
    # Rajouter un niveau 8 pour les dates partielles sur ligne seule
    # (Déplacer la conversion du texte pour la recherche de date dans cette fonction pour avoir acces aux
    # deux textes ?)
    
    # Niveau de complexité 9 (Date partielle unique)
    match_list = find_pattern(partial_date_regex, text)
    if (len(match_list) == 1) or (len(set(map(lambda mobj : mobj.group(), match_list))) == 1):
        return match_list, "date_partielle_unique"
    
    else:
        return list(), "no_match"
    
    

# Fonction de choix de date (decide_extrem, donc pas besoin de faire un alias si hors compil.)
# Quant aux compil, il faudra gérer quand on a tous les éléments en même temps.

# Fonction de conversion de date
# Cette fonction extrait la date et le lieu de la str trouvée et les converti.
# Plutôt une fonction spéciale pour le lieu ?
def normalize_date(date_text):
    """
    Extract and convert a date from a date text that contains the wanted date.
    The day and the year must be in a numerical format.
    """
    republican_flag = False
    
    date_text = date_text.lower()
    
    # Extraction des données spécifiques des dates
    full_date = search_for_pattern(full_date_regex, date_text)    
    if full_date is not None:
        full_date_str = full_date.group()
        day = retrieve_unique_pattern("^\d\d?",full_date_str, allow_part_of_word = True)
        month = retrieve_unique_pattern(month_group_regex,full_date_str)
        
        if month in revolutionary_month_list:
            republican_flag = True
            year = retrieve_unique_pattern("\d+$",full_date_str)
        else:
            year = retrieve_unique_pattern("\d\d\d\d$",full_date_str)

        
    else:
        partial_date = search_for_pattern(partial_date_regex, date_text)
        if partial_date is not None:
            partial_date_str = partial_date.group()
            day = None
            month = retrieve_unique_pattern(month_group_regex,partial_date_str)
            
            if month in revolutionary_month_list:
                republican_flag = True
                year = retrieve_unique_pattern("\d+$",partial_date_str)
            else:
                year = retrieve_unique_pattern("\d\d\d\d$",partial_date_str)
        
        else:
            day = None
            month = None
            year = retrieve_unique_pattern("\d\d\d\d$",date_text)
          
        
    # Conversion de ces données
    
    ## Conversion en chiffres
    if month is not None:
        month_num = normalize_month_dict[month]
    else:
        month_num = None
        
    day_num = int(day) if day is not None else None
    year_num = int(year)
    
    ## Conversion des dates républicaines en dates grégoriennes
    if republican_flag:
        if day is not None:
            year_num, month_num, day_num = convertdate.french_republican.to_gregorian(year_num, month_num, day_num)
        else: # Il n'existe aucune correspondance parfaite entre mois républicain et grégorien, donc faut approximer
            year_num, month_num, _ = convertdate.french_republican.to_gregorian(year_num, month_num, 15)
            
    return day_num, month_num, year_num


# In[20]:


find_most_probable_dates("fait a Sion le 9 janvier 1815")


# ## Complément aux dates
# Pour les dates qui n'ont pas été trouvées dans les documents, on va utiliser si possible les dates des titres originaux des documents, afin d'avoir une couverture maximale (certes, certaines de ces dates sont fausses, mais c'est dans l'ensemble toujours mieux que rien)

# In[21]:


#TODO

# Considérer should_be_name, pas Z_name pour prendre en compte les éventuels recotages

# Dans la recherche par range de date sur le site, il faudra pas oublier que certaines dates n'ont que
# mois+année, voire que année. Les inclure systématiquement si l'année/le mois entre dans le range
# ou alors faire un range par année uniquement ?


# In[22]:


def find_and_normalize_date_in_filename(cote):
    """
    Using the original log_df, find the date of a given cote written in the filename of the first page of the cote
    """
    
    should_be_name = log_df[log_df.cote == cote]["should_be_name"].iloc[0]
    
    end_date = -8 # "_XXX.jpg" has length 8
    
    begin_date = re.search(cote,should_be_name).span()[1]
    
    date = should_be_name[begin_date:end_date]
    
    date_match = re.search("\d\d\d\d-\d\d-\d\d", date)
    if date_match is not None:
        year, month, day = date_match.group().split("-")
        day_num, month_num, year_num = int(day), int(month), int(year)
        return day_num, month_num, year_num, "filename_date"

    date_match = re.search("\d\d\d\d-\d\d", date)
    if date_match is not None:
        year, month = date_match.group().split("-")
        month_num, year_num = int(month), int(year)
        return None, month_num, year_num, "filename_date"
    
    date_match = re.search("\d\d\d\d", date)
    if date_match is not None:
        year = date_match.group()
        year_num = int(year)
        return None, None, year_num, "filename_date"
    
    else:
        return None, None, None, "no_match"
    


# In[ ]:





# ## Lieux
# Souvent lié à la date, mais plus ou moins éloigné. On peut rencontrer des formules comme "Fait à Sion, en séance du Grand Conseil, le X mai YYYY" ou "Fait à Genève à l'hotel de la prefecture".

# In[23]:


# Utiliser la liste de lieux trouvés pour les dates + Les résultats de "Fait à"


# In[24]:


def find_lieu_in_date(date_text):
    """
    Find a location name in the given date_text
    """
    if date_text is None:
        return None
    
    
    # Pour affiner un peu la recherche, on va isoler le lieu comme premier élément de la str à analyser.
    lieu_and_date = search_for_pattern(lieu_full_date_regex, date_text)
    
    if lieu_and_date is None:
        lieu_and_date = search_for_pattern(lieu_partial_date_regex, date_text)
        
    if lieu_and_date is not None:
        lieu = retrieve_unique_pattern("^"+make_re_group(lieux),lieu_and_date.group())
        return lieu
    else:
        return None


# In[25]:


def normalize_lieu(lieu_text):
    """
    Normalize a lieu name
    lieu_text : a string containing the lieu
    """
    if pd.isnull(lieu_text):
        return lieu_text
    
    lieu_text = lieu_text.lower()
    
    # On fait la correspondance uniquement avec des mots en minuscule, par simplification
    # Utiliser les dictionnaires juste là svp.
    
    lieu = retrieve_unique_pattern(make_re_group(lieux),lieu_text)
    
    corrected_lieu = correct_lieux_dict.get(lieu, lieu)
    
    return corrected_lieu


# In[26]:


def find_lieu_in_full_text(text):
    """
    Try to find a lieu in a full text if the lieu+date method didn't give anything, using precise formulae.
    """
    
    # Formules : Fait à, Adopté à... Reprendre celles d'en haut, je suppose ?
    optional_chambre_regex = "( en " + make_re_group(chambres_pour_formules_preliminaires) + ")?"
    formule_lieu = make_re_group(formules_preliminaires) + optional_chambre_regex +     " a " + make_re_group(lieux)
    
    match_list = find_pattern(formule_lieu,text)
    if len(match_list) > 0:    
        return match_list, "Fait_a+lieu_seul"
    
    # Impression
    regex_lieu_impression = make_re_group([make_re_group(lieux) + " ((-)+ )?" + "imprimerie",
                                           "a "+make_re_group(lieux)+" chez"])
    match_list = find_pattern(regex_lieu_impression,text)
    if len(match_list) > 0:
        return match_list, "lieu_impression"
    
    else:
        return list(), "no_match"


# In[ ]:





# ## Intitulés
# L'intitulé est le "titre" du document. C'est ce par quoi commence le document. Par opposition au "type" présumé du document, l'intitulé est orienté pour être lisible par l'humain. Il est une forme de résumé du document. On s'attend à ce qu'un certain nombre de ces intitulés soient faux, incomplets ou bruités, mais s'ils donnent une certaine information à l'humain qui lirait la base de donnée, c'est déjà bien.
# 
# **Logique de captation de l'intitulé ; Séparation intitulé VS émetteur :**
# 
# Classiquement, un document commence par l'intitulé, l'émetteur, ou rien du tout. Parfois, l'intitulé est situé sous l'émetteur. L'émetteur et l'intitulé commencent de la même façon, c'est-à-dire qu'ils commencent par une ligne qui n'est pas une ligne de continuation, et s'enchainent ensuite de lignes de continuation ou de connecteurs. Les lignes de continuation sont des lignes qui commencent par un connecteur, ou des lignes précédées par un connecteur. Dès lors, un simple processus greedy linéaire peut assimiler toute une entité *intitulé* ou *émetteur* (ou *paire émetteur-récepteur*). Il suffit donc de classer la première ligne comme émetteur ou comme intitulé, puis de refaire le processus sur les lignes suivantes jusqu'à ce qu'un intitulé ait été trouvé.
# 
# Particularités des intitulés:
# * Dans les livrets, l'intitulé prend toute la première page. Ca serait bien de reconnaitre quand c'est le cas. Indices: beaucoup de majuscules sur la première page ; peu de texte, bien moins que sur les pages suivantes ; répétition partielle de la page de titre dans la première page de texte.
# * Chercher des mots clefs comme *"portant", "du/des", "à/aux", "sur", "et", "au sujet de", "concernant", "modifiant", "par", "relatif/ve à", "contre", "qui" ou "pour"* dans les premières lignes de texte peut être une bonne façon d'isoler les intitulés. Les sauts de ligne et potentiellement la taille d'alto sont importantes ici ! Typiquement, les lignes paires vont comporter les mots "du", "sur", etc.. et les lignes impaires les infos importantes, ou alors toutes les lignes dès la seconde vont commencer par un de ces mots.
# * "prononcé par" est aussi une liaison possible.
# * les participes présents comme portant... ordonnant... sont aussi des liaisons possibles.
# * Si la dernière ligne d'un intitulé se termine par le/la, alors prendre aussi la ligne suivante.
# * Décret du... sur...
# * Il faut à tout prix utiliser une liste de mots-clefs car certains mots sont perdus au milieu du document, notamment dans les structures vides (PERMIS DE SÉJOUR, etc) ou les termes "ARRÊTE", "DÉCRÈTE", ... quand le document commence par l'énonciateur. Là encore, connaître la hauteur des lignes correspondantes dans l'alto pourrait être très utile. ->
# * Si l'intitulé est suivi de l'émetteur quelques lignes plus bas, on pourrait peut-être supposer que l'intitulé s'étant jusqu'à l'émetteur, sauf si une phrase comme "Sur la proposition du conseil..." précède l'émetteur.
# * Une page de titre tout en majuscule pourrait être un intitulé valide ?
# * Il faut une règle pour éliminer les intitulés qui sont en fait des émetteurs

# In[27]:


non_intitule_when_head_starts = ["le grand conseil","le conseil","republique","liberte","le departement",
                                "ministere", "canton", "au nom", "(circulaire )?"+lieu_full_date_regex,
                                "la chambre","administration", "departement", "une et indivisible",
                                 "de la republique", "ponts et chaussees", "helvetique", "empire", "ndeg",
                                "egalite", "le comite", "le conseiller", "le grand-conseil", "l'administration",
                                "lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]

non_intitule_starting_regex = "^"+make_re_group(non_intitule_when_head_starts)
connecting_expr = ["[a-z]+ant","du","des","a","aux","au","sur","et","au sujet de","par","relatif a","relative a",
                  "contre","qui","pour","prononce par"]
ending_only_connecting_expr = ["le","la","les"]

false_positive_intitule_starts = ["en execution", "informe", "avise", "vu", "chers", "monsieur", "messieurs",
                                 "tres-honores", "nous", "rend public"]
false_positive_regex = "^"+make_re_group(false_positive_intitule_starts)


# In[28]:


def greedy_demarkate_and_remove_connected_expr(lines):
    """
    Remove of lines all the elements belonging to the first connected expr of lines and return it.
    lines' first elements are removed in place.
    """
    
    stop_here_flag = False
    
    result = []

    while (not stop_here_flag) and len(lines) > 0:
        result.append(lines.pop(0))

        if search_for_pattern(
            make_re_group(connecting_expr + ending_only_connecting_expr)+"$",result[-1]
        ) is None and (
            len(lines) > 0 and search_for_pattern("^"+make_re_group(connecting_expr),lines[0]) is None):
            stop_here_flag = True
    
    return result
                
    

def demarkate_intitule(text):
    """
    Try to extract the intitule (title of the document) from the text (with linebreaks)
    
    ! Pas utilisable en l'état pour les compilations car ne retourne pas une liste.
    """
    
    lines = text.split("\n")
    
    intitule_head = lines[0]
    
    while search_for_pattern(non_intitule_starting_regex, intitule_head) is not None:
        unwanted_lines = greedy_demarkate_and_remove_connected_expr(lines)
        intitule_head = lines[0]
    
    intitule_lines = greedy_demarkate_and_remove_connected_expr(lines)
    
    # Elimination des intitulés qui sont en fait la première ligne de texte (il n'y a alors aucun intitulé)
    if len(intitule_lines) > 0 and search_for_pattern(false_positive_regex, intitule_lines[0]) is not None:
        intitule = pd.NA
    else:
        intitule = " ".join(intitule_lines)
    
    if intitule is "":
        intitule = pd.NA
    
    return intitule


# ## Imprimeries
# Particularités des imprimeries:
# * Notées Imprimerie de, Imprimerie X, Imprim., Imp.(?) Nat., Imp. et lith., voire "Chez... " Commencer par faire une recherche pour isoler le nom des principaux imprimeurs, puis chercher directement ces noms, car les imprimeurs ne sont pour ainsi dire jamais mentionnés pour d'autres raisons que d'être les imprimeurs du papier en question.
# * Parfois "De l'Imprimerie de"
# * Appelée occasionnellement "Typographie" ou "Typographia"
# * Parfois noté "Imprimé chez..."
# * Parfois "Imprimé à (Sion,Paris,etc) , chez..."
# * L'imprimeur est généralement à la fin. On utilisera donc decide_end

# In[29]:


imprimerie_heading = ["(l|i)mprimeri(e|s) (des citoyens )?", "imprim ","imp( nat)? ","imp et lith ",
                      "tyopgraphie ","typographia ",
                      "imprime chez ", "(imprime )?(a )?{}( |\n)chez ".format(make_re_group(lieux)),
                      "imprime dans notre "]
# Rajouter le "chez" seul mais uniquement avec la liste prête. On le fera donc à la fin.
#Uniformiser : rajouter le de,des,d' après une seule fois.

imprimerie_transition = "(de |des |d')?"

vrais_imprimeurs_nom = ["henri vincent", "emanuel hortin", "hignou", "antoine advocat",
                       "andre fischer et luc vincent", "michel leroy", "luc sestie", "gauthier pere et fils",
                       "hignou aine", "etienne ganioz", "calpini et holdermann", "emanuel vincent fils",
                       "freres blanchard", "samuel delisle", "ch gruaz", "calpini-albertazzi", "l schmid",
                       "marius olive", "a morand", "schmid et murmann", "louis advocat", "sylvain theubet",
                       "vaney", "etienne ganioz", "haller", "monaldi", "corbaz et robellaz", "pache",
                       "p -v oursel", "p meyll", "delisle", "david rachor", "chr fischer", "ferdinand penon",
                       "marc ducloux", "ganioz", "ceresole et panizza", "gay et steinbach", "p -a donnant",
                       "edouard laederich", "ch steinbach", "corbaz et rouiller fils", "a larpin",
                       "attinger", "rieder et simmen", "p dubois", "f aymon", "georges bridel", "dufour",
                       "schmid", "k -j wyss", "vincent", "j schmid", "dulex-ansermoz", "michel vester",
                       "gueffier", "coesnonpellerin", "gruner gessner", "joseph louis salzmann"]

vrais_imprimeurs_titre = ["imprimerie imperiale", "imprimerie nationale", "imprimerie des citoyens",
                         "imprimerie de la republique", "imprimerie du gouvernement", "imprimerie de l'echo",
                         "imprimerie de la nation suisse", "imprimerie roiale"]

## Dictionnaires pour la normalization des imprimeurs
coquille_imprimeurs = {"art advocat":"antoine advocat",
                      "antoine anvocat":"antoine advocat","calpini-albertazzl":"calpini-albertazzi",
                      "catpini-albertazzi":"calpini-albertazzi", "a abborat":"antoine advocat"}

uniformize_imprimeurs = {"imprim nat":"imprimerie nationale","imp nat":"imprimerie nationale",
                        "henri em vincent":"henri vincent", "imprim national":"imprimerie nationale",
                        "imprim nation":"imprimerie nationale",
                        "imprim du gouvernement":"imprimerie du gouvernement",
                        "ant advocat":"antoine advocat","imprim du gouvern":
                        "imprimerie du gouvernement", "s delisle":"samuel delisle","calbini-albertazzi":
                        "calpini-albertazzi", "a advocat":"antoine advocat",
                        "emm vincent fils":"emanuel vincent fils", "l advocat":"louis advocat",
                        "d rachor":"david rachor", "e laederich":"edouard laederich", "le citoyen coesnonpellerin":
                        "coesnonpellerin"}

normalize_imprimeurs_dict = dict()
normalize_imprimeurs_dict.update(coquille_imprimeurs)
normalize_imprimeurs_dict.update(uniformize_imprimeurs)

## Listes de tous les match valides pour le regex
tous_imprimeurs_nom = sorted(vrais_imprimeurs_nom + [k for k in coquille_imprimeurs
                                              if coquille_imprimeurs[k] in vrais_imprimeurs_nom] +
[k for k in uniformize_imprimeurs if uniformize_imprimeurs[k] in vrais_imprimeurs_nom],
                             key = len)

tous_imprimeurs_titre = sorted(vrais_imprimeurs_titre + [k for k in coquille_imprimeurs
                                                 if coquille_imprimeurs[k] in vrais_imprimeurs_titre] +
[k for k in uniformize_imprimeurs if uniformize_imprimeurs[k] in vrais_imprimeurs_titre],
                               key = len)

## Creation des regex
imprimerie_nom_regex = make_re_group([make_re_group(imprimerie_heading) + imprimerie_transition +
                                  make_re_group(tous_imprimeurs_nom), "chez "+ make_re_group(tous_imprimeurs_nom)
                                      +" (imprime(ur|rie)|impr(im)?)"])

imprimerie_titre_regex = make_re_group(tous_imprimeurs_titre)

imprimerie_nom_uniquement_regex = make_re_group(tous_imprimeurs_nom) #Gardé ou pas selon la qualité des résultats

# Autoriser aussi la recherche de lieux grace aux imprimeurs !
# formule : <lieu> -* imprimerie ou a <lieu> chez ... imprimeur/imprimerie


# In[30]:


def find_imprimeur(text):
    """
    Find the imprimeur (printer) name in a document text
    """
    
    match_list = find_pattern(imprimerie_nom_regex, text)
    if len(match_list) > 0:
        return match_list, "nom_imprimeur"
    
    match_list = find_pattern(imprimerie_titre_regex, text)
    if len(match_list) > 0:
        return match_list, "titre_imprimerie"
    
    else:
        return list(), "no_match"
    
    return match_list


# In[31]:


def normalize_imprimeur(imprim_text, code):
    
    if pd.isnull(imprim_text):
        return imprim_text
    
    imprim_text = imprim_text.lower()
    
    if code == "nom_imprimeur":
        imprimeur = retrieve_unique_pattern(make_re_group(tous_imprimeurs_nom), imprim_text)
    elif code == "titre_imprimerie":
        imprimeur = retrieve_unique_pattern(make_re_group(tous_imprimeurs_titre), imprim_text)
    else:
        imprimeur = imprim_text
        
    norm_imprimeur = normalize_imprimeurs_dict.get(imprimeur, imprimeur)
    
    return norm_imprimeur


# ## Thème
# On testera la LDA pour le thème. Toutefois, certains mots-clefs sont très caractéristiques, et reviennent quasi systématiquement dans des contextes précis. On peut donc associer ces mots-clef à des thêmes.
# *collecte, vaccination, maladie, épizootie, ban, fanfare, forestier, école, régent, sage-femme, militaire, chemin de fer, correction du Rhône, incendie, bains, thermale, Loëche, finances, heimatlos(-/es/at), concours, collèges, lycée, surlangue, fusion des communes, poids et mesures, système métrique*. Penser au pluriel possible. Le thème par mot-clef devra être exclusif, c'est-à-dire que si deux mots de deux thèmes différents se retrouvent dans le document, on n'osera pas en choisir un de préférence à l'autre. Des mots comme "finance" sont dès lors délicats. Séparer entre mots exclusifs et mots non-exclusifs ? (Si on trouve 1 exclusif et plusieurs non-exclusifs, l'exclusif l'emporte, comme dans "fanfare" + "militaire", mais si on trouve plusieurs exclusifs, on est probablement dans un Rapport plus général ou un document de ce genre.
# 
# On essaiera de nommer les sujets de la LDA pour les faire correspondre aux sujets mot-clef. Ainsi, si la LDA et les mots-clef donnent le même résultat, on fusionnera.

# In[32]:


# Dans la generation des textes, creer un dataframe separe [cote,tokens] qui serve de base avec laquelle travailler.


# In[33]:


# 60 topics, lb = 40, ub = 1/8
topics_names_0_9 = ["armée","catastrophe",
                   "débits de marchandise","x", "x",
              "x","bourgeoisie/naturalisation","x","x","rivières"]
topics_names_10_19 = ["salaire","x","chemin de fer",
                    "répression judiciaire","poids et mesures",
                    "x","guerre","banque","état civil",
                    "habillement"]
topics_names_20_29 = ["République Helvétique","impôts","x",
                    "Rhône","x","x","bétail","x","école","x"]
topics_names_30_39 = ["transactions financiaires", "x","x",
                     "gendarmerie", "constitution", "Confédération",
                     "x","voyageurs","concours", "soldats"]
topics_names_40_49 = ["x","lettre/proclamation","patriotisme","forêts",
                     "routes","x","état civil","dixains",
                     "commerce des denrées","x"]
topics_names_50_59 = ["x","tribunaux","monnaie","armée","assemblée",
                     "budget", "charité", "x", "patentes", "x"]
topics_names = topics_names_0_9 + topics_names_10_19 + topics_names_20_29 + topics_names_30_39 +topics_names_40_49 + topics_names_50_59


# In[34]:


if PERFORM_TOPIC_MODELLING:
    
    topic_modelling_useful_df = topic_modelling_df[(topic_modelling_df["is_compil"] == False) &
                                                  ~pd.isnull(topic_modelling_df["tokens"])].reset_index()
    
    lda, lda_proba, tokens_index = Tutil.create_lda_model_and_stuff(topic_modelling_useful_df, 60, 40, 1/8)
    
    #Tutil.print_top_words(lda, tokens_index, 12)
    
    assert lda_proba.shape[0] == len(topic_modelling_useful_df)
    
    topic_modelling_useful_df["3_best_topics"] = pd.NA
    for i in range(lda_proba.shape[0]):
        topic_modelling_useful_df.at[i,"3_best_topics"] = Tutil.choose_most_probable_topics(
            lda_proba[i,:], topics_names, 3)
    
    topic_modelling_useful_df.to_csv("topic_modelling_subjects.csv")

else:
    
    topic_modelling_useful_df = pd.read_csv("topic_modelling_subjects.csv")
    topic_modelling_useful_df["3_best_topics"] = topic_modelling_useful_df["3_best_topics"].apply(
    lambda l : eval(l) if not pd.isnull(l) else l)


# In[35]:


keyword_exclusive_topics = {"collecte":"charité","vaccination":"vaccination","epizootie":"épizooties",
                  "ban":"ban","fanfare":"fanfare","forestier":"forêts","regent":"école",
                  "sage-femme":"sage-femmes","chemin de fer":"chemin de fer",
                  "correction du rhone":"Rhône","incendie":"incendie", "baigneurs":"bains thermaux",
                  "heimatlos":"heimatlosat", "heimatloses":"heimatlosat","concours":"concours",
                  "college":"enseignement supérieur","lycee":"enseignement supérieur",
                  "université":"enseignement supérieur","surlangue":"épizooties",
                   "fusion des communes":"fusion de communes","poids et mesures":"poids et mesures",
                  "systeme metrique":"poids et mesures", "heimatlosat":"heimatlosat",
                           "inondations":"catastrophe","sonderbund":"Sonderbund",
                           "comite de martigny":"Comité de Martigny"}

keyword_indication_topics = {"maladie":"santé","ecole":"école", "militaire":"affaires militaires",
                   "finances":"affaires financières","bains":"bains thermaux",
                   "tribunal":"affaires judiciaires","tribunaux":"affaires judiciaires"}


# In[36]:


def guess_3_most_probable_topics(text, cote):
    """
    Determine the 3 most probable topics using first the text for at most one topic,
    then the topic modellig results
    Return : a set of length 0 to 3
    """
    
    possible_topics_keywords = list(map(lambda m_obj : m_obj.group().lower(),
                               find_pattern(make_re_group(list(keyword_exclusive_topics.keys())), text)))
    
    type_topics = "no_topics"
    i = 0
    chosen_topics = set()
    while i < len(possible_topics_keywords) and len(chosen_topics) < 3:
        chosen_topics.add(keyword_exclusive_topics[possible_topics_keywords[i]])
        i += 1
    
    if len(chosen_topics) > 0:
        type_topics = "exclusive_keywords_only"
    
    if len(chosen_topics) < 3:
        type_topics = "keywords"
        possible_topics_keywords = list(map(lambda m_obj : m_obj.group().lower(),
                               find_pattern(make_re_group(list(keyword_indication_topics.keys())), text)))
        
        i = 0
        while i < len(possible_topics_keywords) and len(chosen_topics) < 3:
            chosen_topics.add(keyword_indication_topics[possible_topics_keywords[i]])
            i += 1
        
    if len(chosen_topics) < 3:
        if len(chosen_topics) == 0:
            type_topics = "lda_only"
        else:
            type_topics = "keywords_and_lda"
        lda_topics = topic_modelling_useful_df[topic_modelling_useful_df["cote"] == cote]["3_best_topics"].iloc[0]
        
        chosen_topics = chosen_topics.union(set(lda_topics[:3-len(chosen_topics)]))
    
    return list(chosen_topics), type_topics


# In[ ]:





# In[ ]:





# ## Signataire
# Il faudra premièrement distinguer les institutions et les noms propres. Par exemple, certains documents sont seulement signés "Le Grand-Baillif" ou "La Chancellerie d'Etat" tandis que d'autres sont signées "Le Grand-Baillif \n de Courten". 
# Formules clef comme "Au nom de"
# Les signatures sont particulièrement sujettes à des déchets entre le titre du signataire et son nom, comme (signé), (L S) sous toutes ses formes, CL S), etc. Tout n'a pas été corrigé parmi ces déchets ! Il pourrait donc être utile de recourir à un test de similarité pour ne retenir que les mots communs de différentes signatures, comme (L S) DE COURTEN et DE COURTEN se verrait réduites à DE COURTEN.
# Les signatures admettent également des variantes orthographiques à considérer.
# Formule "Le conseiller d'état chargé du département... X".
# *Le chef du département, Le conseiller d'état chargé du département, le président du ...* Dans la plupart des cas, la signature prend une à trois lignes.
# Rarement, le titre suit le nom au lieu de le précédé.
# Parfois, on voit le nom précédé du mot-clef "signé". Pour détecter au mieux, chercher les mots clefs comme Le Chancelier/Président/Secrétaire(s)/Vice-président adjoint, etc. puis chercher l'existence d'un "de". Si oui, chercher à la ligne suivante, sinon, chercher plutôt sur la même ligne (mais c'est pas forcément un indicateur absolu) Si la date est de type "Donné à ... + date", chercher les signatures après (? Peut-être que cette condition est inutile). Commencer par composer une liste de noms cherchés à partir des fonctions, puis faire une recherche inversée à partir des noms pour trouver là où ils ne sont pas reliés à une fonction (Pour combler les vides. On se contentera typiquement de ceux en fin de document ou sur une ligne seule)
# 
# Il existe des signatures constituées juste d'un nom et d'un nom de famille, ou d'une lettre et d'un nom de famille. Ici, les retours de ligne et les majuscules peuvent nous être très utile.
# Une formule type "une lettre + évent. De + un nom inconnu" est quasiment sure de récupérer un nom propre si tous les noms commencent par une majuscule ou sont entièrement en majuscule.

# In[37]:


## Liste des éléments qui précèdent le début de la signature (avant "le <titre>")
precede_signature = ["\n","\d\d\d\d ","\nPar ","\nL S "]

## Liste de titres pour les signatures avec noms propres
signataire_titre = ["conseiller d'etat","grand baillif", "grand-baillif", "chef",
                    "president", "conseiller national", "depute",
                   "conseiller fédéral", "prince", "prefet", "general", "ministre",
                   "eveque", "bourguemaitre", "vice-president", "colonel",
                   "commissaire(s)? federa(ux|l)? dans le canton du valais", "vice-baillif",
                   "lieutenant", "ingenieur", "forestier", "intendant", "1er vice-president", "sous-prefet"]

## Liste de compléments aux titres pour les signatures avec noms propres
signataire_extra_titre = ["provisoire","adjoint","substitue", "cantonal", "en chef", "general"]

## préposition d'introduction à la fonction précise (peut suivre un titre ou une institution)
signataire_intro_fonction = ["du ","des ","de ","d'"]

## Liste d'institutions et de complément aux institutions, pour les signatures d'institution uniquement
signataire_institution = ["chancellerie", "departement", "comite", "conseil","inspection", "commission"]
signataire_institution_extra = ["federal(e)?","national(e)?","cantonal(e)?", "en chef"]
# En general sur une seule ligne, pas de nom propre

## Liste de titres pour les signatures des secrétaires
signataire_secretaire = ["secretaire", "chancelier", "secretaires"]
# Est-ce que le chancelier est parfois un titre plutôt qu'un secretaire ?

## Liste de formules de transitions parfois présentes entre le titre et la fonction précise
# elles peuvent parfois ajouter une ligne de plus aux signatures, pour un total de 3 lignes.
signataire_transition = ["(\n| )charge du departement(( de| des)? [a-z']+)?","(\n| )et de la legion d'honneur"]

## Liste de formules supplémentaires qui précèdent parfois le nom propre dans une signature.
signataire_extra = ["CLS","CL S","L S", "signe"]

## Liste de faux positifs qui se substituen en noms propres dans les signatures
faux_positif_nom = ["en son absence"]

## Regex potentiellement sensible à la casse qui délimite un nom propre
nom_regex = "\\b[A-Z]+[a-z]*\\b" # CE REGEX EST SENSIBLE A LA CASSE ! N'utiliser que pour isoler le nom

## Liste de séparateurs entre deux mots dans un nom propre
inter_nom = [" ","'","' ","-"," -","'' "]

## Regex pour délimiter le nom propre complet (en plusieurs mots) d'une signature
nom_complet_regex = "(" + make_re_group([nom_regex,"\\bde\\b"]) + make_re_group(inter_nom
    ) + ")*(" + nom_regex + ")"

def generate_grande_signature_regex(signataire_titre):
    """
    Create a regex for a grande signature using the appropriate titres (main signatory, or secretary)
    signataire_titre : the list of usable titres (president, conseiller national, secretaire...) for the signature
    Return : (grande_signature_regex, titre_fonction_regex)
    grande_signature_regex : the full regex to find a grande signature
    titre_fonction_regex : the heading of the grande signature, which can sometimes be a signature as a whole,
    like the signature_institution.
    """
    ## Regex englobant le titre et la fonction d'un signataire
    titre_fonction_regex = "(?P<fonction>" + make_re_group(
        precede_signature) + "l(e |es |')" + make_re_group(signataire_titre) + "( " + make_re_group(
        signataire_extra_titre) + ")?(" + make_re_group(signataire_transition) +")?( " + make_re_group(
        signataire_intro_fonction) + "[^\n]+)?" + ")"

    ## Regex pour une signature au format "le <titre> (de <fonction>) \n <nom propre>"" et ses variantes
    grande_signature_regex = titre_fonction_regex + "( |\n)" +        "(" + make_re_group(signataire_extra) + " )?" + "(?P<nom>[^\n]+)"
    
    return grande_signature_regex, titre_fonction_regex

## Regex pour une signature au format "le/la <institution> (de <fonction>)"
signature_institution_regex = "(?P<fonction>" + make_re_group(
    precede_signature) + "l(e |a |es |')" + make_re_group(signataire_institution) \
+ "( " + make_re_group(signataire_institution_extra) + ")?" \
+ "( " + make_re_group(signataire_intro_fonction) + "[^\n]+)?)"

def generate_petite_signature_regex(signataire_titre):
    """
    Create a regex for a petite signature using the appropriate titres (main signatory, or secretary)
    signataire_titre : the list of usable titres (president, conseiller national, secretaire...) for the signature
    """
    ## Regex pour une signature au format "<nom propre> <titre>"
    petite_signature_regex = "(?P<nom>" + make_re_group(
        precede_signature) + "[^\n]+) (?P<fonction>" + make_re_group(signataire_titre) + ")(\n|$)"
    
    return petite_signature_regex

## Regex pour retrouver un nom propre seul sur une ligne (! utiliser uniquement en case-sensitive)
#TODO

# Choix avec decide_end


# In[38]:


def find_signature(text, mode):
    """
    Find the main signatures of a given text and return them as a match object list
    text : the text in which to search
    mode : the type of signature to look for. Must be "principal" or "secretary"
    Return : (match_object_list, code)
    match_object_list : the list of match object containing the signatures
    code : the type of signature
    """
    
    if mode == "principal":
        grande_signature_regex, titre_fonction_regex = generate_grande_signature_regex(signataire_titre)
        petite_signature_regex = generate_petite_signature_regex(signataire_titre)
    elif mode == "secretary":
        grande_signature_regex, titre_fonction_regex = generate_grande_signature_regex(signataire_secretaire)
        petite_signature_regex = generate_petite_signature_regex(signataire_secretaire)
    else:
        raise Exception("Unvalid mode. Must be 'principal' or 'secretary'.")
    
    
    # Signature complète sur plusieurs lignes
    match_list = find_pattern(grande_signature_regex, text, allow_part_of_word = True)
    match_list = [m_obj for m_obj in match_list if m_obj.group("nom").lower() not in faux_positif_nom]
    if len(match_list) > 0:
        return match_list, "grande_signature"
 

    # Signature de l'institution uniquement ou du titre du signataire uniquement, sans nom propre
    if mode == "principal":        
        signature_institution_ou_fonction = make_re_group( #trick : (?P<>...)|(?P<>...) --> (?<>...|...)
            [titre_fonction_regex,signature_institution_regex]).replace(")|(?P<fonction>","|")
        match_list = find_pattern(signature_institution_ou_fonction, text,
                                 allow_part_of_word = True)
        if len(match_list) > 0:
            return match_list, "signature_institution"
        
    
    # Signature réduite sur une seule ligne (vient apres signature institution, sinon prendrait "le" comme nom.)
    match_list = find_pattern(petite_signature_regex, text, allow_part_of_word = True)
    if len(match_list) > 0:
        return match_list, "petite signature"
    
    # Signature avec nom propre uniquement
    #if mode == "principal":
        #TODO si nécessaire
        
    return list(), "no_match"


# In[39]:


# Choix avec decide_end en mode match_objet complet. Si nom propre uniquement, choix d'abord avec decide_maj ?


# In[40]:


# Normaliser la signature : utiliser les tags fonction et nom ; remplacer les \n par des espaces, puis strip
def normalize_signature(signature_match_obj, signature_type_code):
    """
    Extract the name, if any, and the function, if any, of the signatory in the given signature.
    signature_match_obj : the match object containing the signature to be processed
    signature_type_code : the type of the signature, which describe its format.
    """
    
    if pd.isnull(signature_match_obj):
        return None, None
    
    if signature_type_code in ["grande_signature", "petite_signature", "nom_uniquement"]:
        
        try:
            nom = signature_match_obj.group("nom")
            nom = re.sub("^"+make_re_group(precede_signature)," ",nom)
            nom = nom.strip()
        except :
            raise Exception(str(signature_match_obj) + " " + signature_type_code)
    else:
        nom = None
    
    if signature_type_code in ["grande_signature", "petite_signature", "signature_institution"]:
        fonction = signature_match_obj.group("fonction")
        fonction = re.sub("^"+make_re_group(precede_signature)," ",fonction)
        fonction = fonction.strip()
    else:
        fonction = None
    
    return fonction, nom


# ## Type
# Une partie des types peut être déterminé par des mots-clefs comme "Décrète", "Arrête", "Informe", "Fait ordre", ou "Ordonne". Ces mots suivent un sujet comme LE CONSEIL D'ÉTAT, mais pas forcément immédiatement après. Ils sont souvent en majuscule, mais pas toujours. Si cette classe de priorité est choisie, il faut prendre celui le plus proche du début du document, pas le plus proche d'une extrémité. En effet, beaucoup de documents sont suivis par un court paragraphe "Le conseil Ordonne la publication du présent décret...", alors que ce ne sont pas des ordonnances. Ces mots permettent souvent de lier les premières lignes du document à la désignation de l'émetteur.
# *Extrait, Loi, Décret, Arrêté, Rapport, Budget, Publication, Proclamation, Instructions, taxe, permis de séjour/chasse/pêche, certificat, lettre, règlement, projet,tractanda, concordat, ordonnance, Suplément, appel...*
# Chercher en priorité les mots qui sont présents sur une seule ligne.
# 
# Chercher en priorité les mots qui sont des noms (loi, etc...) à ceux qui sont des verbes, car parfois le verbe ne correspond pas au nom (par exemple "loi" mais "arrête")

# In[41]:


category_names = ["Extrait","Loi","Decret","Rapport","Budget","Publication","Proclamation","Instruction(s)?","Taxe",
                 "permis de sejour","permis de chasse","permis de peche","certificat","lettre","Reglement",
                 "Tractanda","Concordat","Ordonnance","Supplement","Appel","Circulaire","Arrete","Signalement",
                 "Etat de paye","Ordre", "Proces verbal", "Etat sommaire", "Depouillement", "Registre"]

# checker ceux qui sont en début de ligne. Decider avec decide_begin

category_verbs = {"decrete":"decret","arrete":"arrete","informe":"information",
                  "fait ordre":"ordre","ordonne":"ordre"}

# Checker ceux qui occupent une ligne seule

precede_name = ["Projet d(e |')"]

category_names_transform_dict = {"instructions":"instruction"}

category_transform_dict = dict()
category_transform_dict.update(category_names_transform_dict)
category_transform_dict.update(category_verbs)

category_names_regex = "(^|\n)" + make_re_group(precede_name) + "?" + make_re_group(category_names)
category_verbs_regex = "\n" + make_re_group(category_verbs.keys()) + "(nt)?" + "\n"


# In[42]:


def find_category(text):
    
    match_list = find_pattern(category_names_regex, text)
    if len(match_list) > 0:
        return match_list, "name"
    
    match_list = find_pattern(category_verbs_regex, text)
    if len(match_list) > 0:
        return match_list, "verb"
    else:
        return list(), "no_match"


# In[43]:


def normalize_category(category_text):
    
    if pd.isnull(category_text):
        return category_text
    
    category_text = category_text.lower()
    
    category_text = category_text.strip("\n")
    
    category = category_transform_dict.get(category_text, category_text)
    
    return category


# ## Nombre d'images
# Le nombre de pages du document n'est pas important. En revanche, on a besoin du nombre d'images par cote.
# 
# Comme cette partie concerne tous les documents, pas seulement ceux en francais, on va le faire directement ici et pas dans les fonctions fetch_metadata.

# In[44]:


metadata_cote_df["nombre_image"] = metadata_cote_df["raw_img_filenames"].apply(len)


# ## Taille des images
# Il faut juste récupérer la taille des images.
# 
# Comme cette partie concerne tous les documents, pas seulement ceux en francais, on va le faire directement ici et pas dans les fonctions fetch_metadata.

# In[45]:


def get_image_size(img_filename,img_directory_path):
    """
    Return the size of the specified image
    img_filename : the name of the image file
    img_directory_path : the path to the directory where the image is
    Return : (width, height)
    width : the width in pixel of the image
    height : the height in pixel of the image
    """
    
    img = PIL.Image.open(img_directory_path+img_filename)
    
    return img.size


# In[ ]:





# In[ ]:





# In[46]:


if PERFORM_LONG_OPERATIONS:
    metadata_img_df["width"] = pd.NA
    metadata_img_df["height"] = pd.NA
    
    for i, row in tqdm(metadata_img_df.iterrows()):
        metadata_img_df.at[i, "width"], metadata_img_df.at[i,"height"] = get_image_size(
            row["raw_img_filenames"], IV_util.one_SAVE_raw_path)
    
    metadata_img_df[["width","height"]].to_csv("images_size.csv")

else:
    img_size_df = pd.read_csv("images_size.csv",index_col=0)
    metadata_img_df["width"] = img_size_df["width"]
    metadata_img_df["height"] = img_size_df["height"]


# ## Emetteur / Destinataire
# Moins important. Chercher pour les formulées "Adressé(es) à(ux) ... par" ou "Du..." ou "Le conseil... au conseil..."
# Le destinataire typiquement n'est vraiment pas très utile, mais facile à repérer avec "à .../aux"

# ## Remplissage des colonnes et stats pour les documents simples
# C'est dans cette section qu'on va effectivement charger les fichiers textes, chercher toutes les métadata, remplir les cellules du tableau et dresser des stats pour les documents hors-compilation

# In[47]:


def fetch_metadata_simple_doc(text, cote, mode):
    """
    Retrieve all metadata from a given non-compilation text and return them as a dictionary
    text : the single document text to be processed
    cote : the cote of the current text
    mode : "simple" or "compilation". If "compilation", topic modelling is not used for subjects (not available)
    Return : a dict
    """
    metadata = dict()
        
    # Recherche de la date et du lieu
    text_for_date_search = re.sub("\n"," ",text) 
    
    date_match_list, date_liability_code = find_most_probable_dates(text_for_date_search)
    metadata["type_date"] = date_liability_code
        
    chosen_date_text = decide_extrem(date_match_list,len(text_for_date_search))
    
    if not pd.isnull(chosen_date_text):
        day_num, month_num, year_num = normalize_date(chosen_date_text)
        metadata["jour"] = day_num
        metadata["mois"] = month_num
        metadata["annee"] = year_num

        ## Partie dédiée aux lieux
        lieu = find_lieu_in_date(chosen_date_text)
        lieu = normalize_lieu(lieu)
        if pd.isnull(lieu):
            
            lieu_match_list, lieu_code = find_lieu_in_full_text(text_for_date_search)
            chosen_lieu_text = decide_extrem(lieu_match_list, len(text_for_date_search))
            lieu = normalize_lieu(chosen_lieu_text)
            if not pd.isnull(lieu):
                metadata["type_lieu"] = lieu_code
            else:
                metadata["type_lieu"] = "no_match"
        else:
            metadata["type_lieu"] = "lieu_avec_date"
        
        metadata["lieu"] = lieu
        
    else:
        metadata["type_lieu"] = "no_match"
        
    # Recherche complémentaire de la date dans les noms de fichier si aucune date n'a été trouvée plus haut
    if metadata["type_date"] == "no_match":
        day_num, month_num, year_num, new_date_liability_code = find_and_normalize_date_in_filename(cote)
        
        metadata["jour"] = day_num
        metadata["mois"] = month_num
        metadata["annee"] = year_num
        metadata["type_date"] = new_date_liability_code
        
        
    # Recherche de l'intitulé
    intitule = demarkate_intitule(text)    
    metadata["intitule"] = intitule
    
    
    # Recherche de l'imprimeur
    imprimeur_match_list, imprimeur_code = find_imprimeur(text)
    chosen_imprimeur_text = decide_end(imprimeur_match_list)
    imprimeur = normalize_imprimeur(chosen_imprimeur_text, imprimeur_code)
    metadata["imprimeur"] = imprimeur
    metadata["type_imprimeur"] = imprimeur_code
    
    
    # Recherche du signataire principal
    signataire_match_list, signataire_code = find_signature(text, "principal")
    chosen_signataire_match_obj = decide_end(signataire_match_list, return_whole_match_object = True)
    signataire_fonction, signataire_nom = normalize_signature(chosen_signataire_match_obj, signataire_code)
    metadata["signataire_nom"] = signataire_nom
    metadata["signataire_titre"] = signataire_fonction
    metadata["type_signataire"] = signataire_code
    
 
    # Recherche du secrétaire signataire
    secretaire_match_list, secretaire_code = find_signature(text, "secretary")
    chosen_secretaire_match_obj = decide_end(secretaire_match_list, return_whole_match_object = True)
    secretaire_fonction, secretaire_nom = normalize_signature(chosen_secretaire_match_obj, secretaire_code)
    metadata["secretaire_nom"] = secretaire_nom
    metadata["secretaire_titre"] = secretaire_fonction
    metadata["type_secretaire"] = secretaire_code
    
    
    # Recherche des sujets
    if mode == "simple":
        topics, topics_code = guess_3_most_probable_topics(text, cote)
        if len(topics) > 0:
            metadata["sujet_1"] = topics[0]
        if len(topics) > 1:
            metadata["sujet_2"] = topics[1]
        if len(topics) > 2:
            metadata["sujet_3"] = topics[2]
        metadata["type_sujet"] = topics_code
    
    
    # Recherche du type/catégorie
    category_match_list, category_code = find_category(text)
    category_text = decide_begin(category_match_list)
    category = normalize_category(category_text)
    metadata["categorie"] = category
    metadata["type_categorie"] = category_code
    
    return metadata


# In[48]:


def display_stats(metadata_cote_df, metadata_name, metadata_liability = None):
    """
    Display the filling statistics of a given metadata for non-compile documents.
    metadata_name : the name of the column in the df for the given metadata
    metadata_liability : the name of the column in the df for the liability codes for the given metadata
    """
    staty_df = metadata_cote_df[~pd.isnull(metadata_cote_df["fr_interm_txt_filenames"]) & 
                               ~metadata_cote_df["is_compil"]]
    
    result_column_name = "% rempli pour "+metadata_name
    result_liability_column_name = "nombre pour "+metadata_name
    
    stats_global_df = staty_df.groupby(["interest"]).count()
    stats_global_df[result_column_name] = stats_global_df[metadata_name]/stats_global_df["fr_interm_txt_filenames"]
    
    display(stats_global_df[result_column_name].round(decimals=3).reset_index())
    
    
    if metadata_liability is not None:
        stats_liability_df = metadata_cote_df.groupby(["interest",metadata_liability]).count()
        stats_liability_df[result_liability_column_name] = stats_liability_df["fr_interm_txt_filenames"]

        display(stats_liability_df[result_liability_column_name].reset_index())


# In[ ]:





# In[ ]:





# ## Gestion des compilations
# Dans la mesure où les métadata ont une forte interactions entre elles dans les compilations, il paraît plus judicieux de traiter les compilations totalement à part.
# 
# Dans un premier temps, tenter de séparer les compilations ordonnées des compilations dégénérées. Les compilations ordonnées sont les extraits de loi, qui comportent généralement des signatures et des dates. Les compilations dégénérées sont plutôt celles dans lesquelles on retrouvera une seule ou aucune date, et une seule ou aucune signature. Pour ces dernières, on va supposer qu'il y a un document par page. S'il n'y a qu'une page, on va chercher des métadata qui apparaissent par deux.
# 
# Ensuite, pour chaque document, on va dans un premier temps chercher quelles métadata sont pertinentes pour établir des "frontières", puis on va sectionner le texte selon les frontières établies, puis relancer la recherche classique de métadatas sur chaque sous-texte. Les frontières seront certainement établies en fonction de plus d'une métadonnée.

# Après quelques essais, le critère le plus robuste est peut-être bien celui des regroupements de signatures, qui indiquent la toute fin du document, et ne bougent pas contrairement à la date qui peut être au début ou à la fin du document, où le type qui n'est pas toujours présent.
# 
# Penser aux cas où des informations manquent ! Par exemple aucune signature (rare). Ou alors une seule signature de
# type Président + secretaire. En fait, si la liste de signataire principale n'a qu'une seule signature, on ne peut pas l'utiliser. Prendre donc la liste la plus grande entre decide_maj sur les catégories (important !), les dates et les signatures. Regrouper seulement dans le cas des signatures, car s'il y a des dates multiples mais pas les signatures qui vont avec, alors forcément chaque date se rapporte à un document différent. Si toutes les listes sont de taille 1 ou inférieur, ne pas séparer (inutile de créer plein de compilations vides de toute métadata). Si toutes les listes sont de taille 2 ou inférieur, séparer par liste (seule différence concerne les signatures)
# 
# S'il n'y a que deux signatures, les considérer comme deux différentes, car on sait que c'est une compilation, donc il y a au minimum deux document différents. Donc pas oublier le >=
# 
# 
# En cas d'égalité, utiliser plutôt les signatures. En cas d'égalité entre dates et catégories, plutôt utiliser les dates. Donc catégories autorisées seulement si > date et > signature. date autorisée seulement si > signature. Signature dans tous les autres cas.
# 
# On dirait que ça marche pas mal du tout !

# In[49]:


def distances_match_obj(match_obj_list):
    """
    Return the distances between objects of the match_obj_list
    match_obj_list : a list of match objects
    Return : float
    """
    
    begin_matches_array = np.array(sorted(list(map(lambda obj : obj.span()[0], match_obj_list))))
    end_matches_array = np.array(sorted(list(map(lambda obj : obj.span()[1], match_obj_list))))
    
    distances = begin_matches_array[1:] - end_matches_array[:-1]

    return distances


# In[50]:


def compare_match_obj(a,b):
    
    if a.span()[0] > b.span()[0] and a.span()[1] > b.span()[1]:
        return 1
    elif a.span()[0] < b.span()[0] and a.span()[1] < b.span()[1]:
        return -1
    else:
        return 0


# In[51]:


def fetch_metadata_compilation_doc(text, cote):
    """
    Retrieve all metadata from a given compilated text and return them as a
    list of dictionaries (one dictionary per sub-document in the document)
    First, split the cote text into sub-documents, then apply fetch_metadata_simple_doc to them
    Return : a list of dicts
    
    Le choix des frontières n'utilise ni le lieu, ni l'intitulé, ni l'imprimeur, ni les sujets,
    car il n'ont pas de sens sur un texte compilé.
    
    La variable "boundaries" contient le 1er caractère de la frontière de chaque document
    
    En général, on élimine le premier ou le dernier élément de la liste des matches qui déterminent
    les frontières car le premier (respectivement le dernier) doc commence non pas à la première
    frontière, mais au début du document (respectivement ne se termine pas à la dernière,
    mais à la fin du document)
    """
    
    
    text_for_date_search = re.sub("\n"," ",text)
    
    date_match_list, date_liability_code = find_most_probable_dates(text_for_date_search)
    signataire_match_list, signataire_code = find_signature(text, "principal")
    secretaire_match_list, secretaire_code = find_signature(text, "secretary")
    category_match_list, category_code = find_category(text)
    
    # Tri supplémentaire, car les catégories dans une compilations ont généralement
    # toutes le même nombre de majusucles. Elimine des faux-positifs
    category_match_list = decide_maj(category_match_list)
    
    num_dates = len(date_match_list)
    num_signataires = len(signataire_match_list)
    num_categories = len(category_match_list)
    
    
    all_signature_list = signataire_match_list + secretaire_match_list
    date_match_list = sorted(date_match_list, key = cmp_to_key(compare_match_obj))
    all_signature_list = sorted(all_signature_list, key = cmp_to_key(compare_match_obj))
    category_match_list = sorted(category_match_list, key = cmp_to_key(compare_match_obj))
    
    
    if num_dates <= 1 and num_signataires <= 1 and num_categories <= 1:
        boundaries = [] 
        
    # A partir d'ici, on sait que la catégorie choisie a au moins une taille 2
        
    elif num_categories > num_signataires and num_categories > num_dates: # Use categories for boundaries    
        begin_categories = list(map(lambda obj : obj.span()[0], category_match_list))        
        boundaries = begin_categories[1:]

        
    elif num_dates > num_signataires: # Use dates for boundaries
        distance_textStart_firstDate = date_match_list[0].span()[0]
        distance_textEnd_lastDate = len(text) - date_match_list[-1].span()[1]
        
        if distance_textStart_firstDate < distance_textEnd_lastDate:
            begin_dates = list(map(lambda obj : obj.span()[0], date_match_list))
            boundaries = begin_dates[1:]
        else:
            end_dates = list(map(lambda obj : obj.span()[1], date_match_list))
            boundaries = end_dates[:-1]
  

    else: # Use signatures for boundaries
        distances = distances_match_obj(all_signature_list)
        mean_distance = distances.mean()
        end_signatures = list(map(lambda obj : obj.span()[1], all_signature_list))
        
        assert len(distances) == len(end_signatures[:-1])
        
        boundaries = []
        i = 0
        while i < len(distances):
            if distances[i] >= mean_distance:
                boundaries.append(end_signatures[i])
            i += 1
    
    
    slices = zip([None] + boundaries, boundaries + [None])
    
    sub_texts = [text[t[0]:t[1]] for t in slices]
    # Ca a l'air de marcher ! Plus qu'à split les textes et appliquer fetch_simple
    
    metadatas = [fetch_metadata_simple_doc(txt, cote, "compilation") for txt in sub_texts]
    
    return metadatas


# In[ ]:





# ## Exécution

# In[52]:


metadata_compilation_df = pd.DataFrame()


# In[53]:


for i, row in tqdm(metadata_cote_df.iterrows()):
    
    try:
        ### Traitement des documents simples
        if not pd.isnull(row["fr_interm_txt_filenames"]) and not row["is_compil"]:
            
            with open(IV_util.four_TEMP_intermediate+row["fr_interm_txt_filenames"], encoding = "utf8") as f:
                text = f.read()
            
            metadata_dict = fetch_metadata_simple_doc(
                text, IV_util.get_cote(row["fr_interm_txt_filenames"]), "simple")

            for k in metadata_dict:
                metadata_to_add = metadata_dict[k]
                if not pd.isnull(metadata_to_add): # Adding a null value may set a column type to float64. Avoid !
                    metadata_cote_df.at[i,k] = metadata_to_add
        
        
        ### Traitement des compilations
        elif not pd.isnull(row["fr_interm_txt_filenames"]) and row["is_compil"]:
            
            cote = IV_util.get_cote(row["fr_interm_txt_filenames"])
            
            with open(IV_util.four_TEMP_intermediate+row["fr_interm_txt_filenames"], encoding = "utf8") as h:
                text = h.read()
                
            metadata_dicts = fetch_metadata_compilation_doc(text, cote)
            
            for i, sub_dict in enumerate(metadata_dicts):
                sub_dict["cote"] = cote
                sub_dict["sous_document"] = str(i+1)
                metadata_compilation_df = metadata_compilation_df.append(sub_dict, ignore_index = True)
            
    except Exception as e:
            raise Exception(str(e) + " (cote {})".format(i))


# In[54]:


display_stats(metadata_cote_df,"annee","type_date")


# In[55]:


display_stats(metadata_cote_df,"lieu","type_lieu")


# In[56]:


display_stats(metadata_cote_df,"imprimeur","type_imprimeur")


# In[57]:


display_stats(metadata_cote_df,"signataire_titre","type_signataire")
# D'après mes calculs, on est à environ 50% de vrais document sans signature et
# 50% d'erreurs dans les documents classés sans match.
# Cela représente donc moins de 10% d'erreur au total.


# In[58]:


display_stats(metadata_cote_df,"secretaire_nom","type_secretaire")


# In[59]:


display_stats(metadata_cote_df,"sujet_1","type_sujet")


# In[60]:


display_stats(metadata_cote_df,"categorie","type_categorie")


# In[61]:


metadata_cote_df.sample(10)


# In[62]:


metadata_cote_df[(metadata_cote_df.type_secretaire == "no_match") & 
                 (metadata_cote_df.interest == 4)].sample(10)[[
    "signataire_titre","signataire_nom", "type_signataire","secretaire_titre",
                            "secretaire_nom","type_secretaire"]]


# In[63]:


metadata_compilation_df.head(15)


# ## Sauvegarde
# On supprime les colonnes "internes" (inutiles pour la base de donnée)
# Rajouter des colonnes "langue", "structure vide", "manuscrit" à la place des states ?
# 
# ### Metadata images
# * Supprimer "has_roman_txt" qui fait doublon avec la présence -ou non- de texte
# * Remplacer les states par sept colonnes booléennes (4 langues, sans texte, structures, et manuscrit)
# * dupliquer roman_txt_filenames en fr, it, la txt_filenames (doublons de texte, mais tant pis, c'est plus clair
# * Supprimer roman_binary_img_filenames
# * Remplacer les filenames des textes par les textes eux-mêmes
# 
# ### Metadata cotes
# * Supprimer fr_interm_txt_filenames et raw_img_filenames (mieux vaut éviter une colonne-liste dans une DB)
# * Renomer is_compil en français clair
# * Supprimer tous les types
# * Remplacer les states par cinq/six colonnes booléennes (pas sans-texte, pas manuscrit non plus (?))
# 
# ### Metadata compilation
# * Supprimer tous les types

# In[64]:


def read_text(filename, directory_path):

    with open(directory_path + filename, "r", encoding = "utf8") as f:
        return f.read()


# In[65]:


if PERFORM_FINAL_SAVE == True:
    # Création des colonnes booléennes des états
    metadata_img_df["lang_fr"] = metadata_img_df["states"].apply(lambda s : "f" in s)
    metadata_img_df["lang_la"] = metadata_img_df["states"].apply(lambda s : "l" in s)
    metadata_img_df["lang_it"] = metadata_img_df["states"].apply(lambda s : "t" in s)
    metadata_img_df["lang_de"] = metadata_img_df["states"].apply(lambda s : "a" in s)
    metadata_img_df["lang_sans_texte"] = metadata_img_df["states"].apply(lambda s : "b" in s)
    metadata_img_df["manuscrit"] = metadata_img_df["states"].apply(lambda s : "m" in s)
    metadata_img_df["structure_vide"] = metadata_img_df["states"].apply(lambda s : "v" in s)

    # Création des textes
    for i, row in tqdm(metadata_img_df.iterrows()):
        
        if not pd.isnull(row["roman_txt_filenames"]):
            metadata_img_df.at[i,"texte_original_roman"] = read_text(
                row["roman_txt_filenames"],IV_util.four_SAVE_texts)

    metadata_img_df["texte_original_fr"] = metadata_img_df["texte_original_roman"].where(metadata_img_df["lang_fr"])
    metadata_img_df["texte_original_la"] = metadata_img_df["texte_original_roman"].where(metadata_img_df["lang_la"])
    metadata_img_df["texte_original_it"] = metadata_img_df["texte_original_roman"].where(metadata_img_df["lang_it"])
    
    for i, row in tqdm(metadata_img_df.iterrows()):
       
        if not pd.isnull(row["fr_word_seq_filenames"]):
            metadata_img_df.at[i,"words_sequence_fr"] = read_text(
                row["fr_word_seq_filenames"], IV_util.four_SAVE_word_sequences)
            
            
        if not pd.isnull(row["la_word_seq_filenames"]):
            metadata_img_df.at[i, "words_sequence_la"] = read_text(
                row["la_word_seq_filenames"], IV_util.four_SAVE_word_sequences)
            
        if not pd.isnull(row["it_word_seq_filenames"]):
            metadata_img_df.at[i, "words_sequence_it"] = read_text(
                row["it_word_seq_filenames"], IV_util.four_SAVE_word_sequences)

    # Renommage
    metadata_img_df = metadata_img_df.rename(columns = {"raw_img_filenames":"image_filename"})
    
    # Suppression des colonnes en trop
    metadata_img_df = metadata_img_df.drop(columns = ["has_roman_txt", "roman_binary_img_filenames", "states",
                                                     "texte_original_roman","roman_txt_filenames",
                                                     "fr_word_seq_filenames","it_word_seq_filenames",
                                                     "la_word_seq_filenames"])
    
    # Séparation de l'ID de l'image et de son extension
    metadata_img_df["file_extension"] = "jpg"
    metadata_img_df["image_filename"] = metadata_img_df["image_filename"].apply(lambda txt : txt.split(".")[0])


# In[66]:


metadata_img_df.head()


# In[ ]:





# In[67]:


if PERFORM_FINAL_SAVE:
    
    metadata_cote_df["lang_fr"] = metadata_cote_df["states"].apply(lambda s : "f" in s)
    metadata_cote_df["lang_de"] = metadata_cote_df["states"].apply(lambda s : "a" in s)
    metadata_cote_df["lang_it"] = metadata_cote_df["states"].apply(lambda s : "t" in s)
    metadata_cote_df["lang_la"] = metadata_cote_df["states"].apply(lambda s : "l" in s)
    
    metadata_cote_df["structure_vide"] = metadata_cote_df["states"].apply(lambda s : "v" in s)
    
    metadata_cote_df = metadata_cote_df.rename(columns = {"is_compil":"compilation","interest":"interet"})
    
    metadata_cote_df = metadata_cote_df.drop(columns = ["fr_interm_txt_filenames","states",
                                                       "type_date","type_lieu","type_imprimeur",
                                                       "type_signataire","type_secretaire","type_sujet",
                                                       "type_categorie", "raw_img_filenames"])


# In[68]:


metadata_cote_df.head()


# In[69]:


if PERFORM_FINAL_SAVE:
    
    metadata_compilation_df = metadata_compilation_df.drop(columns = ["type_categorie","type_date",
                                                                      "type_imprimeur", "type_lieu",
                                                                      "type_secretaire", "type_signataire"])


# In[70]:


metadata_compilation_df.head()


# #### Post-processing
# Initialement, ce travail a été fait juste avant la création des manifestes
# Comme on s'est rendu compte qu'il pouvait être bien utile pour la base de donnée aussi,
# on le met là.
# 
# (Si par hasard il devait ne pas fonctionner ici, sachez qu'il n'a jamais été testé à cet emplacement du code)

# In[ ]:


if PERFORM_FINAL_SAVE
    # Les sujets sont propres à chaque document. On les ajoute ici pour des raisons de cohérence
    metadata_compilation_df["sujet_1"] = pd.NA
    metadata_compilation_df["sujet_2"] = pd.NA
    metadata_compilation_df["sujet_3"] = pd.NA

    # C'est plus naturel d'écrire les lieux avec des majuscules
    capitalize_place_dict = {"palais de l'elysee":"Palais de l'Elysée","st-gall":"St-Gall","st-maurice":"St-Maurice",
                        "palais des tuileries":"Palais des Tuileries",
                         "palais de saint cloud":"Palais de Saint Cloud","lons-le-saunier":"Lons-le-Saunier",
                        "brougg argovie":"Brougg Argovie",
                        "bale":"Bâle","geneve":"Genève","genes":"Gênes","viege":"Viège","neuchatel":"Neuchâtel"}

    metadata_cote_df["lieu"] = metadata_cote_df["lieu"].apply(lambda l : capitalize_place_dict.get(l, l.capitalize()
                                                                                              if not pd.isnull(l)
                                                                                              else l))
    metadata_compilation_df["lieu"] = metadata_compilation_df["lieu"].apply(
    lambda l : capitalize_place_dict.get(l, l.capitalize() if not pd.isnull(l) else l))

    # Idem pour les types
    capitalize_category_dict = {"arrete":"Arrêté","decret":"Décret","proces verbal":"Procès verbal",
                           "supplement":"Supplément","reglement":"Règlement","projet d'arrete":
                           "Projet d'Arrêté","depouillement":"Dépouillement","projet de decret":
                           "Projet de décret","projet de reglement":"Projet de règlement",
                           "ordonnent":"Ordre"}

    metadata_cote_df["categorie"] = metadata_cote_df["categorie"].apply(
    lambda c : capitalize_category_dict.get(c, c.capitalize()) if not pd.isnull(c) else c)
    metadata_compilation_df["categorie"] = metadata_cote_df["categorie"].apply(
    lambda c : capitalize_category_dict.get(c, c.capitalize()) if not pd.isnull(c) else c)

    # Comme il n'y a pas de cote bis dans les compilations, pandas a automatiquement converti les cotes en int64 !
    metadata_compilation_df["cote"] = metadata_compilation_df["cote"].apply(str)

    # Certaines dates dans les noms de fichier sont fantaisistes.
    metadata_cote_df["mois"] = metadata_cote_df["mois"].apply(lambda m : m if m > 0 and m < 13 else np.nan)
    metadata_compilation_df["mois"] = metadata_compilation_df["mois"].apply(
    lambda m : m if m > 0 and m < 13 else np.nan)
    
    # Certaines colonnes ont été transformées en float alors que ce sont des int. On retransforme.
    metadata_cote_df = metadata_cote_df.astype(
        {"interet":pd.Int64Dtype(),"jour":pd.Int64Dtype(),"mois":pd.Int64Dtype(),
                        "annee":pd.Int64Dtype()})
    metadata_compilation_df = metadata_compilation_df.astype({"jour":pd.Int64Dtype(),"mois":pd.Int64Dtype(),
                                                         "annee":pd.Int64Dtype()})


# In[71]:


if PERFORM_FINAL_SAVE:
    metadata_img_df.to_csv("metadata_image.csv",encoding = "utf8",index = False)
    metadata_cote_df.to_csv("metadata_cote.csv",encoding = "utf8")
    metadata_compilation_df.to_csv("metadata_compilation.csv",encoding = "utf8", index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[77]:





# In[ ]:




