#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# # Manifestes IIIF
# Le but de ce notebook est de générer les manifestes IIIF à partir des 3 csv de metadata qu'on a créé.

# In[1]:


import pandas as pd
import numpy as np
import IV_util
from tqdm import tqdm


# In[2]:


metadata_img_df = pd.read_csv("metadata_image.csv", encoding = "utf8")
metadata_cote_df = pd.read_csv("metadata_cote.csv", encoding = "utf8")
metadata_compilation_df = pd.read_csv("metadata_compilation.csv", encoding = "utf8")


# ### Fonctions génériques pour créer le JSON

# In[3]:


def stringize(obj):
    """
    Surround obj with quotation marks unless it is a string version of a dict or a list
    Eg. @value --> "@value" ; [{"label": "cote"}] --> [{"label":"cote"}] ; {"@id" : "x"} --> {"@id" : "x"}
    obj : the string to be analysed
    Return : str
    """
    
    obj = str(obj)
    
    if len(obj) > 0:
        if (obj[0] == "[" and obj[-1] == "]") or         (obj[0] == "{" and obj[-1] == "}") or         (obj[0] == '"' and obj[-1] == '"') or         obj.isdigit():
            return obj
        else:
            return '"' + obj + '"'
    else:
        return '"' + obj + '"'


# In[4]:


def dict_to_json(dictionary):
    """
    Transform a dictionnary into a string dictionary that can be integrated as a tile in a IIIF manifest.
    dictionary : the dict to be processed
    Return : str
    """
    
    lines_list = []
    
    for k in dictionary:
        indented_quoted_k = stringize(k).replace("\n","\n    ")
        indented_quoted_value = stringize(dictionary[k]).replace("\n","\n    ")
        lines_list.append("    " + indented_quoted_k + ": " + indented_quoted_value)
    
    lines = ",\n".join(lines_list)
    
    return "{\n" + lines + "\n}"
        


# In[5]:


def list_to_json(liste):
    """
    Transform a list into a string list that can be integrated as a tile in a IIIF manifest.
    liste : the list to be processed
    Return : str
    """
    indented_liste = list(map(lambda elem : "    " + stringize(elem).replace("\n","\n    "),liste))
    
    lines = ",\n".join(indented_liste)
    
    return "[\n" + lines + "\n]"


# ### Conversion des noms des métadonnées
# On convertit les noms des métadonnées orientés ordinateur en noms en français commun, et on les trie en fonction de quand elles interviennent.

# In[6]:


# Metadata qui seront toujours liées à la cote dans son ensemble
cote_metadata_mapper = {"cote":"Côte","interet":"Intérêt","compilation":"Compilation",
                        "nombre_image":"Nombre d'images", "structure_vide":"Structure vide"}

# Metadata qui seront toujours liées au document (1 par cote sauf pour les cotes compilées)
date_document_metadata = ["jour","mois","annee"] # Les dates sont traitées différement
document_metadata_mapper = {"lieu":"Lieu", "categorie":"Type", "intitule":"Intitulé", 
                            "signataire_nom":"Signataire principal","signataire_titre":
                            "Fonction du signataire principal","secretaire_nom":"Secrétaire signataire",
                            "secretaire_titre":"Fonction du secrétaire signataire",
                            "imprimeur":"Imprimeur",
                            "sujet_1":"Sujet 1","sujet_2":"Sujet 2","sujet_3":"Sujet 3"}

# Metadata qui apparaissent seulement dans les compilations
compil_only_metadata = ["sous_document"]

# Metadata qui sont toujours liées à une image précise (Ne seront pas tous affichés comme labels)
image_metadata_mapper = {"manuscrit":"Manuscrit","texte_original_fr":"Texte français",
                        "texte_original_la":"Texte latin","texte_original_it":"Texte italien"}
unprinted_image_metadata = ["cote","image_filename", "file_extension", "width","height", "structure_vide", 
                           "words_sequence_fr","words_sequence_la","words_sequence_it"] # Pas affichés comme labels
image_only_lang_metadata_mapper = {"lang_sans_texte":"sans texte"} # Les langues sont traitées séparément

# Les langues sont traitées à part. Elles apparaissent à la fois dans les cotes et dans les images
lang_metadata_mapper = {"lang_fr":"français","lang_de":"allemand","lang_it":"italien","lang_la":"latin"}


assert set(list(cote_metadata_mapper.keys()) + list(lang_metadata_mapper.keys())
           + list(document_metadata_mapper.keys()) + date_document_metadata) == set(metadata_cote_df.columns)
assert set(compil_only_metadata + date_document_metadata
           + list(document_metadata_mapper.keys()) + ["cote"]) == set(metadata_compilation_df.columns)
assert set(list(image_metadata_mapper.keys()) + unprinted_image_metadata
           + list(lang_metadata_mapper.keys())
                  + list(image_only_lang_metadata_mapper.keys())) == set(metadata_img_df.columns)

calendrier = {1:"janvier",2:"février",3:"mars",4:"avril",5:"mai",6:"juin",7:"juillet",8:"août",
             9:"septembre",10:"octobre",11:"novembre",12:"décembre"}


# ### Création du manifeste

# #### Fonctions de normalisation et transformation des données

# In[7]:


def normalize_metadata_value(val):
    """
    Transform the dataFrame value val into a human-friendly string
    val : the value to normalize
    key : the label of the value in the csv. Allow to perform some kind of special operations if needed
    Return : str
    
    Remplace les valeurs nulles par des tirets
    Remplace les booléens par des mentions "oui"/"non"
    """
    
    if pd.isnull(val):
        output =  "-"
    
    elif isinstance(val,bool) or isinstance(val,np.bool_):
        output =  "oui" if val else "non"
    
    elif isinstance(val,float) or isinstance(val,np.floating):
        output = str(int(val))
    
    else:
        output = str(val)
    
    # Les métadata sont toujours des string, même si ce sont des chiffres.
    # Cette ligne est obligatoire pour être compatible avec le comportement de stringize.
    if output.isdigit(): 
        output = '"' + output + '"'
        
    return output


# In[8]:


def generate_date_json(day_num, month_num, year_num):
    """
    Transform the numerical values of the date into a french string.
    The values correspond to the gregorian calender.
    day_num : int or null ; can be 0. The number of the day
    month_num : int or null ; 1 <= month_num <= 12. The number of the month. 1=janvier and 12=decembre
    year_num : int or null. The year number
    Return : str
    """
    
    date_list = []

    if not pd.isnull(day_num):
        date_list.append(str(int(day_num)))

    if not pd.isnull(month_num):
        date_list.append(calendrier[int(month_num)])

    if not pd.isnull(year_num):
        date_list.append(str(int(year_num)))

    if len(date_list) == 0:
        date = "-"
    else:
        date = " ".join(date_list)

    return date


# In[9]:


def generate_language_json(row, mode):
    """
    Transform the boolean language values into a nice french string
    row : the dataFrame row that contains the language informations
    mode : if "cote", can handle fr, de, la, it. If "image", can handle fr, de, la, it, sans_texte
    """
    lang_list = []
    
    for k in lang_metadata_mapper:
            if row[k] == True:
                lang_list.append(lang_metadata_mapper[k])
                
    if mode == "image": # Dictionnaire supplémentaire pour les images      
        for k in image_only_lang_metadata_mapper:
            if row[k] == True:
                lang_list.append(image_only_lang_metadata_mapper[k])
    elif mode != "cote":
        raise Exception("Invalid mode. Must be 'cote' or 'image'")
        
    if len(lang_list) == 0:
        return "-"
    else:
        return ", ".join(lang_list)


# #### Fonctions de génération du manifeste

# In[10]:


# Visiblement, en IIIF, on ne trouve pas des listes de listes, mais uniquement des listes de dictionnaires.
# Les dictionnaires contiennent des champs identificateurs dont certaines valeurs peuvent éventuellement
# être d'autres listes. On va donc reprendre ce principe.

def generate_compilation_metadata_json(cote, current_metadata_compil):
    """
    Generate the metadata of the sub-documents of a compilation
    cote : the current cote
    current_metadata_compil : the dataFrame which contains exactly the rows for the given cote.
    """
    
    sous_documents_list = []
    
    for i, row in current_metadata_compil.iterrows():
        
        local_metadata_list = []
        
        # Dates
        date = generate_date_json(row["jour"], row["mois"], row["annee"])        
        local_metadata_list.append(dict_to_json({
            "label" : "Date",
            "value" : date
        }))
        
        # Autre métadonnées
        for k in document_metadata_mapper:
            local_metadata_list.append(dict_to_json({
                "label" : document_metadata_mapper[k],
                "value" : normalize_metadata_value(row[k])
            }))
            
        local_metadata = list_to_json(local_metadata_list)
        
        # Identificateur de sous-document
        sous_documents_list.append(dict_to_json({
            "label" : "Sous-document " + str(row["sous_document"]),
            "metadata" : local_metadata
        }))
        
    return list_to_json(sous_documents_list)


# In[ ]:





# In[11]:


def generate_cote_metadata_json(row, cote, current_metadata_compil):
    """
    Generate the metadata specific to the row, with variation according to wether
    the cote is a compilation or not.
    """
    
    # Grâce aux assert, on sait que les dictionnaires sont complets.
    # On peut donc partir des dictionnaires pour parcourir les metadata à ajouter
    
    # Quand la metadata est nulle, mettre un tiret
    # Remplacer les booleens par des oui/non.
    # Regrouper les langues en liste
    
    is_compil = row["compilation"]
    
    metadata_list = []
    
    # Gestion des metadata de cote
    for k in cote_metadata_mapper:
        metadata_list.append(dict_to_json({
            "label" : cote_metadata_mapper[k],
            "value" : normalize_metadata_value(row[k])
        }))
    
    if is_compil == True: # "if np.nan:" resolve as np.nan was True !
        metadata_list.append(dict_to_json({
            "label" : "Contenu",
            "value" : generate_compilation_metadata_json(cote, current_metadata_compil)
        }))
        
    else:
        
        # Gestion des dates
        date = generate_date_json(row["jour"], row["mois"], row["annee"])
        
        metadata_list.append(dict_to_json({
            "label" : "Date",
            "value" : date
        }))
        
        # Gestion des métadata propres au document simple
        for k in document_metadata_mapper:
            metadata_list.append(dict_to_json({
                "label" : document_metadata_mapper[k],
                "value" : normalize_metadata_value(row[k])
            }))
            
        # Gestion des langues
        langues = generate_language_json(row, mode = "cote")
        metadata_list.append(dict_to_json({
            "label" : "Langue",
            "value" : langues
        }))
    
    
    metadata_json = list_to_json(metadata_list)
    
    return metadata_json


# In[ ]:





# In[ ]:





# In[12]:


def generate_image_metadata_json(row):
    """
    Generate the metadata specific to a given image
    row : the metadata_img_df row 
    Return : str
    """
    
    # Pas oublier ni la date, ni les langues !
    
    metadata_list = []
    
        
    # Gestion des métadata propres à chaque image
    for k in image_metadata_mapper:
        metadata_list.append(dict_to_json({
            "label" : image_metadata_mapper[k],
            "value" : normalize_metadata_value(row[k])
        }))

    # Gestion des langues
    langues = generate_language_json(row, mode = "image")
    metadata_list.append(dict_to_json({
        "label" : "Langue",
        "value" : langues
    }))
    
    metadata_json = list_to_json(metadata_list)
    
    return metadata_json


# In[13]:


def generate_image_sequence_json(current_metadata_img):
    """
    Generate the json sequence of images for a given cote
    current_metadata_img : the dataframe which contains exactly the rows of images of the given cote.
    Return : str
    """
    
    canvases_list = []
    for i, row in current_metadata_img.iterrows():
        
        image_name = row["image_filename"]
        image_primary_id = "https://imprimes.vallesiana.ch/iiif/2/" + image_name
                
        image_metadata = generate_image_metadata_json(row)
        
        ressources = dict_to_json({
            "@type": "dctypes:Image",
            "@id": image_primary_id + "/full/full/0/default" + "." + row["file_extension"],
            "service" : dict_to_json({
                "@context": "http://iiif.io/api/image/2/context.json",
                "@id": image_primary_id,
                "profile": "http://iiif.io/api/image/2/level2.json"
            })
        })
        
        current_image = dict_to_json({
            "@type" : "oa:Annotation",
            "motivation" : "sc:painting",
            "on" : "https://imprimes.vallesiana.ch/iiif/2/" + image_name + "/canvas/1",
            "resource" : ressources,
            "metadata" : image_metadata
        })
        
        current_canvas = dict_to_json({
            "@type": "sc:Canvas",
            "@id" : "https://imprimes.vallesiana.ch/iiif/2/" + image_name + "/canvas/1",
            "label" : "Page " + row["image_filename"][-3:],
            "width" : row["width"],
            "height" : row["height"],
            "images" : list_to_json([
                current_image
            ])
        })
        
        canvases_list.append(current_canvas)
    
    
    sequence = list_to_json([
        dict_to_json({
            "@type": "sc:Sequence",
            "viewingHint": "individuals/paged",
            "viewingDirection": "left-to-right",
            "canvases": list_to_json(canvases_list)
        })
    ])
    
    return sequence


# In[14]:


def create_manifest(metadata_cote_row):
    """
    Create the IIIF manifest for a given cote
    metadata_cote_row : a given row of the metadata_cote_df
    Return : str
    """
    
    # Variables générales
    row = metadata_cote_row
    cote = row["cote"]
    current_metadata_img = metadata_img_df[metadata_img_df["cote"] == cote]
    current_metadata_compil = metadata_compilation_df[metadata_compilation_df["cote"] == cote]
    
    # Données relatives à la cote
    cote_identifier = "CH-AEV_IV_" + cote
    
    cote_metadata = generate_cote_metadata_json(row, cote, current_metadata_compil)
    
    # Thumbnail (Prévisualisation. On affiche la première image du document)
    first_image_row = current_metadata_img[current_metadata_img["image_filename"] ==
                                          current_metadata_img["image_filename"].min()]
    
    first_image_name = first_image_row["image_filename"].item()
    first_image_width = first_image_row["width"].item()
    first_image_height = first_image_row["height"].item()
        
    thumbnail_data = dict_to_json({
        "@id" : "https://imprimes.vallesiana.ch/iiif/2/" + first_image_name + "/thumb/canvas/1",
        "@type" : "dctypes:images",
        "width" : str(first_image_width)+"px",
        "height" : str(first_image_height)+"px"
    })
    
    
    # Service   
    service_data = list_to_json([
        dict_to_json({
            "@context" : "http://iiif.io/api/search/1/context.json",
            "@id" : "",
            "profile" : "http://iiif.io/api/search/1/search",
            "label" : "Search within this thing",
            "service" : dict_to_json({
                "profile": "http://iiif.io/api/search/1/autocomplete",
                "label": "Get suggested words"
            })
        })
    ])
    
    # Séquence des images du document
    image_sequence = generate_image_sequence_json(current_metadata_img)  
    
    # Manifeste
    manifest = dict_to_json({
        "@context" : "http://iiif.io/api/presentation/2/context.json",
        "@type" : "sc:Manifest",
        "@id" : "https://imprimes.vallesiana.ch/manifest/" + cote_identifier + ".json",
        "label" : "archive",
        "description" : "Imprimés Valaisans XIVe-XXe siècles.",
        "metadata" : cote_metadata,
        "thumbnail" : thumbnail_data,
        "license" : "CC BY-NC 4.0<br>https://creativecommons.org/licenses/by-nc/4.0/",
        "service" : service_data,
        "attribution" : "Archives de l'Etat du Valais",
        "sequences" : image_sequence
    })
    
    
    return manifest


# In[ ]:





# In[16]:


for _, row in tqdm(metadata_cote_df.iterrows()):
    text = create_manifest(row)
    filename = "CH-AEV_IV_{}.json".format(row["cote"])
    
    with open(IV_util.six_SAVE_manifests+filename,"w",encoding = "utf8") as f:
        f.write(text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




