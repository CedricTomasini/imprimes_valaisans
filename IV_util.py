# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

import re
import PIL.Image
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import unicodedata


### Static variables

## Paths
log_csv_path = "log.csv"

angle_csv_path = "angle.csv"

model_path_fr = "kraken/model_best_cedric_v1.mlmodel"

zero_to_select_path = "0_TEMP_raw_images_to_select/"

one_SAVE_raw_path = "1_SAVE_raw_images/"
one_garbage_path = "1_raw_images_not_chosen/"

#two_TEMP_color_equal_path = "2_1_TEMP_equalized_images/"
#two_TEMP_binary_path = "2_2_TEMP_binary_images/"
#two_TEMP_deskewed_path = "2_3_TEMP_binary_images_deskewed/"
#two_TEMP_segments_path = "2_4_TEMP_segments/"
#two_SAVE_alto_path = "2_SAVE_alto/"

two_TEMP_francais_transcription = "2_TEMP_fr_transcription/"
two_TEMP_deutsch_transcription = "2_TEMP_de_transcription/"
two_TEMP_no_transcription = "2_TEMP_no_transcription/"
two_SAVE_binary_images = "2_SAVE_binary_images/"

#three_SAVE_texts_path = "3_SAVE_texts/"

#three_TEMP_segments = "3_TEMP_segments/"
#three_SAVE_alto = "3_SAVE_alto/"
three_TEMP_left_right_texts = "3_TEMP_lr_texts/"

four_SAVE_texts = "4_SAVE_texts/"
four_TEMP_intermediate = "4_TEMP_intermediate_processed_texts/"
four_SAVE_word_sequences = "4_SAVE_word_sequences/"

six_SAVE_manifests = "6_SAVE_IIIF_manifests/"


### Selection step

class CodeConstants():
    """
    This static class contains informations about the code that map a keypress to a decision.
    It can be change from here if needed and the script will adapt,
    as long as the changes are consistants.
    """
        
    def __init__(self):
        
        #Letter code with its translation in french
        self.code = {"f":"en français","a":"en allemand","l":"en latin","t":"en italien",
            "v":"structure vide","p":"pas la peine","m":"manuscrit","b":"sans texte","d":"Doublon",
            "x":"n'existe plus","r":"à renommer dans Z:","s":"spécial"} 
        
        # When yes is decided but no state has been chosen, this state is chosen
        self.default_yes_state = "f"
        
        # Key that denote a language
        self.lang_keys = ["f","a","l","t"]
        
        # When a state in this list is chosen but no decision has been taken, yes is taken.
        # For other states, no is taken.
        self.yes_binding = self.lang_keys+["m","b","r"] 
        
        # Keys that signal duplicates
        self.doublon_keys = ["d"]
        
        # Default key to signal a duplicate
        self.doublon_default_key = "d"
        
        # Default key to signal that the file needs to be renamed in Z:
        self.renaming_default_key = "r" 
        
        # Key for special case (no decision is taken for special cases, this means we need manual processing.)
        self.special_case_key = "s" 
        
        # Default key to signal that the physical cote do not exists anymore
        self.suppressed_default_key = "x"



### CSV interaction

def open_log_csv(path = log_csv_path):
    """
    Wrapper that open the log_csv and perform all needed operations to make it readable.
    """
    
    complex_columns = ["states","binary_files"]
    str_columns = ["cote","doublon"]
    
    df = pd.read_csv(path,sep = "\t",encoding = "utf8")
    
    # Column formatting
    for col in df.columns:
        if col in complex_columns:
            df[col] = df[col].apply(lambda x : eval(x) if not pd.isnull(x) else x)
        elif col in str_columns:
            df[col] = df[col].apply(lambda e: e if pd.isnull(e) else str(e))
    
    df = df.convert_dtypes()
    
    return df


def save_log_csv(df, path = log_csv_path):
    """
    Wrapper that save the log_df to the log_csv and perform all needed operations to make it reopenable
    """
    df = df.convert_dtypes()
    df.to_csv(path, sep = "\t", encoding = "utf8", index = False)


### Filename interaction    
    
def get_cote(filename):
    """
    Get the full cote out of a filename
    Constraints:
    - the cote must be the first number to appear in the filename.
    -  the cote extension must be lowercase letters without accentuations.
    
    filename: str, the name of the file
    return : str
    """
    full_cote = re.search("\d+[a-z]*",filename).group()
    cote_num = str(int(re.search("^\d+",full_cote).group()))
    cote_ext_match = re.search("[a-z]+$",full_cote)
    if cote_ext_match:
        cote_ext = cote_ext_match.group()
    else:
        cote_ext = ""
    
    return cote_num+cote_ext


def get_page(filename,file_ext):
    """
    Get the page number of a file that has the given extension
    """
    
    suffix = re.search("\d\d\d{}".format(file_ext),filename).group()
    page_num = re.search("\d\d\d",suffix).group()
    return page_num


def get_prefix(filename):
    """
    Get the name of the file without the extension
    """
    return filename.split(".")[0]
 
 
def check_valid_filename_format(filename,file_ext):
    """
    return True if the filename is valid, meaning that it has a cote and a page number that are findable.
    """
    match = re.search("\d+[a-z]*.*\d\d\d\{}".format(file_ext),filename)
    
    if match:
        return True
    else:
        return False


def rename(filename):
    """
    Rename a file according to our new conventions : CH-AEV_IV_<cote>_<pagenum(3digits)>.jpg
    """
    try:
        doc_cote = get_cote(filename)
        page_num_ext = re.search("\d\d\d\.jpg$",filename).group()
    
    except:
        raise Exception("{} could not be renamed".format(filename))
    
    else:
    
        new_name = "CH-AEV_IV_"+doc_cote+"_"+page_num_ext

        return new_name


### Images interaction

def load_and_resize_PIL_image(file_path, max_frame_size):
    """
    Load an image with PIL, resize it to fit in a square of size max_frame_size x max_frame_size
    and return the PIL object along with the resizing ratio
    """

    im = PIL.Image.open(file_path)
    width, height = im.size

    if height > width:
        im = im.resize((int(max_frame_size*width/height), max_frame_size), PIL.Image.ANTIALIAS)
        resizing_ratio = height/max_frame_size
    else:
        im = im.resize((max_frame_size, int(max_frame_size*height/width)), PIL.Image.ANTIALIAS)
        resizing_ratio = width/max_frame_size

    return im, resizing_ratio


### Alto interaction

def alto_to_txt(alto_path,output_path=None):
    """
    Transform an alto file into a plain text file.
    Also convert the text into NFC unicode normalization form, because Kraken use (probably) NFD
    
    alto_path : the path to the alto file
    output_path : the path to the txt file being created. If None, the function return the text instead
    
    UNUSED
    """
    
    min_large_space_ratio = 3
    
    # Allow two lines to overlap when one is on the top of the other.
    # See Segmentation_util lower_limit_margin_ratio to understand
    lower_margin_tolerance = 3/4 

    input_file = open(alto_path,"r",encoding = "utf8")
    alto = input_file.read()
    input_file.close()


    soup = BeautifulSoup(alto,"lxml-xml")

    text_lines = soup.findAll("TextLine")


    previous_VPOS = -1
    previous_HEIGHT = -1
    output_text_l = []

    for text_line in text_lines:

        current_VPOS = int(text_line["VPOS"])
        current_HEIGHT = int(text_line["HEIGHT"])
        
        if previous_VPOS > 0 and previous_HEIGHT > 0 and current_VPOS > previous_VPOS + previous_HEIGHT*lower_margin_tolerance:
            # We replace the last space at the end of a line if we can, to avoid having " "+"\n"
            if len(output_text_l) > 0 and output_text_l[-1] == " ":
                output_text_l[-1] = "\n"
            else:
                output_text_l.append("\n")
        
        # We add an extra linebreak if the space after the line is large.
        if previous_VPOS > 0 and previous_HEIGHT > 0 and current_VPOS > previous_VPOS + min_large_space_ratio*previous_HEIGHT:
            output_text_l.append("\n")
        
        for string in text_line.findAll("String"):
            output_text_l.append(string["CONTENT"])
            output_text_l.append(" ")
        
        previous_VPOS = current_VPOS
        previous_HEIGHT = current_HEIGHT

    # Normalizing the text to avoid accent issues later !
    NFD_text = "".join(output_text_l)
    NFC_text = unicodedata.normalize("NFC",NFD_text)
    
    if output_path is None:
        return NFC_text
    else:
        output_file = open(output_path,"w",encoding = "utf8")
        output_file.write(NFC_text)
        output_file.close()
    
    
### Controller actions

def construct_derived_name(log_df, action_ext, file_ext):
    """
    Construct a name from the official name of a file
    log_df : the dataFrame that contains the csv log info
    action_ext : the extension that denote the current action (_binary, _colorequal, etc...)
    file_ext : the file extension (.jpg, .tif, etc)
    
    UNUSED
    """
    
    return log_df.apply(lambda r : 
    get_prefix(r["official_name"])+action_ext+file_ext if r["needs_transcription"] else None,axis = 1)


def needs_processing_mask(log_df,next_col,next_path):
    """
    Create a mask that indicate which rows needs to be processed
    log_df : the dataFrame that contains the csv log info
    next_col : the column that will be processed
    next_path : the directory path where the processed files will be created
    
    UNUSED
    """
    
    # Pour gagner de la place, les fichiers temporaires (colorequal et image binaire sans deskewing)
    # sont potentiellement déplacées manuellement dès que le processing est arrivé au niveau des
    # images binaires avec deskewing et des segments.
    # Il faut donc éviter que ces images soient reprises si elles ont déjà été traitées à un moment.
    
    if next_col == "alto_name":
        return log_df[next_col].apply(lambda n : not os.path.exists(next_path+str(n))) & log_df["needs_transcription"]
    
    else:    
        return log_df[next_col].apply(lambda n : not os.path.exists(next_path+str(n))) & log_df["needs_transcription"] &\
    (log_df["deskewed_binary_name"].apply(lambda n : not os.path.exists(two_TEMP_deskewed_path+str(n))) |
     log_df["segments_name"].apply(lambda n : not os.path.exists(two_TEMP_segments_path+str(n))) )
    


    
    
### Degree of interest

degrees_of_interest = {"Structure vide (sans valeur)":1, "Structure vide (digne d'intérêt)":2,
                      "Document de faible intérêt":3, "Document commun":4, "Document significatif":5}

# Le degré d'intérêt est évalué pour répondre à la question suivante :
# A quel point un usage occasionnel du portail d'accès aux archives serait intéressé à tomber sur cette archive ?
# Il n'a aucune prétention scientifique.

# Structure vide (Sans valeur) = Document sans date qui ne porte aucune véritable information (eg : papier à lettre à en-tête)
# Structure vide (digne d'intérêt) = Document sans date commun (eg : formulaire)
# Document de faible intérêt = Document peu informatifs (eg : modification d'articles de loi, document partiels)
# Documment commun = Notation par défaut des document. Ordre d'idée: 80% des documents non-vides
# Document significatif = Document unique relié à des circonstances historiques particulières (eg : Déclaration de guerre)