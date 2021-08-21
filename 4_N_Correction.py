#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Le but de ce notebook est de transformer les alto en textes, puis de générer une interface graphique pour les corriger.
# 
# Maintenir une colonne dans le log_csv pour indiquer quels textes viennent d'être créés et lesquels ont déjà été corrigés.
# 
# Faire de la correction automatique. Soit custom edit distance + dictionnaire créé au fur et à mesure, soit une méthode plus élaborée.

# In[ ]:


from bs4 import BeautifulSoup
import IV_util
import Correction_util as Cutil
import os
import re
import tkinter as tk
import PIL.ImageTk, PIL.Image
import numpy as np
import pandas as pd
import difflib


# ### Interface graphique
# Comme pour la V1, on va souligner les chiffres et les zones de majuscules. On va proposer de la correction automatique à l'aide de quelques règles simples et aussi souligner les zones corrigées. On va également proposer de la correction assistée à l'aide des mots corrigés précédement et de la librairie difflib
# 
# On va aussi prévoir une fonction de zoom pour l'image.
# 
# Pour la correction manuelle, trouver le moyen de permettre qu'une sélection approximative considère les mots en entier. ou alors faire mot à mot ? Bref, à voir pour que ça soit le plus ergonomique possible.
# 
# Pour la détection des mots corrigés en correction manuelle, la seule possibilité me semble de le faire après coup en comparant les deux textes, sinon on va se faire *** à mettre en place un tracking du curseur dans le texte.

# In[ ]:


class Model():
    
    def __init__(self,view,texts_path,original_images_path,max_frame_size):
        
        self.view = view
        
        self.max_frame_size = max_frame_size
        
        self.original_images_path = original_images_path
        
        self.texts_path = texts_path
        self.log_df = IV_util.open_log_csv()
        
        self.log_df["txt_name"] = self.log_df["official_name"].apply(lambda n : 
        IV_util.get_prefix(n)+"_fr.txt" if not pd.isnull(n) else n) 
        #CHANGER SI TRAITEMENT ALLEMAND OU SI UNIVERSALISATION DU MODULE
        
        """
        assert all(self.log_df[self.log_df["needs_transcription"] & self.log_df["states"].apply(
        lambda s : "f" in s or "l" in s or "t" in s)]["txt_name"].apply( #CHANGER SI TRAITEMENT ALLEMAND OU UNIFORM.
            lambda n : os.path.isfile(self.texts_path+n))),\
                   "Some files should be here according to the log csv but are missing."
        """
        
        # Ajouter ici la colonne "has_been_corrected" si elle n'existe pas
        if "has_been_corrected" not in self.log_df.columns:
            self.log_df["has_been_corrected"] = False
        else:
            self.log_df = self.log_df.fillna(value = {"has_been_corrected":False})
            
        # Ajouter ici la colonne "interest" si elle n'existe pas
        if "interest" not in self.log_df.columns:
            self.log_df["interest"] = pd.NA
            
        # Ajouter ici la colonne "compilation" si elle n'existe pas
        if "is_compil" not in self.log_df.columns:
            self.log_df["is_compil"] = pd.NA
        
        self.all_txt_df = self.log_df[self.log_df["needs_transcription"] & self.log_df["states"].apply(
        lambda s : "f" in s or "l" in s or "t" in s)]
        
        self.needs_transcription_indexes = self.log_df[self.log_df["needs_transcription"] &
                                                       self.log_df["states"].apply(
        lambda s : "f" in s or "l" in s or "t" in s)].index
        self.current_file_index_position = None      
        
        self.current_im = None
        self.no_zoom_im = None
        self.displayed_im = None # Utile pour le zoom
        self.current_zoom_level = None
        self.anchor_x_im = None
        self.anchor_y_im = None
        
        self.auto_corr_indices = None
        self.original_text = None
        self.expressions_set = None
        
        self.is_compil_var = tk.BooleanVar()
        
        self.all_rules = Cutil.create_all_rules()
        
        self.save_flag = True # We can temporary set this to False if we do not want to save the current corrections
        
        self.interest_stringvar = tk.StringVar()
        self.interest_options = IV_util.degrees_of_interest
        self.interest_options_inv =  {v: k for k, v in self.interest_options.items()}
    
    def reload_expressions_set(self):
        """
        Reload the last version of expression set into the model
        """
        self.expressions_set = Cutil.open_expr_file()
            
    def save_current_model_to_file(self):
        """
        Save the corrected text to the file and update the log df accordingly.
        And prepare the model to be filled with a new file.
        
        If self.save_flag is False, the save is not made.
        """
        corrected_text = self.view.text_widget.get("1.0",tk.END)
        
        # Update the stored expressions file
        new_exprs = Cutil.find_all_corrected_exprs(self.original_text, corrected_text)
        
        Cutil.update_expr_file(new_exprs)
        
        # Ici, passer la cellule concernée à True
        current_file_loc = self.needs_transcription_indexes[self.current_file_index_position]
        
        if self.save_flag:
            #Indiquer que le texte a été corrigé
            self.log_df.at[current_file_loc,"has_been_corrected"] = True
            
            # Sauver le texte concerné dans un fichier
            filename = self.log_df.at[current_file_loc,"txt_name"]
            if os.path.isfile(self.texts_path+filename):
                with open(self.texts_path+filename,"w",encoding = "utf8") as f:
                    f.write(corrected_text)
            else:
                raise Exception("Pas de texte correspondant")
        
        else: # Si on a demandé explicitement de ne pas sauvegarder, on indique le texte comme non-corrigé
            self.log_df.at[current_file_loc,"has_been_corrected"] = False
        
        #Interest
        current_cote = self.log_df.at[current_file_loc,"cote"]
        
        self.log_df.loc[self.log_df["cote"] == current_cote,"interest"] = self.interest_options.get(
        self.interest_stringvar.get())
        
        #Compilation
        self.log_df.loc[self.log_df["cote"] == current_cote,"is_compil"] = self.is_compil_var.get()
        
        
    
    def load_file_to_model(self,in_list_index):
        """
        Update the model info to match the df_index given.
        in_list_index : the index in the df index list
        of the df index that points to the file to load in the log dataFrame
        (It's a bit confusing, I know)
        """
        self.view.alert("")
        
        self.current_file_index_position = in_list_index
        current_file_loc = self.needs_transcription_indexes[self.current_file_index_position]
        
        filename = self.log_df.loc[current_file_loc,"txt_name"]
        
        if os.path.isfile(self.texts_path+filename):
            with open(self.texts_path+filename,"r",encoding = "utf8") as f:

                text = f.read()

                # Correction automatique
                text, auto_corr_start_indices = Cutil.automatic_correction(text,self.all_rules)
                self.auto_corr_indices = list(zip(auto_corr_start_indices,np.array(auto_corr_start_indices)+1))
                
                # Conserver le texte original pour comparaison ultérieure
                self.original_text = text
                
                # charger le texte dans le widget Text (c'est lui qui fera office de groundtruth)
                self.view.text_widget.delete("1.0",tk.END)
                self.view.text_widget.insert("1.0",text)
        else:
            self.view.text_widget.delete("1.0",tk.END)
            self.view.alert("Le texte correspondant à cette image est introuvable !")
        
        #Image initialisation
        im_name = self.log_df.loc[current_file_loc,"official_name"]
        
        self.original_im = PIL.Image.open(self.original_images_path+im_name)
        
        im, resizing_ratio = IV_util.load_and_resize_PIL_image(self.original_images_path+im_name,self.max_frame_size)
        self.resizing_ratio = resizing_ratio
        
        self.no_zoom_im = PIL.ImageTk.PhotoImage(im)
        self.current_zoom_level = 1
        self.anchor_x_im = 0
        self.anchor_y_im = 0
        
        self.displayed_im = self.no_zoom_im # Dans la fonction de Zoom, ! remplacer display, pas update
        
        #File info informations
        current_cote = self.log_df.at[current_file_loc,"cote"]
        cote_len = len(self.log_df[(self.log_df["cote"] == current_cote) & (self.log_df["needs_transcription"])
                                  & (self.log_df["states"].apply(
                                      lambda s : "f" in s or "l" in s or "t" in s))]) # Changer si ALLEMAND
        
        if not self.log_df.loc[current_file_loc,"has_been_corrected"]:
            info_text = "Fichier {} (total {}); pas encore corrigé.".format(
                self.log_df.loc[current_file_loc,"official_name"], cote_len)
            self.view.info_label.config(bg = "SystemButtonFace")
        else:
            info_text = "Fichier {} (total {}); déjà marqué comme corrigé.".format(
                self.log_df.loc[current_file_loc,"official_name"], cote_len)
            self.view.info_label.config(bg = "lightcyan")
            
        self.view.info_label.config(text = info_text)
        
        #Save flag
        self.save_flag = True
        self.view.no_save_button.config(bg = "SystemButtonFace")
        
        #Interest
        previous_interest = self.log_df.loc[current_file_loc,"interest"]
        
        if pd.isnull(previous_interest):
            if "v" in self.log_df.loc[current_file_loc,"states"]:
                self.interest_stringvar.set(self.interest_options_inv.get(2))
            else:
                self.interest_stringvar.set(self.interest_options_inv.get(4))
        
        else:
            self.interest_stringvar.set(self.interest_options_inv.get(previous_interest))
        
        #Compil
        previous_is_compil = self.log_df.loc[current_file_loc,"is_compil"]
        
        if pd.isnull(previous_is_compil):
            self.is_compil_var.set(False)
        else:
            self.is_compil_var.set(bool(previous_is_compil))
            
    
    def are_all_texts_corrected(self):
        "Return a boolean that is True if all texts in the dataFrame have been corrected"
        
        return all(self.all_txt_df["has_been_corrected"])
    
    
    def zoom_in(self,event):
        """
        Update the displayed image representation to a zoomed version
        event: the click event
        """
        
        zoom_factor = 2 #Must be an integer
        
        original_x = self.anchor_x_im + int(event.x*self.resizing_ratio/self.current_zoom_level)
        original_y = self.anchor_y_im + int(event.y*self.resizing_ratio/self.current_zoom_level)
        
        self.current_zoom_level *= zoom_factor
        
        self.anchor_x_im = original_x - self.max_frame_size*self.resizing_ratio/2/self.current_zoom_level
        self.anchor_y_im = original_y - self.max_frame_size*self.resizing_ratio/2/self.current_zoom_level
        
        antipod_x_im = original_x + self.max_frame_size*self.resizing_ratio/2/self.current_zoom_level
        antipod_y_im = original_y + self.max_frame_size*self.resizing_ratio/2/self.current_zoom_level
        
        cropped_image = self.original_im.crop((self.anchor_x_im,self.anchor_y_im,antipod_x_im,antipod_y_im))
        
        resized_image = cropped_image.resize((max_frame_size,max_frame_size),PIL.Image.LANCZOS)
        
        self.displayed_im = PIL.ImageTk.PhotoImage(resized_image)
        
    
    def zoom_out(self,event):
        """
        Return the displayed image representation to its original form
        event: the click event
        """
        
        self.displayed_im = self.no_zoom_im
        self.current_zoom_level = 1
        self.anchor_x_im = 0
        self.anchor_y_im = 0


# In[ ]:


class Controller():
    
    def __init__(self, model, view):
        
        self.view = view        
        self.model = model
        
    
    ## Loading into model and display
    
    def load_first_text(self):
        """
        Find the first text file of the directory and load it, if any. Else send an alert on the view
        """
        
        if len(self.model.log_df) > 0:
            next_file_index_position = 0
            self.full_load(next_file_index_position)
        else:
            self.view.alert("Tous les textes ont été corrigés !")
    
    
    def load_next_non_corrected_file(self):
        """
        Find the next non-corrected file, according to the df, and load it, if any.
        """
        needs_transcription_df = self.model.log_df.loc[self.model.needs_transcription_indexes]
        
        if not all(needs_transcription_df["has_been_corrected"]):
            next_file_index_position = needs_transcription_df["has_been_corrected"].apply(int).argmin()

            assert needs_transcription_df.iloc[next_file_index_position].name ==            self.model.needs_transcription_indexes[next_file_index_position],            "Internal error : index mismatch" #Supprimer l'assertion si on constate qu'elle passe

            self.full_load(next_file_index_position)
        
        else:
            self.view.alert("Tous les textes ont été corrigés !")
    
    
    def full_load(self,next_file_index_position):
        """
        Perform all the operation corresponding to the loading and display of a new file.
        Update the model so that the new file index position is set as the current index.
        Load the file corresponding to the next_file_index into the model, and then display it.
        next_file_index_position: the position in the file indexes list of the next file df index.
        """
        # Chargement des données dans le modèle et du texte dans le display
        self.model.load_file_to_model(next_file_index_position)
        
        # Mise à jour des expressions disponibles pour la correction manuelle
        self.model.reload_expressions_set()
        
        # Surlignage des zones sensibles
        self.view.update_text_highlighting()
        
        # Affichage de l'image
        self.view.load_image_in_view()
        
        
        
    
    ## Navigation callback
    
    def go_left_callback(self):
        
        self.model.save_current_model_to_file()
        
        current_file_index_position = self.model.current_file_index_position
        
        if current_file_index_position > 0:
            current_file_index_position -= 1
            self.full_load(current_file_index_position)
        
        if self.model.are_all_texts_corrected():
            self.view.alert("Tous les textes ont été corrigés !")
            
    
    def go_right_callback(self):
        
        self.model.save_current_model_to_file()
        
        current_file_index_position = self.model.current_file_index_position
        
        if current_file_index_position < len(self.model.needs_transcription_indexes)-1:
            current_file_index_position += 1
            self.full_load(current_file_index_position)
        
        if self.model.are_all_texts_corrected():
            self.view.alert("Tous les textes ont été corrigés !")
    
    
    def go_next_callback(self):
        
        self.model.save_current_model_to_file()
        
        self.load_next_non_corrected_file()
        
    
    def goto_callback(self,event):
        
        self.model.save_current_model_to_file()
        
        file_to_go = self.view.goto_entry.get()
        self.view.goto_entry.delete(0,tk.END)
        
        candidates = self.model.all_txt_df[
            self.model.all_txt_df["official_name"] == file_to_go]
        
        if len(candidates) > 0:
            index_in_df = candidates.index[0]
            
            if index_in_df in self.model.needs_transcription_indexes:
                current_file_index_position = list(self.model.needs_transcription_indexes).index(index_in_df)
                self.full_load(current_file_index_position)
            else:
                self.view.alert("Erreur. Incohérence dans les états internes")
        
        else:
            self.view.alert("Le fichier donné est introuvable. Entrez un nom officiel d'image brute.")


# In[ ]:


class Correction_Application(tk.Tk):
    
    
    def __init__(self, max_frame_size,texts_path,original_images_path):
        
        tk.Tk.__init__(self)
        
        
        # Class main objects
        
        self.model = Model(self,texts_path,original_images_path,max_frame_size)
        self.controller = Controller(self.model,self)
        
        
        # Widgets
        
        self.image_label = tk.Label(self)
        
        self.text_widget = tk.Text(self,height=36, font = ("Courier",12))
        
        self.go_left_button = tk.Button(self,text="Précédent", command = self.controller.go_left_callback)
        self.go_right_button = tk.Button(self,text="Suivant", command = self.controller.go_right_callback)
        self.go_next_button = tk.Button(self,text="Prochain sans correction",
                                        command = self.controller.go_next_callback)
        
        self.goto_label = tk.Label(self,text="Aller à un fichier :")
        self.goto_entry = tk.Entry(self)
        
        self.alert_label = tk.Label(self, bg = "lightcoral", text = "")
        self.info_label = tk.Label(self, text = "Fichier _ ; _")
        
        self.no_save_button = tk.Button(self,text="Ne pas sauvegarder", command = self.no_save_callback)
        
        self.interest_menu = tk.OptionMenu(self,self.model.interest_stringvar,
                                           *self.model.interest_options.keys())
        
        self.is_compil_checkbox = tk.Checkbutton(self, text = "est une compilation ?",
                                                 variable = self.model.is_compil_var,
                                                onvalue = True, offvalue = False)
        
        
        # Teyt highlightning initialisation
        self.text_widget.tag_configure("high_num",background = "#fffacd")
        self.text_widget.tag_configure("high_maj",background = "#ffe4e1")
        self.text_widget.tag_configure("high_cor",background = "#d8bfd8")
        
        
        # Packing
        
        self.image_label.grid(row=0,column=3,rowspan=5)
        self.text_widget.grid(row=0,column=0,columnspan=3,sticky="nsew")
        self.info_label.grid(row=1,column=0,columnspan=2)
        self.interest_menu.grid(row=1,column=2)
        self.alert_label.grid(row=2,column=0,columnspan=2)
        self.is_compil_checkbox.grid(row=2,column=2)
        self.go_left_button.grid(row=3,column=0)
        self.go_next_button.grid(row=3,column=1)
        self.go_right_button.grid(row=3,column=2)
        self.no_save_button.grid(row=4,column=0)
        self.goto_label.grid(row=4,column=1)
        self.goto_entry.grid(row=4,column=2)
        
        
        # Binding
        
        self.goto_entry.bind("<Return>",self.controller.goto_callback)
        
        self.image_label.bind("<Button-1>",self.zoom_callback)
        self.image_label.bind("<Button-2>",self.dezoom_callback)
        self.image_label.bind("<Button-3>",self.dezoom_callback)
        
        self.text_widget.bind("<Button-2>",self.popup_menu_callback)
        self.text_widget.bind("<Button-3>",self.popup_menu_callback)
        
        # Launching
        
        self.controller.load_first_text()
        
        self.mainloop()
      
    def destroy(self):
        """
        Save all data before closing
        """
        
        # Sauvegarder le fichier actuel
        self.model.save_current_model_to_file()
        
        # Sauvegarder le df
        IV_util.save_log_csv(self.model.log_df)
        
        tk.Tk.destroy(self)
    
    def load_image_in_view(self):
        self.image_label.destroy()
        self.image_label = tk.Label(self, image = self.model.displayed_im)
        self.image_label.image = self.model.displayed_im
        self.image_label.grid(row=0,column=4,rowspan=6)
        
        self.image_label.bind("<Button-1>",self.zoom_callback)
        self.image_label.bind("<Button-2>",self.dezoom_callback)
        self.image_label.bind("<Button-3>",self.dezoom_callback)
        
    
    ## Text highlight
    
    def highlight_pattern(self, text_widget, pattern, tag, start="1.0", end="end",
                      regexp=True):
        '''Apply the given tag to all text that matches the given pattern

        If 'regexp' is set to True, pattern will be treated as a regular
        expression according to Tcl's regular expression syntax.

        This function has been reproduced from
        https://stackoverflow.com/questions/3781670/how-to-highlight-text-in-a-tkinter-text-widget
        '''

        start = text_widget.index(start)
        end = text_widget.index(end)
        text_widget.mark_set("matchStart", start)
        text_widget.mark_set("matchEnd", start)
        text_widget.mark_set("searchLimit", end)

        count = tk.IntVar()
        while True:
            index = text_widget.search(pattern, "matchEnd","searchLimit",
                                count=count, regexp=regexp)

            if index == "": break
            if count.get() == 0: break # degenerate pattern which matches zero-length strings
            text_widget.mark_set("matchStart", index)
            text_widget.mark_set("matchEnd", "%s+%sc" % (index, count.get()))
            text_widget.tag_add(tag, "matchStart", "matchEnd")
            
    
    def highlight_indexes(self,text_widget,start_end_indices,tag):
        """
        start_end_indices: indices in the text in the form of tuple (start,end) end excluded
        """
        
        for first_i, last_i in start_end_indices:
            text_widget.mark_set("corrStart","1.0+"+str(first_i)+"c")
            text_widget.mark_set("corrEnd","1.0+"+str(last_i)+"c")
            text_widget.tag_add(tag,"corrStart","corrEnd")
            
    
    
    def update_text_highlighting(self,is_first_pass=True):
        "Apply the highlight to the text in the Text widget. May be called each time the text is changed"
        
        self.text_widget.tag_remove("high_num", "1.0", "end")
        self.text_widget.tag_remove("high_maj","1.0","end")
        self.highlight_pattern(self.text_widget,"\d+","high_num")
        self.highlight_pattern(self.text_widget,"[A-Z][A-Z]+","high_maj")
        
        # Apply the highlight according to automatic correction based on the indexes
        if is_first_pass:
            self.text_widget.tag_remove("high_cor","1.0","end")
            self.highlight_indexes(self.text_widget, self.model.auto_corr_indices,"high_cor")
        
        
    ## Display messages
    def alert(self,text):
        "Display the given text in the red alert label"
        self.alert_label.config(text = text)
        
    
    ## Save flag management (Here because it changes more things in view)
    def no_save_callback(self):
        
        self.model.save_flag = not self.model.save_flag
        
        if not self.model.save_flag:
            self.no_save_button.config(bg = "coral")
        else:
            self.no_save_button.config(bg = "SystemButtonFace")
    
    
    ## Zoom and dezoom
    def zoom_callback(self,event):
        self.model.zoom_in(event)        
        self.load_image_in_view()
    
    def dezoom_callback(self,event):
        self.model.zoom_out(event)
        self.load_image_in_view()
        
    
    ## Manual correction
    
    def popup_menu_callback(self,event):
        """
        Mark the position of the word subject to manual correction and open the associate menu
        
        Note importante : Si cette fonction ne considère pas les mots en entier
        en les coupant au niveau des accents, c'est que le texte n'est pas sous
        unicode normalization form NFC. utiliser unicodedata.normalize sur le
        texte avant de l'insérer dans le widget pour régler le problème.
        """
        self.text_widget.mark_unset("start_idx")
        self.text_widget.mark_unset("end_idx")
        
        start_idx = "@{x},{y} wordstart".format(x = event.x,y = event.y)
        end_idx = "@{x},{y} wordend".format(x = event.x,y = event.y)
        
        if self.text_widget.get(start_idx+" - 1 chars") == "'": # Gestion des mots avec appostrophe
            self.text_widget.mark_set("start_idx",start_idx+" - 2 chars")
        else:
            self.text_widget.mark_set("start_idx",start_idx)
        
        self.text_widget.mark_set("end_idx",end_idx)
        
        current_word = self.text_widget.get("start_idx", "end_idx")
        
        current_word = Cutil.strip_punctuation(current_word)
        
        self.generate_popup_menu(current_word,event.x_root,event.y_root)
        
    
    def generate_popup_menu(self,current_word,x,y):
        
        max_proposition_num = 10
        minimum_similarity = 0.4
        
        menu = tk.Menu(self,tearoff = 0)
        
        propositions = difflib.get_close_matches(current_word,self.model.expressions_set,
                                  n = max_proposition_num, cutoff = minimum_similarity)
        
        if len(propositions) > 0:
            for prop in propositions:
                menu.add_command(label = prop, command = lambda prop=prop:self.popup_menu_execute_replacement(prop))
                #lambda prop=prop est un trick pour forcer la fonction à fixer sa valeur d'entrée pendant la loop
        
        else:
            menu.add_command(label = "(Aucune proposition)")
            menu.entryconfig(0,foreground = "gray")
            
        
        menu.tk_popup(x,y)
        
        menu.grab_release()
    
    def popup_menu_execute_replacement(self,chosen_word):
        """
        Replace the word marked by the start_idx and end_idx marks in the text widget by the chosen word
        chosen_word : the word that must replace the current word.
        """
        
        # Gestion de la ponctuation (Si de la ponctuation entourait le mot actuel, cette dernière est réappondue)
        current_word = self.text_widget.get("start_idx", "end_idx")
        
        end_punct = ""
        while len(current_word)>0 and current_word[-1] in Cutil.basic_punctuation:
            end_punct = current_word[-1] + end_punct
            current_word = current_word[:-1]
        
        start_punct = ""
        while len(current_word)>0 and current_word[0] in Cutil.basic_punctuation:
            start_punct = start_punct + current_word[0]
            current_word = current_word[1:]
        
        # Remplacement du mot
        chosen_word_extended = start_punct+chosen_word+end_punct
        self.text_widget.delete("start_idx", "end_idx")
        self.text_widget.insert("start_idx",chosen_word)


# In[ ]:


texts_path = IV_util.four_SAVE_texts
original_images_path = IV_util.one_SAVE_raw_path

max_frame_size = 750

app = Correction_Application(max_frame_size,texts_path, original_images_path)


# In[ ]:


to_correct = app.model.log_df[app.model.log_df.needs_transcription & app.model.log_df.states.apply(lambda s :
                                                        "f" in s or "l" in s or "t" in s)]

print("{:10.4f}% effectués".format(to_correct.has_been_corrected.sum()/len(to_correct)*100))


# In[ ]:


print("Nombre d'image par catégorie")
app.model.log_df.groupby("interest").count()["Z_name"]


# In[ ]:


print("Nombre de cote par catégorie")
app.model.log_df.groupby("cote")["interest"].apply(lambda x : np.mean(list(x))).reset_index().groupby("interest").count()


# In[ ]:


print("Nombre de compilations")
app.model.log_df.groupby("cote")["is_compil"].apply(lambda x : list(x)).apply(lambda l :
                                                                False if any(pd.isnull(l)) else any(l)).sum()


# In[ ]:


app.model.log_df.tail(95)


# In[ ]:




