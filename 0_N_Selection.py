#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Génère une interface pour choisir les documents à inclure


# In[ ]:


from tkinter import *
import pandas as pd
import numpy as np
import os
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
import IV_util
import PIL.Image, PIL.ImageTk
import copy
import collections


# In[ ]:


# Pas d'output_directory, comme on ne modifie que le dataFrame


# In[ ]:


class Constants():
    """
    This static class contains all variables that are constant
    for a given instance of the application.
    """
    
    def __init__(self,input_directory_path,log_csv_path,version):
        
        self.input_directory_path = input_directory_path
        self.log_csv_path = log_csv_path
        self.input_images_filenames = os.listdir(self.input_directory_path)
        
        self.version = version
        
        self.input_cotes_n = collections.Counter(map(IV_util.get_cote,self.input_images_filenames))


class Model():
    """
    This class contains all variables that will change during
    the execution of the application.
    It contains current variable for the current file, and
    global variables such as the dataFrame that store everything
    
    The consistency_flag is set to False each time the model is
    modified and set to True by the checking function.
    """    
    def __init__(self, constants, code_constants, view):
        
        self.columns = ["Z_name","should_be_name","cote","is_included","states","doublon","remarque",
                       "selection_version"]
        
        if os.path.isfile(constants.log_csv_path):            
            self.df = IV_util.open_log_csv(constants.log_csv_path)                
        else:
            self.df = pd.DataFrame(columns = self.columns)
           
        self.constants = constants
        self.code_constants = code_constants
        self.view = view
            
        self.current_values = None
        self.current_z_name = None
        
        self.consistency_flag = False
    
    
    ### Change of current filename ###
    
    def reset_current_to_empty(self,z_name):
        """
        Reset the current_values dictionary to an empty dictionary.
        """
        
        self.current_z_name = z_name
        
        self.current_values = {"Z_name":self.current_z_name,"should_be_name":self.current_z_name,
                               "states":set(),"cote":IV_util.get_cote(self.current_z_name),
                               "selection_version":self.constants.version,
                              "is_included":None,"doublon":None,"remarque":None}
        
        assert set(self.current_values.keys()) == set(self.columns)
        
        self.consistency_flag = False
    
    
    def load_df_to_current(self,z_name):
        """
        Load the informations related to the given z_name
        to the current_values
        We make a deep copy of the current values row, in order to keep the model and the df independant.
        """

        self.current_z_name = z_name
        
        rows = self.df[self.df["Z_name"] == z_name][self.columns].to_dict(orient="records")
        
        assert len(rows) == 1, "Trying to load a non-existent or duplicated entry into the model"
        
        self.current_values = copy.deepcopy(rows[0])
        
        self.consistency_flag = False # The loaded rows are expected to be consistent, but just in case...
        

    def change_current_file(self,z_name):
        """
        Load the new filename information in the current model if the file already exists in the df,
        else reset the current model.
        """

        if z_name in self.df["Z_name"].to_list():
            self.load_df_to_current(z_name)
        else:
            self.reset_current_to_empty(z_name)

        
    ### Update values in current model ###
    
    def record(self,key,value):
        """
        Record a given current value in the model.
        key : the column name of the value.
        value : the value to be recorded.
        """
        
        assert key in self.columns, "Invalid key {} not in {}".format(key,self.columns)
        
        assert key != "Z_name",        "Z_name is used as an index column and cannot be changed without reseting the current model"
        
        self.current_values[key] = value
        
        self.consistency_flag = False
        
    
    def get(self,key):
        """
        Give the current value for the given key.
        Return : the value for the given key, None if no value have been set for the given key.
        """
        assert key in self.columns,  "Invalid key {} not in {}".format(key,self.columns)
        
        val = self.current_values.get(key,None)
        
        return val if not pd.isnull(val) else None #All of our functions expect None as a null value, not <NA>
     
    
    def add(self,key,value,f):
        """
        Merge a value to the existing one at a given key.
        key : the column name to treat
        value : the given value
        f : f(A old_val,B value) -> A new_val
        """
        
        old_val = self.get(key)
        
        if old_val == None:
            self.record(key,value)
        else:
            new_val = f(old_val,value)
            self.record(key,new_val)
        
        self.consistency_flag = False
    
    
    ### Current_file-dataFrame-csv interaction ###
    
    def save_current_to_df(self):
        """
        Save the current values stored in the model in the dataFrame of the model
        """
        
        assert self.current_z_name == self.current_values["Z_name"],        "The current filename ({}) and the indicated filename in the values ({}) are inconsistent"        .format(self.current_z_name,self.current_values["Z_name"])
        assert self.df["Z_name"].is_unique, "Z_name column in df is no longer unique"
        
        if self.consistency_flag == True:
            
            # Save the df at the right place
            current_row = self.df[self.df["Z_name"] == self.current_z_name]

            if current_row.empty:
                self.df = self.df.append(copy.deepcopy(self.current_values),ignore_index = True)
            else:
                current_index = current_row.index[0]
                
                self.df.loc[current_index,self.columns] = pd.Series(copy.deepcopy(self.current_values))
                
        else:
            self.view.alert_label.config(
                text = self.view.alert_label.cget("text") +
                "Sauvegarde échouée !\nVérifiez la consistance des valeurs entrées !")
        
        
        return self.consistency_flag
    
        
    def save_df_to_csv(self):
        """
        Save the local df into the csv
        We expect the local df to contains all the content of the previous csv file.
        """
        
        assert self.df["Z_name"].is_unique, "Z_name column in df is no longer unique"
        
        IV_util.save_log_csv(self.df)
        
    
    ### Consistency check of current file info ###
    
    def consistency_check(self):
        """
        Check if the current_values are consistent, and but the consistency_flag to True if it is the case.
        If not, raise an alert in the view
        """
        # Conditions pour contrôler la cohérence doublon - doublon sélectionné, renommage - renommage sélectionné
        # Conditions pour contrôler que le nouveau nom ait une cote valide
        # Condition pour controler que la cote indiquée soit celle du nouveau should_be_name
        # Lancer une alerte sinon

        doublon_cond =         (not pd.isnull(self.get("doublon")) and bool(set(self.code_constants.doublon_keys) & self.get("states")))        or        (pd.isnull(self.get("doublon")) and not bool(set(self.code_constants.doublon_keys) & self.get("states")))
        
        rename_cond =         (self.get("Z_name") == self.get("should_be_name") and
         self.code_constants.renaming_default_key not in self.get("states"))\
        or\
        (self.get("Z_name") != self.get("should_be_name") and
         self.code_constants.renaming_default_key in self.get("states"))
        
        cote_cond = IV_util.get_cote(self.get("should_be_name")) == self.get("cote")
        
        # Si on ajoute une colonne avec le numéro de page, la tester ici
        
        if not doublon_cond:
            old_text = self.view.alert_label.cget("text")
            self.view.alert_label.config(text = old_text+"""Consistance du champ "doublon" VS état "doublon"\n""")
        if not rename_cond:
            old_text = self.view.alert_label.cget("text")
            self.view.alert_label.config(text = old_text+"""Consistance du champ "nom" VS état "renommer"\n""")
        if not cote_cond:
            old_text = self.view.alert_label.cget("text")
            self.view.alert_label.config(text = old_text+"""La cote ne correspond pas au nom de fichier.\n""")
            
        
        self.consistency_flag = doublon_cond and rename_cond and cote_cond
        
        # Lancer ce truc après chaque modif et au moment d'enregistrer


# In[ ]:


class Controller():
    """
    This class contains most of the control functions that update the app values
    or react to key press.
    Exceptions are in-model changes, which are in the Model class
    and model-to-display updates which are in the View class.
    """
    
    def __init__(self, model, view, constants, code_constants):
        
        self.model = model
        self.view = view
        self.constants = constants
        self.code_constants = code_constants
      
    
    ### File navigation ###
    
    def load_first_file(self):
        """
        Load the first file of the directory in the model and the view
        """
        
        if len(self.constants.input_images_filenames) > 0:
            next_filename = self.constants.input_images_filenames[0]
            self.full_load_next(next_filename)
        
        else:
            self.view.task_done_warning()
        
        if self.find_next_non_decided_image() == None:
            self.view.task_done_warning()
    
    
    def find_next_non_decided_image(self):
        """
        Return a z_name which has currently no decision.
        Do not return anything if all z_names in the directory have already a decision.
        
        Return : string
        """
        assert self.model.df["Z_name"].is_unique
        
        is_not_decided = pd.isnull(self.model.df.is_included) |         (self.model.df.is_included != 0) & (self.model.df.is_included != 1)
        
        if is_not_decided.sum() > 0:
            next_filename = self.model.df[is_not_decided].iloc[0]["Z_name"]
            return next_filename
  

        already_treated_files = self.model.df["Z_name"].to_list()
        not_treated_files = [f for f in self.constants.input_images_filenames if f not in already_treated_files]
        
        if len(not_treated_files) > 0:
            next_filename = not_treated_files[0]
            return next_filename
        
    
    ### File loading ###
    
    def full_load_next(self,filename):
        """
        Proceed to all model and view updates to load the given filename
        """

        self.model.change_current_file(filename)
        self.view.load_image()
        self.view.load_model_to_display()
    
    
    def save_display_entries_to_model(self):
        """
        Save the content of the text entries into the model.
        No consistency check are made here, they are done in the model
        when saving the current values to the 
        """
        
        doublon_text = self.view.doublon_text.get()
        
        if len(doublon_text) > 0:
            self.model.record("doublon",doublon_text)
        else:
            self.model.record("doublon",None)
            
            
        remarque_text = self.view.remarque_text.get()
        
        if len(remarque_text) > 0:
            self.model.record("remarque",remarque_text)
        else:
            self.model.record("remarque",None)
        
        
        sbname = self.view.image_ref_text.get()
        self.model.record("should_be_name",sbname) # Cannot be empty
        
        
    
    ### Callback functions (run when a key is pressed) ###
    
    """
    Note : callback functions also contain the code that launch cascade reactions
    when a state is selected or an Entry is filled to select other related states or decisions.
    (For exemple, when pressing yes, the default_yes_state is selected if no state is selected yet)
    If you want to modify them, look here.
    """
    
    def global_callback(self, event):
        """
        Function called whenever a key is pressed.
        """
        key_code = event.keysym
        
        # This first condition is not related to the navigation nor the decisions
        # but is used to validate an entry text field.
        if key_code == "Return":
            self.return_callback()
            
        
        #Condition for navigation
        elif key_code == "Up" or key_code == "Left" or key_code == "Right":
            self.navigation_callback(key_code)
        
        #Condition for state update (every other key press)
        # Do not proceed if we are currently filling an entry box
        elif not "entry" in str(self.view.focus_get()):
            self.update_states_decision_callback(key_code)
    
    
    def return_callback(self):
        """
        Supposed to be called when an entry text field is validated.
        Defocus the entry widget and check if state match with entry text filling.
        """
        
        ## Sanity check for the entries
        if not IV_util.check_valid_filename_format(self.view.image_ref_text.get(),".jpg"):
            self.view.alert_label.config(text = "Nom de fichier invalide saisi ! (Cote ou page introuvable)")
        else:
            original_name = self.model.get("should_be_name")
            
            if len(self.view.doublon_text.get())>0:
                try:
                    IV_util.get_cote(self.view.doublon_text.get())
            
                except:
                    self.view.alert_label.config(text = "Doublon n'est pas un format de cote valide !")
                
                else:
                    self.view.focus_set()
                    self.save_display_entries_to_model()
            
            else:
                self.view.focus_set()
                self.save_display_entries_to_model()

            
            ## Bind the content of the entries to states so that everything is consistent
            key_code = None
            
            # Doublons
            if len(self.view.doublon_text.get())>0 and            not (set(self.code_constants.doublon_keys) & self.model.get("states")):
                key_code = self.code_constants.doublon_default_key
                
            # Renaming
            new_name = self.view.image_ref_text.get()
            
            if new_name != original_name:
                key_code = self.code_constants.renaming_default_key
                
                new_cote = IV_util.get_cote(new_name)
                original_cote = IV_util.get_cote(original_name)
                if new_cote != original_cote:
                    self.view.alert_label.config(text = """Attention : Ce renommage changera la cote de l'image
                    Nouvelle cote : {}""".format(new_cote))
                    self.model.record("cote",new_cote)
                #else:
                    #self.view.alert_label.config(text = "")
            
            if key_code:
                self.model.add("states",set([key_code]),lambda s1,s2: s1.union(s2))
                self.flow_back_state_info_to_decision(key_code)

            self.view.load_model_to_display()          
        
        ### Gestion du GOTO (Le goto_text exige un Z_name valide)
        if len(self.view.goto_text.get()) > 0 and "entry" not in str(self.view.focus_get()):
            # Usage détourné de navigation callback avec une str qui n'est pas un vrai key_code
            self.navigation_callback("goto_filename") 
                
    
    
    def navigation_callback(self, key_code):
        """
        Load a new file if possible when an arrow is pressed
        """
        
        # Do not allow to save and navigate if we are still filling an entry
        if "entry" in str(self.view.focus_get()):
            self.view.alert_label.config(text = "Valider le champ avec ⏎ avant de continuer !")
        else:
            self.view.alert_label.config(text = "")
            
            current_cursor = self.constants.input_images_filenames.index(self.model.get("Z_name"))
            next_filename = None
            
            self.model.consistency_check()
            is_save_successful = self.model.save_current_to_df()
            
            if is_save_successful:
                if key_code == "Up":
                    next_filename = self.find_next_non_decided_image()
                
                if key_code == "goto_filename":
                    z_name_to_go = self.view.goto_text.get()
                    # Seul goto peut introduire des noms invalides dans le système --> Essentiel de tester ici
                    if z_name_to_go in self.constants.input_images_filenames:
                        next_filename = z_name_to_go
                    else:
                        self.view.alert_label.config(
                            text = "Aller à : Nom invalide ! Entrez un nom de fichier présent dans le dossier")
                    
                elif key_code == "Left":
                    if current_cursor > 0:
                        current_cursor -= 1
                        next_filename = self.constants.input_images_filenames[current_cursor]

                elif key_code == "Right":
                    if current_cursor < len(self.constants.input_images_filenames)-1:
                        current_cursor += 1
                        next_filename = self.constants.input_images_filenames[current_cursor]

                if next_filename != None:
                    self.full_load_next(next_filename)

                if self.find_next_non_decided_image() == None:
                    self.view.task_done_warning()
    
        
    def update_states_decision_callback(self,key_code):
        """
        Update the states or the decision according to the given key code
        """
        
        #Decision
        if key_code == "1" or key_code == "y":
            self.model.record("is_included",1)

            # Indique une valeur par défaut pour l'état si le fichier est sélectionné pour être inclu.
            # mais qu'aucune valeur d'état n'a été indiquée
            if not self.model.get("states"):
                self.model.add("states",self.code_constants.default_yes_state,lambda s,e : s.union(set(e)))

        elif key_code == "0" or key_code == "n":
            self.model.record("is_included",0)

        #States
        elif key_code in self.code_constants.code:

            if key_code not in self.model.get("states"):

                self.model.add("states",set([key_code]), lambda s1,s2 : s1.union(s2))

                self.flow_back_state_info_to_decision(key_code)

                # Tente de compléter automatiquement le champ doublon
                if key_code in self.code_constants.doublon_keys and len(self.view.doublon_text.get())==0:

                    current_cote = self.model.get("cote")
                    previous_possible_doublons = self.model.df[self.model.df["cote"] == current_cote]["doublon"]
                    previous_doublons = previous_possible_doublons[~pd.isnull(previous_possible_doublons)].tolist()
                    if previous_doublons:
                        self.model.record("doublon",previous_doublons[-1])
                
                # Lie "n'existe plus" à la key "doublon"
                if key_code in self.code_constants.doublon_keys:
                    self.model.add("states",set([self.code_constants.suppressed_default_key]),
                                  lambda s1,s2 : s1.union(s2))
                
                # Désactive la décision si le champ est "special"
                if key_code == self.code_constants.special_case_key:
                    self.model.record("is_included",None)
            
            # Remove the code from the states if pressed while already present
            elif key_code in self.model.get("states"):
                self.model.record("states", self.model.get("states") - set([key_code]))

        self.view.load_model_to_display()
        
        
    def flow_back_state_info_to_decision(self,key_code):
        """
        Pick a default decision if a state is indicated while no decision has been made.
        """
        if self.model.get("is_included") != 1 and self.model.get("is_included") != 0:
                if key_code in self.code_constants.yes_binding:
                    self.model.record("is_included",1)
                else:
                    self.model.record("is_included",0)
        
        if key_code == self.code_constants.doublon_default_key:
            self.model.add("states",set([self.code_constants.suppressed_default_key]), lambda s1,s2 : s1.union(s2))


# In[ ]:


class Selection_application(Tk):

    ### Initialisation ###
    
    def __init__(self,log_csv_path,input_directory_path, max_frame_size, version):
        """
        input_directory_path : the input directory containing all raw images to select
        log_csv_path : the mega csv that contain the status of each cote.
        max_frame_size : the size of the frame that will display the image
        version : the version of the selection process
        
        Two files that have the same version have followed the same selection and states criteria and definitions
        """
        
        Tk.__init__(self)
        
        # Class main elements
        self.code_constants = IV_util.CodeConstants()
        self.constants = Constants(input_directory_path = input_directory_path,
                                   log_csv_path = log_csv_path,
                                   version = version)
        self.model = Model(constants = self.constants,code_constants = self.code_constants, view = self)
        self.controller = Controller(model = self.model, view = self,
                                    constants = self.constants, code_constants = self.code_constants)
        
        # GUI constants
        
        self.max_frame_size = max_frame_size
        
        
        # Control text
        control_explain =        """
        Contrôles :
        Toujours appuyer sur ⏎ après avoir rempli un champ.
        Inclure l'image : [y] ou [1]
        Ne pas inclure l'image : [n] ou [0]
        Passer d'une image à l'autre : [←] et [→]
        Trouver la prochaine image sans décision : [↑]
        """
        for i, decision_key in enumerate(self.code_constants.code):
            ending = "\n" if i%2 == 1 or i == len(self.code_constants.code)-1 else " / "
            control_explain += "{} : [{}]".format(self.code_constants.code.get(decision_key), decision_key)+ending
        
        
        # Widgets
        lw = 18
        lh = 2
        basis_color = "silver"
        
        self.image_label = Label(self)
        self.inclusion_label = Label(self, text = "_\n\nInclure ?")
        self.yes_key = Label(self, text = "oui",bg = basis_color,width = lw, height = lh)
        self.no_key = Label(self, text = "non", bg = basis_color,width = lw, height = lh)
        self.options_label = Label(self, text = "Etat")
        self.doublon_label = Label(self, text = "Doublon de")
        self.doublon_text = Entry(self)
        self.remarque_label = Label(self, text = "Remarque")
        self.remarque_text = Entry(self,width=50)
        self.image_ref_label = Label(self, text = "Renommer _ ?")
        self.image_ref_text = Entry(self,width=50)
        self.summary_ref_label = Label(self, text = "Cote _ \nNombre de fichiers : _")
        self.alert_label = Label(self, bg = "lightcoral", text = "")
        self.controls_label = Label(self, text = control_explain)
        self.goto_label = Label(self, text = "Aller à :")
        self.goto_text = Entry(self,width = 30)
        
        self.keys = {}
        for decision_key in self.code_constants.code:
            self.keys[decision_key] = Label(self, text = self.code_constants.code.get(decision_key),
                                            bg = basis_color, width = lw, height = lh)
            
        
        # Packing
        self.size_options = len(self.keys)//2 + len(self.keys)%2
        
        self.grid_image_label()
        self.inclusion_label.grid(row = 0, column = 1, columnspan = 2)
        self.yes_key.grid(row = 1, column = 1)
        self.no_key.grid(row = 1, column = 2)
        self.options_label.grid(row = 2, column = 1, columnspan = 2)
        
        for i, k in enumerate(self.keys):
            self.keys[k].grid(row = 3+i//2, column = 1+i%2, padx = 2, pady = 2)
        
        self.doublon_label.grid(row = 3 + self.size_options, column = 1, columnspan = 2)
        self.doublon_text.grid(row = 4 + self.size_options, column = 1, columnspan = 2)
        self.remarque_label.grid(row = 5 + self.size_options, column = 1, columnspan = 2)
        self.remarque_text.grid(row = 6 + self.size_options, column = 1, columnspan = 2)
        self.image_ref_label.grid(row = 7 + self.size_options, column = 1, columnspan = 2)
        self.image_ref_text.grid(row = 8 + self.size_options, column = 1, columnspan = 2)
        self.summary_ref_label.grid(row = 9 + self.size_options, column = 1, columnspan = 2)
        self.alert_label.grid(row = 10 + self.size_options, column = 1, columnspan = 2)
        self.controls_label.grid(row = 11 + self.size_options, column = 1, columnspan = 2)
        self.goto_label.grid(row = 12 + self.size_options, column = 1)
        self.goto_text.grid(row = 12 + self.size_options, column = 2)
        
        
        # Command binding
        self.bind("<KeyPress>",self.controller.global_callback)
        
        
        # Launching
        self.controller.load_first_file()
        
        self.mainloop()
 
    
    def grid_image_label(self):
        self.image_label.grid(row = 0, column = 0, rowspan = self.size_options + 13)
    
    
    ### Overloading of closing function ###
    
    def destroy(self):
        if len(self.constants.input_images_filenames) > 0:
            if not "entry" in str(self.focus_get()):
                self.model.consistency_check()
                is_save_successful = self.model.save_current_to_df()
                if is_save_successful:
                    self.model.save_df_to_csv()
                    Tk.destroy(self)
            else:
                self.alert_label.config(text = "Finissez de remplir les champs avant de fermer.")
        else:
            Tk.destroy(self)
            

    ### Loading element in display ###
    # (Display to model is, on the other hand, done in the controller)
        
    def load_model_to_display(self):
               
        ## Entries and Labels
               
        # Load current filename and should_be_name to display
        z_filename = self.model.get("Z_name")
        self.inclusion_label.config(text = "{}\n\nInclure ?".format(z_filename))
        self.image_ref_label.config(text = "Renommer {} ?".format(z_filename))
        self.image_ref_text.delete(0,END)
        self.image_ref_text.insert(0,str(self.model.get("should_be_name")))
        
        # Load current remarque to display
        remarque = self.model.get("remarque")
        self.remarque_text.delete(0,END)
        self.remarque_text.insert(0,"" if pd.isnull(remarque) else str(remarque))
        
        # Load current doublon to display
        doublon = self.model.get("doublon")
        self.doublon_text.delete(0,END)
        self.doublon_text.insert(0, "" if pd.isnull(doublon) else str(doublon))
        
        # Load cote information
        cote = self.model.get("cote")
        n_files = self.constants.input_cotes_n[cote]
        self.summary_ref_label.config(text = "Cote {}\nNombre initial de fichiers : {}".format(cote,n_files))
        
        
        
        ## Color of the inclusion and state panels.
         
        selected_color = "mediumspringgreen"
        unselected_color = "silver"
        
        if self.model.get("is_included") == 1:
            self.yes_key.config(bg = selected_color)
            self.no_key.config(bg = unselected_color)
        elif self.model.get("is_included") == 0:
            self.no_key.config(bg = selected_color)
            self.yes_key.config(bg = unselected_color)
        else:
            self.yes_key.config(bg = unselected_color)
            self.no_key.config(bg = unselected_color)
        
        for k in self.keys:
            if k in self.model.get("states"):
                self.keys[k].config(bg = selected_color)
            else:
                self.keys[k].config(bg = unselected_color)
    
    
    
    def load_image(self):
        """
        Load the next image with its infos, if present (pas oublier de checker si des infos sont déjà présentes!)
        Réinitialise tout le contenu des widgets et des variables -> Non. Déléguer à update_display
        
        Load the image corresponding to the current file
        """
        
        file_to_load = self.constants.input_directory_path + self.model.current_z_name
        
        im, resize_ratio = IV_util.load_and_resize_PIL_image(file_to_load, self.max_frame_size)
        
        im_object = PIL.ImageTk.PhotoImage(im)
        self.image_label.destroy()
        self.image_label = Label(self, image = im_object)
        self.image_label.image = im_object
        self.grid_image_label()
        
        # Global display reseting
        self.goto_text.delete(0,END)
        
        self.alert_label.config(text = "")
    
    
    ### Warnings
    
    def task_done_warning(self):
        self.config(bg = "lemonchiffon")
        self.alert_label.config(text = "### Tous les fichiers ont été traîtés ! ###")


# In[ ]:


# Selection application
log_csv_path = IV_util.log_csv_path
input_path = IV_util.zero_to_select_path
max_frame_size = 750
version = "prod_2"

app = Selection_application(log_csv_path, input_path, max_frame_size, version)

# Creation of the "needs_transcription" column
log_df = IV_util.open_log_csv()
cc = IV_util.CodeConstants()
langs = set(cc.lang_keys)
rename_key = set(cc.renaming_default_key)
doublon_keys = set(cc.doublon_keys) # On peut choisir de représenter une cote par son doublon.
log_df["needs_transcription"] = log_df["is_included"].apply(lambda d: False if pd.isnull(d) else bool(d))                                & log_df["states"].apply(lambda s : bool(s) and not(
    "b" in s or "p" in s or "m" in s))
IV_util.save_log_csv(log_df)


# In[ ]:


log_df.sample(5)


# Les états renvoient au document et à la cote du should-be-name de ce document, pas à la cote de l'original name ! En effet, si une cote numérique ne correspond pas à une cote physique, on commence par renommer avant d'indiquer les états (Avec exception de l'état "renommer" lui-même). Ceci permet également, et c'est là que c'est intéressant, d'intervertir deux images dont une est le doublon de l'autre afin d'utiliser l'image portant la cote originale supprimée comme représentante de la cote gardée, si elle est de meilleure qualité.

# **Historique des versions**
# * prod_1 : originale
# * prod_2 : suppression de "i":"illisible". Ajout de "v":"structure vide" pour les formulaires/tableaux/textes à trou

# In[ ]:





# In[ ]:


log_df[log_df.cote == "3058"]


# In[ ]:




