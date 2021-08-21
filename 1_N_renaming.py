#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# In[1]:


import os
import re
import IV_util
import pandas as pd
import numpy as np


# In[2]:


input_path = IV_util.zero_to_select_path
renamed_path = IV_util.one_SAVE_raw_path
garbage_path = IV_util.one_garbage_path
df_path = IV_util.log_csv_path


# In[3]:


log_df = IV_util.open_log_csv()


# ### Renommage de chaque fichier selon les infos indiquées par should_be_name vers leur nouveau nom.

# In[4]:


def rename_files_from_row(row):
    
    filename = row["Z_name"]
    new_name = row["official_name"]
    
    if row["is_included"] == 1:
        if not os.path.isfile(renamed_path+new_name):
            if os.path.isfile(input_path+filename):
                os.rename(input_path+filename,renamed_path+new_name)
            else:
                print("ERROR: {} ({}) should be included but is not in the pipeline".format(filename,new_name))
        else:
            if os.path.isfile(input_path+filename):
                print("WARNING: {} coud not be renamed because {} already exists.".format(filename,new_name))
    elif row["is_included"] == 0:
        if os.path.isfile(input_path+filename):
            os.rename(input_path+filename,garbage_path+filename)
    else:
        print("{} has not been treated because no decision has been taken.".format(filename))


# In[19]:


log_df["official_name"] =       log_df.apply(lambda r : IV_util.rename(r["should_be_name"]) if r["is_included"] == 1 else None,axis = 1)

#Verification of uniqueness of each official name
assert log_df[~pd.isnull(log_df["official_name"])]["official_name"].is_unique, "Two or more names led to the same official name.\n{}".format(
    log_df[log_df["official_name"].duplicated()]["official_name"].tolist())

assert all(log_df[~pd.isnull(log_df["official_name"])]["official_name"].apply(IV_util.get_cote) ==
           log_df["cote"][~pd.isnull(log_df["official_name"])])

log_df.apply(rename_files_from_row,axis=1)


# Création des champs pour la partie 2 (DEPRECATED)
#color_equal_ext = "_colorequal"
#log_df["color_equal_name"] = IV_util.construct_derived_name(log_df,color_equal_ext,".jpg")

#log_df["raw_binary_name"] = IV_util.construct_derived_name(log_df,"",".tif")

#dskwd_ext = "_deskewed"
#log_df["deskewed_binary_name"] =IV_util.construct_derived_name(log_df,dskwd_ext,".tif")

#segments_ext = "_segments"
#log_df["segments_name"] = IV_util.construct_derived_name(log_df,segments_ext,".json")

#alto_ext = "_alto"
#log_df["alto_name"] = IV_util.construct_derived_name(log_df,alto_ext,".json")


IV_util.save_log_csv(log_df)


# In[ ]:




