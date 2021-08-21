#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Ce petit notebook crée un champ dans les logs pour lier chaque image originale à ses extraits binaires associés. Il doit être exécuté avant le script 3_N_OCRize

# In[1]:


import IV_util
import pandas as pd
import os
import re
from tqdm import tqdm


# In[2]:


log_df = IV_util.open_log_csv()

binary_images_names = pd.Series(os.listdir(IV_util.two_SAVE_binary_images))

#log_df["binary_files"] = log_df["binary_files"].apply(lambda x : eval(x) if not pd.isnull(x) else x)


# In[3]:


for i, row in tqdm(log_df.iterrows()):
    
    try:
        if ("f" in row["states"] or "l" in row["states"] or "t" in row["states"])        and row["needs_transcription"] == 1 and         (pd.isnull(row["binary_files"]) or len(row["binary_files"]) == 0):

            associated_tifs = binary_images_names[
                binary_images_names.apply(lambda n : re.search("\d+[a-z]*_\d+",n).group() ==
                                          re.search("\d+[a-z]*_\d+",row["official_name"]).group()).tolist()
            ]

            if not(len(associated_tifs) > 0 and len(associated_tifs) <= 2):
                display(row["official_name"]+" has an unexpected number of binary images : "+str(associated_tifs))

            log_df.at[i,"binary_files"] = set(associated_tifs)
            
    except:
        raise Exception(str(i))


# In[4]:


IV_util.save_log_csv(log_df)


# In[5]:


log_df[log_df.needs_transcription & pd.isnull(log_df.binary_files)
                & log_df.states.apply(lambda s : not "a" in s)]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




