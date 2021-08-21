#!/usr/bin/env python
# coding: utf-8

# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# In[1]:


import os
import IV_util
import pandas as pd
import shutil
import re
from tqdm import tqdm


# In[2]:


log_df = IV_util.open_log_csv()

save_path = IV_util.one_SAVE_raw_path

fr_path = IV_util.two_TEMP_francais_transcription
de_path = IV_util.two_TEMP_deutsch_transcription
no_path = IV_util.two_TEMP_no_transcription

fr_suffix = "_fr.jpg"
de_suffix = "_de.jpg"


# In[3]:


# NE PAS REUTILISER. SI DE NOUVEAUX FICHIERS ARRIVENT, FAIRE UN BATCH A PART

for i, row in tqdm(log_df.iterrows()):
    
    
    if row["is_included"] == 1:
        
        
        if row["needs_transcription"]:
            
            batch_num = int(re.search("\d+",row["official_name"]).group())//100
            

            if "f" in row["states"] or "t" in row["states"] or "l" in row["states"]:
                
                dir_path = fr_path+"batch{:02d}/".format(batch_num)
                
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                
                original_path = save_path+row["official_name"]
                scantailor_path = dir_path + IV_util.get_prefix(row["official_name"]) + fr_suffix
                
                if not os.path.isfile(scantailor_path):
                    #shutil.copyfile(original_path, scantailor_path)
            
            
            if "a" in row["states"]:
                
                dir_path = de_path+"batch{:02d}/".format(batch_num)
                
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                    
                original_path = save_path+row["official_name"]
                scantailor_path = dir_path + IV_util.get_prefix(row["official_name"]) + de_suffix
                
                if not os.path.isfile(scantailor_path):
                    #shutil.copyfile(original_path, scantailor_path)


# In[27]:


for d in tqdm(os.listdir(fr_path)):
    
    for im_name in os.listdir(fr_path+d+"/out"):
        
        if im_name.endswith(".tif"):
            os.rename(fr_path+d+"/out/"+im_name,"2_SAVE_binary_images/"+im_name)


# In[13]:





# In[7]:


for i, row in log_df.tail(90).iterrows():
    
    if row["is_included"] == 1:
        
        if row["needs_transcription"]:
            
            if "f" in row["states"] or "t" in row["states"] or "l" in row["states"]:
                
                dir_path = fr_path+"extra_1/"
                
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                
                original_path = save_path+row["official_name"]
                scantailor_path = dir_path + IV_util.get_prefix(row["official_name"]) + fr_suffix
                
                if not os.path.isfile(scantailor_path):
                    shutil.copyfile(original_path, scantailor_path)
                    
                    
            if "a" in row["states"]:
                
                dir_path = de_path+"extra_1/"
                
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                    
                original_path = save_path+row["official_name"]
                scantailor_path = dir_path + IV_util.get_prefix(row["official_name"]) + de_suffix
                
                if not os.path.isfile(scantailor_path):
                    shutil.copyfile(original_path, scantailor_path)


# In[4]:


int(re.search("\d+","Ch-AEV_00822_001").group())//100


# In[ ]:





# In[ ]:





# In[6]:





# In[ ]:




