# @author : Cédric Tomasini (cedric.tomasini@epfl.ch)

# Create the txt files from the binary images

import IV_util
import os
import Segmentation_util as Sutil
import pandas as pd

BATCH_SIZE = 5000

log_df = IV_util.open_log_csv()

binary_path = IV_util.two_SAVE_binary_images

texts_LR_path = IV_util.three_TEMP_left_right_texts
texts_path = IV_util.four_SAVE_texts

counter = 0
for i, row in log_df.iterrows():
    
    if counter >= BATCH_SIZE:
        break
    
    fail_flag = False
    
    if not pd.isnull(row["binary_files"]):
        
        txt_name = IV_util.get_prefix(row["official_name"])+"_fr.txt"
        txt_build = []
        
        for bn in sorted(row["binary_files"]):
            
            prefix = IV_util.get_prefix(bn)
            lrtn = prefix+"_temp"
            lrtntxt = lrtn+".txt"
            
            if not os.path.isfile(texts_LR_path+lrtntxt):
                print("Processing {}/{} ({})".format(counter+1,BATCH_SIZE,prefix))
                counter += 1
                os.system("tesseract {} {} -l fra".format(binary_path+bn,texts_LR_path+lrtn))
                          
                with open(texts_LR_path+lrtntxt,"r",encoding="utf8") as f:
                    current_text = f.read()
                    
                    if len(current_text) == 0: # Check if Tesseract has fail to transcribe anything for some reason
                        os.remove(texts_LR_path+lrtntxt) # Delete the file so that it is transcribed next time
                        fail_flag = True
                    txt_build.append(current_text)
                    
        if not os.path.isfile(texts_path+txt_name) and not fail_flag:
            with open(texts_path+txt_name,"w",encoding = "utf8") as output_file:
                output_file.write("\n".join(txt_build))
