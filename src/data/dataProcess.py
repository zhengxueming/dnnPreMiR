"""Generate the new csv data files only containing the "Accession","seq_struc" and 
   "Classification".The seq_struct data are merged from the sequences and 
   the second structure data
"""

import numpy as np
import pandas as pd 
from sklearn.utils import shuffle

# merge the base and the structure information into one column
def merge_loci(df_column1,df_column2):
    new_column = []
    for i in range(len(df_column1)):
        new_item = ""
        for j in range(len(df_column1[i])):
            new_item += df_column1[i][j]
            new_item += df_column2[i][j]
            new_item += " "
        new_column.append(new_item)
    return new_column

# generate the new csv data files
def generate_new_data():
    FILE_PATH_HSA = "hsa.csv" 
    FILE_PATH_PSEUDO = "pseudo.csv"
    try:
        hsa = pd.read_csv(FILE_PATH_HSA)
        pseudo = pd.read_csv(FILE_PATH_PSEUDO)
    except IOError as err:
        print("File not found!Please copy the datasets to the current directory.")
        exit(1)
    # process the positive data
    hsa = hsa.loc[:,["Accession","HairpinSequence","RNAFolds"]]
    hsa['Classification'] = "TRUE"
    # merge HairpinSequence and RNAFolds in hsa
    hsa_merge_column = merge_loci(hsa["HairpinSequence"],hsa["RNAFolds"])
    hsa_merge_column = np.array(hsa_merge_column)
    # print(hsa_merge_column)

    hsa_merge_column = pd.Series(hsa_merge_column)
    hsa["seq_struc"] = hsa_merge_column
    hsa_new = hsa.loc[:,["Accession","seq_struc","Classification"]]
    # print(hsa_new)
    
    # process the negative data
    pseudo = pseudo.loc[:,["Accession","HairpinSequence","RNAFolds"]]
    # get the same number as postive dataset
    hsa_rows = len(hsa)
   # print("hsa_rows:",hsa_rows)
    pseudo = shuffle(pseudo,random_state = 0)
    pseudo = pseudo.iloc[0:hsa_rows,:]
    pseudo = pseudo.reset_index(drop=True)
    # print(pseudo)
    pseudo['Classification'] = "FALSE"

    # merge HairpinSequence and RNAFolds in pseudo
    pseudo_merge_column = merge_loci(pseudo["HairpinSequence"],pseudo["RNAFolds"])
    pseudo_merge_column = np.array(pseudo_merge_column)
    # print(pseudo_merge_column)

    pseudo_merge_column = pd.Series(pseudo_merge_column)
    pseudo["seq_struc"] = pseudo_merge_column
    # print(pseudo)

    pseudo_new = pseudo.loc[:,["Accession","seq_struc","Classification"]]
    # print(pseudo_new)

    #write to file
    pseudo_new.to_csv("pseudo_new.csv")
    hsa_new.to_csv("hsa_new.csv")

    print("写入数据完成")

if __name__ == "__main__":
    generate_new_data()

