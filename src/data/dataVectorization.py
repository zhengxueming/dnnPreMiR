"""Transform the seq_struct data and classification values into vectors 
   using one-hot encoding. All the sequences are padded into 180 length.
   The returned x_dataset,y_dataset are numpy arrays.
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import dataSetGenerate 

x_cast = {"A.":[1,0,0,0,0,0,0,0,0,0,0,0],"U.":[0,1,0,0,0,0,0,0,0,0,0,0],\
          "G.":[0,0,1,0,0,0,0,0,0,0,0,0],"C.":[0,0,0,1,0,0,0,0,0,0,0,0],\
          "A(":[0,0,0,0,1,0,0,0,0,0,0,0],"U(":[0,0,0,0,0,1,0,0,0,0,0,0],\
          "G(":[0,0,0,0,0,0,1,0,0,0,0,0],"C(":[0,0,0,0,0,0,0,1,0,0,0,0],\
          "A)":[0,0,0,0,0,0,0,0,1,0,0,0],"U)":[0,0,0,0,0,0,0,0,0,1,0,0],\
          "G)":[0,0,0,0,0,0,0,0,0,0,1,0],"C)":[0,0,0,0,0,0,0,0,0,0,0,1] }
y_cast = {True: [1,0],False:[0,1]}

# transform every seq_struc items into a vector of (180,12),padding with zero vectors
def transform_xdata(df_column):
    SEQ_LEN = 180
    x_dataset = []
    for line in df_column:
        line = line.strip()
        line = line.split(" ")
        temp_list = []
        for item in line:
            temp_list.append(x_cast[item])
        for i in range(SEQ_LEN-len(line)):
            temp_list.append([0,0,0,0,0,0,0,0,0,0,0,0])
        x_dataset.append(temp_list)
    return x_dataset

# transform the TRUE and FALSE into vector
def transform_ydata(df_column):
    y_dataset = []
    for line in df_column:
        y_dataset.append(y_cast[line])
    return y_dataset

# read the data and transform into vectors of numpy array for deep learning
def vectorize_data(positive_file_path,negative_file_path):
    # read files as dataframe    
    dataframe = dataSetGenerate.read_new_csv(positive_file_path,negative_file_path)
    # transform the dataset into vectors for deep learning
    x_dataset = transform_xdata(dataframe["seq_struc"])
    y_dataset = transform_ydata(dataframe["Classification"])
    # transform into numpy array
    x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)
    #print(x_dataset)
    #print(y_dataset)
    print("data vectorization finished!")
    return x_dataset,y_dataset

if __name__ == "__main__":
    positive = "hsa_new.csv"
    negative = "pseudo_new.csv"

    x_dataset,y_dataset =  vectorize_data(positive,negative)

