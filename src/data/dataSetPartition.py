""" Partition the data into train, cross validation and test datasets.
"""

import numpy as np
from dataVectorization import vectorize_data

# generate the train(4/5) and test datasets(1/5)
def train_test_partition(positive_file,negative_file):
    # generate the vectorized xs and ys
    x_dataset,y_dataset =  vectorize_data(positive_file,negative_file)
    print("data vectorization in function train_test_partition finished!")
    print(len(x_dataset))
    # generate test and train dataset
    x_test_dataset = x_dataset[0:752]
    y_test_dataset = y_dataset[0:752]
    x_train_dataset = x_dataset[752:]
    y_train_dataset = y_dataset[752:]
    print("train_test_partition finished!")
    return x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset

# Partition of data set for 10-fold cross validation
def fold10_cv_partition(positive_file,negative_file):
    # generate the vectorized xs and ys
    x_dataset,y_dataset =  vectorize_data(positive_file,negative_file)
    print("data vectorization in function fold10_cv_partition finished!")
    # Divided into 10 parts
    max_len = len(x_dataset)
    m = int(max_len/10)
    # define lists storing the different partitions 
    x_train_segment = []
    y_train_segment = []
    x_validation_segment = []
    y_validation_segment = []

    # Partition of data set for 10-fold cross validation 
    # generate the first nine dataset segments 
    for i in range(9):
        x_validation_segment.append(x_dataset[m*i:m*(i+1)])
        y_validation_segment.append(y_dataset[m*i:m*(i+1)])
        x_train_segment.append(np.concatenate([x_dataset[0:m*i], x_dataset[m*(i+1):max_len]]))
        y_train_segment.append(np.concatenate([y_dataset[0:m*i], y_dataset[m*(i+1):max_len]]))                    
    # generate the last dataset segments 
    x_validation_segment.append(x_dataset[m*9:max_len])
    y_validation_segment.append(y_dataset[m*9:max_len])
    x_train_segment.append(x_dataset[0:m*9])   
    y_train_segment.append(y_dataset[0:m*9])
    print("Partition of fold 10 finished!")
    return x_train_segment,y_train_segment,x_validation_segment,y_validation_segment

if __name__ == "__main__":
    positive = "hsa_new.csv"
    negative = "pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
    train_test_partition(positive,negative)
    print(len(x_train_dataset),len(y_train_dataset))    
    print(len(x_test_dataset),len(y_test_dataset))

    x_train_segment,y_train_segment,x_validation_segment,y_validation_segment = \
    fold10_cv_partition(positive,negative)
    #print(y_train_segment[0])
    print(len(x_train_segment),len(y_train_segment))
    print(len(x_validation_segment),len(y_validation_segment))
