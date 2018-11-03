""" Train the CNN model using dataset
"""
import sys
sys.path.append("../data") 
from RNNModel import RNN_model
import dataSetPartition
import keras
import os

def RNN_train(x_dataset,y_dataset):
    model = RNN_model()
    # transfer learning
    if os.path.exists("RNN_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("RNN_model_preTrained.h5")
        
    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 300,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = RNN_train(x_train_dataset,y_train_dataset)
    model.save("RNN_model_preTrained.h5")
    print("The model is saved as RNN_model_preTrained.h5 in the current directory")

