""" Train the CNN model using dataset
"""
import sys
sys.path.append("../data") 
from CNNModel import CNN_model
import dataSetPartition
import keras
import os


def CNN_train(x_dataset,y_dataset):
    model = CNN_model()
    if os.path.exists("CNN_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("CNN_model_preTrained.h5")
    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 600,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = CNN_train(x_train_dataset,y_train_dataset)
#   model.save("CNN_model_preTrained.h5")
#   print("The model is saved as CNN_model_preTrained.h5 in the current directory")

