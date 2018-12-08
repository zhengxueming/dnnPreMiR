""" Plotting the ROC curve of CNN, RNN and CNN_RNN model on test dataset
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
#from sklearn import cross_validation
import sys
sys.path.append("./data")
from data import dataSetPartition
from keras.models import load_model

def ROC_curve():
    positive = "data/hsa_new.csv"
    negative = "data/pseudo_new.csv"
    CNN_model_path = "CNN/CNN_model.h5"
    RNN_model_path = "RNN/RNN_model.h5"
    CNNRNN_model_path = "CNN_RNN/CNNRNN_model.h5"

    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
            dataSetPartition.train_test_partition(positive,negative)
    print("load the model")
    try:
        CNN_model = load_model(CNN_model_path)
        RNN_model = load_model(RNN_model_path)
        CNNRNN_model = load_model(CNNRNN_model_path)
    except Exception:
        print("The model file doesn't exist!")
        exit(1)
    cnn_predict_result = CNN_model.predict(x_test_dataset)
    rnn_predict_result = RNN_model.predict(x_test_dataset)
    cnnrnn_predict_result = CNNRNN_model.predict(x_test_dataset)
#    print(predict_result)

    # Compute ROC curve and ROC area for each class
    cnn_fpr,cnn_tpr,cnn_threshold = roc_curve(y_test_dataset[:,0],cnn_predict_result[:,0]) 
    rnn_fpr,rnn_tpr,rnn_threshold = roc_curve(y_test_dataset[:,0],rnn_predict_result[:,0]) 
    cnnrnn_fpr,cnnrnn_tpr,cnnrnn_threshold = roc_curve(y_test_dataset[:,0],cnnrnn_predict_result[:,0]) 
     ## calculate the AUC value
    cnn_roc_auc = auc(cnn_fpr,cnn_tpr) 
    rnn_roc_auc = auc(rnn_fpr,rnn_tpr) 
    cnnrnn_roc_auc = auc(cnnrnn_fpr,cnnrnn_tpr)
    # plotting
    plt.figure(figsize=(10,10))
    plt.plot(cnn_fpr, cnn_tpr, '-',\
         linewidth=2, label='CNN model-AUC:%0.4f)' %cnn_roc_auc) 
    plt.plot(rnn_fpr, rnn_tpr, '--',\
         linewidth=2, label='RNN model-AUC:%0.4f)' %rnn_roc_auc) 
    plt.plot(cnnrnn_fpr, cnnrnn_tpr, '-.',\
         linewidth=2, label='CNN_RNN model-AUC:%0.4f)' %cnnrnn_roc_auc) 
   # plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   # plt.title('Receiver operating characteristic')
    plt.legend(loc = "center right")
    plt.savefig("ROC_curve.png",dpi=600)
    plt.show()
if __name__  == "__main__":
    ROC_curve()
