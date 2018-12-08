"""Evaluate the performance of the trained model using the test dataset
"""
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import math
import sys
sys.path.append("../data")
import dataSetPartition

# calculate TP,TN, FP and FN
def predict_comparision(y_predict,y_test):
    tp,tn,fp,fn = 0,0,0,0
    y_predict_index = np.argmax(y_predict,axis = 1)
    y_test_index = np.argmax(y_test,axis = 1)
    m = len(y_predict_index)
    for i in range(m):
        if y_predict_index[i] == 0:
            if y_test_index[i]==0:
                tp +=1    
            else:
                fp +=1
        else:
            if y_test_index[i]==1:
                tn +=1
            else:
                fn += 1 
    return tp,tn,fp,fn

# calculate SENS, SPEC, ACC and Matthews Correlation Coefficient (MCC)
def test_evaluation(model_path,x_test_dataset, y_test_dataset):
    print("load the model")
    try:
        model = load_model(model_path)
    except Exception:
        print("The model file doesn't exist!")
        exit(1)
    predict_result = model.predict(x_test_dataset)
    "calculate model's performance"
#     calculate tp,tn,fp,fn
    tp,tn,fp,fn = predict_comparision(predict_result,y_test_dataset)
    
#    calculate sens, spec, f1,mcc and acc based on tp,tn,fp,fn   
    try:
        sensitivity = float(tp)/(float(tp) + float(fn))
        specifity = float(tn)/(float(tn) + float(fp))
        precision = float(tp)/(float(tp) + float(fp))
        accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + \
                                            float(fn) + float(tn))
        f1_score = (2 * (precision * sensitivity)) / (precision + sensitivity)
        mcc = ((float(tp) * float(tn)) - (float(fp) * float(fn))) /\
                math.sqrt((float(tp) + float(fp)) * (float(tp) + float(fn))*\
                (float(tn) + float(fp)) * (float(tn) + float(fn)))
    except ZeroDivisionError as err:
        print("Exception:",err)
        exit(1)
    print("Sensitivity/recall on the test data is :{}".format(sensitivity)) 
    print("specifity on the test data is :{}".format(specifity)) 
    print("precision on the test data is :{}".format(precision))
    print("accuracy on the test data is :{}".format(accuracy))
    print("f1_score on the test data is :{}".format(f1_score))
    print("mcc on the test data is :{}".format(mcc))

    return sensitivity,specifity,f1_score,mcc,accuracy

if __name__ == "__main__":
    model_path = "RNN_model.h5"

    positive = "../data/hsa_new.csv"
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
          dataSetPartition.train_test_partition(positive,negative)

    sensitivity,specifity,f1_score,mcc,accuracy =\
    test_evaluation(model_path,x_test_dataset,y_test_dataset)
#    print(sensitivity,specifity,f1_score,mcc,accuracy)
