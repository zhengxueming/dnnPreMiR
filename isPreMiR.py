""" miRNA prediction of a sequence input or a file input containing many sequences
"""
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import numpy as np
import os
from keras.models import load_model
#import sys
#sys.path.append("./src/data")
#from dataVectorization import transform_xdata
#from pandas import Series


x_cast = {"A.":[1,0,0,0,0,0,0,0,0,0,0,0],"U.":[0,1,0,0,0,0,0,0,0,0,0,0],\
          "G.":[0,0,1,0,0,0,0,0,0,0,0,0],"C.":[0,0,0,1,0,0,0,0,0,0,0,0],\
          "A(":[0,0,0,0,1,0,0,0,0,0,0,0],"U(":[0,0,0,0,0,1,0,0,0,0,0,0],\
          "G(":[0,0,0,0,0,0,1,0,0,0,0,0],"C(":[0,0,0,0,0,0,0,1,0,0,0,0],\
          "A)":[0,0,0,0,0,0,0,0,1,0,0,0],"U)":[0,0,0,0,0,0,0,0,0,1,0,0],\
          "G)":[0,0,0,0,0,0,0,0,0,0,1,0],"C)":[0,0,0,0,0,0,0,0,0,0,0,1] }


def usage():
    print("""
          USAGE:
          python isPreMiR.py -s RNAsequence 
          for example: python isPreMiR.py -s 

          python isPreMiR.py -i inputFilePath [-o outputFilePath]

          """)
def seq_process(seq):
    # remove line break   
    seq = seq.strip('\n')
    # remove sapce
    seq =seq.replace(' ', '')
    # onvert the string to all uppercase
    seq = seq.upper()
    # print(seq)
    # check correctness of the RNA sequence
    for char in seq:
        #print (char)
        if char not in ["A","U","G","C"]:
            print("Please input the right RNA sequence")
            exit(1)
    return seq


def parse_opt(argv):
    infile = ''
    outfile = ''
    seq = ''
    try:
        opts, args = getopt.getopt(argv,"hs:i:o:",["sequence=","infile=","outfile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (usage())
            sys.exit(0)
        elif opt in ("-s", "--sequence"):
            seq = arg
            seq = seq_process(seq)
        elif opt in ("-i","--infile"):
            infile = arg
        elif opt in ("-o","--outfile"):
            outfile = arg
    return seq,infile,outfile

# predict the second structure of RNA sequence and return seq_struc dimer list 
# like this:["G.", "A(","G(","C(", "U(", "G(", "A.", "G(", "C(", "U(", "G.", "G("]
def second_struct_predict(seq):
    seq_struct = []
    os.system('./bin/RNAfold -i ./temp/temp_sequence.fa --noPS \
               > ./temp/temp_seq_struct.fa')
    with open("./temp/temp_seq_struct.fa","r") as fd:
        fd.readline()
        seq = fd.readline().strip()
        struct = fd.readline().strip()
    for index in range(len(seq)):
        seq_struct.append(seq[index]+struct[index])
    return seq_struct

def transform_seq_struct(seq_struct):
    SEQ_LEN = 180
    seq_struct_vector = []
    if len(seq_struct) > SEQ_LEN:
        seq_struct = seq_struct[:SEQ_LEN]
    for item in seq_struct:
        seq_struct_vector.append(x_cast[item])
    # padding to SEQ_LEN
    m_len = len(seq_struct_vector)
    for i in range(SEQ_LEN-m_len):
        seq_struct_vector.append([0,0,0,0,0,0,0,0,0,0,0,0])
    return seq_struct_vector
     

def predict_results(seq_struct_vector):
    # reload the model
    model = load_model('src/CNN/CNN_model.h5')
    # prediction
    result =  model.predict(seq_struct_vector)

    return result


def main(argv):
    #parse_opt(argv)
    seq,infile,outfile = parse_opt(argv)
    if seq:
        # write to the temp file
        with open("./temp/temp_sequence.fa","w") as fd:
            fd.write(">\n")
            fd.write(seq)
        # predict the second structure of the RNA sequence
        seq_struct = second_struct_predict(seq)
        #print(seq_struct)
        # transform to Series
        #print(seq_struct)
        seq_struct_vector = transform_seq_struct(seq_struct)
        seq_struct_vector = np.array(seq_struct_vector)
        seq_struct_vector = seq_struct_vector.reshape(1,180,12)
        #print(np.shape(seq_struct_vector))
        #print(len(seq_struct_vector))

        # predict based on the trained model.
        result = predict_results(seq_struct_vector)
        #print(result)
        # print the prediction based on the result
        print ("Your input pre-miRNA sequence: {} ".format(seq))
        #print(seq_struct)
        if np.argmax(result) == 0:
            print("Yes,it is a pre-miRNA")
        else:
            print("No,it is not a pre-miRNA")
        exit(0)

    elif infile:
        print (infile)
        # calculate the second structure
        os.system("./bin/RNAfold -i " + infile + " --noPS" \
                   + "> ./temp/temp_infile_seq_struct")
        name_list = []
        seq_list = []
        seq_struct_list = []
        seq_struct_vector_list = []
        # open and read the generated file containing the second structure into lists
        print("read the infile start:")
        fd = open("./temp/temp_infile_seq_struct","r")
        while True:
            name = fd.readline()
            if name:
                name_list.append(name)
                seq_struct = []
                seq = fd.readline().strip()
                seq_list.append(seq)
                struct = fd.readline().strip()
                for index in range(len(seq)):
                    seq_struct.append(seq[index]+struct[index])
                seq_struct_list.append(seq_struct)
            else:
                break
        fd.close()    
        
        # transform the seq_struct into vector
        for seq_struct in seq_struct_list:
            seq_struct_vector = transform_seq_struct(seq_struct)
            #print(seq_struct_vector)
            seq_struct_vector_list.append(seq_struct_vector)
            
        # transform to numpy array
        seq_struct_vector_array = np.array(seq_struct_vector_list)
        # make sure the dimension of input data 
        if len(seq_struct_vector_array) == 1: 
            seq_struct_vector_array = seq_struct_vector_array.reshape(1,180,12)
        # prediction results 
        print("prediction start:")
        result = predict_results(seq_struct_vector_array)
        #print("result:{}".format(result))
        # print(type(result))
        # write to output file or print out
        prediction = np.argmax(result,axis = 1) 
        #print("prediction:{}".format(prediction))
        if outfile:
            fd = open(outfile,"w")
            for i in range(len(name_list)):
                fd.write(name_list[i])
                fd.write(seq_list[i])
                if prediction[i] == 0:
                    fd.write("  True\n")
                else:
                    fd.write("  False\n")
                fd.write("===========================\n")
            fd.close()
        else:
            for i in range(len(name_list)):
                print(name_list[i])
                print(seq_list[i])
                if prediction[i] == 0:
                    print("True")
                else:
                    print("False")
                print("===========================")

        print("prediction finished!")
        exit(0)


if __name__ == "__main__":
   main(sys.argv[1:])
