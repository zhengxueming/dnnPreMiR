"""Generate the random sequences with the length ranging 
   from 80 to 160 """
import numpy as np
import csv
import random

# generate 1881 RNA sequences with length of 80 to 160
def generate_random_seq(num):
    charater_collection = ["A","U","G","C"]
    random_seq = []
    for i in range(num):
        seq_len = np.random.randint(80,160)
        seq = np.random.choice(charater_collection,seq_len)
        seq = ''.join(seq)
        random_seq.append(seq)
#         print(len(seq))
    return random_seq
# write to csv file in the current directory
def write_to_file():
    random_seq_collection = generate_random_seq(1881)
    csvFile = open('psudo_random.csv','w') 
    writer = csv.writer(csvFile)
    m = len(random_seq_collection)
    for i in range(m):
        writer.writerow(["hsa_random_"+str(i+1),random_seq_collection[i],"FALSE"])
    csvFile.close()
    print("done!")
if __name__ == "__main__":
    write_to_file()
