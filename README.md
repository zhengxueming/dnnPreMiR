###################################################################################### 
Using Trained CNN model to predict pre-miRNAs.
Data are from the human pre-miRNAs and hairpin structure sequences from human coding regions.

######################################################################################

USAGE:
          python isPreMiR.py -s RNAsequence 
          for example: python isPreMiR.py -s CUCCGGUGCCUACUGAGCUGAUAUCAGUUCUCAUUUUACACACUGGCUCAGUUCAGCAGGAACAGGA

          python isPreMiR.py -i inputFilePath [-o outputFilePath]

(note:The length of RNA should be not more than 180, all the input RNA sequence will be truncated or padded into 180 length)
      

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy
- RNAfold (/bin/RNAfold)

## Training
We trained the CNN model,RNN model and the concat model to train the dataset.

## Evaluating
Evaluate the trained model on test dataset.

