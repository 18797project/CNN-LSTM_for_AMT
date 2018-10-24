# CNN-LSTM_for_AMT

preprocessing.py serves to read music pieces for dataset and convert to spectrom and label files, save them

data_loader.py serves to load the spectrom and label files of each sample, then cut them, and prepare for future use in CNN

conv.py defines a CNN net and loss function

main.py serves to train the CNN and validate it

visualize.py serves to convert from output to transcription image
