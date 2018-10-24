# CNN-LSTM_for_AMT
The Data folder in this project is not usable.
For whole dataset, go to google drive link: 
https://drive.google.com/drive/folders/1jT1FVvRrFn_mrtsc8eEBPzCZ50dOMRSG?usp=sharing

The code includes some .py files

preprocessing.py serves to read music pieces for dataset and convert to spectrom and label files, save them

data_loader.py serves to load the spectrom and label files of each sample, then cut them, and prepare for future use in CNN

conv.py defines a CNN net and loss function

main.py serves to train the CNN and validate it

visualize.py serves to convert from output to transcription image
