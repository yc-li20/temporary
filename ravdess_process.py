import os, librosa
import torch  
from torch.utils.data import Dataset, DataLoader  


path = '/afs/inf.ed.ac.uk/user/s20/s2057508/Documents/Corpora/RAVDESS/Data'

names_train, names_test = [], []
emolabels_train, emolabels_test = [], []
feats_train, feats_test = [], []

# randomly split the actors into 4 folds
folds = [['2', '8', '10', '16', '18', '24'], ['1', '3', '9', '11', '19', '23'], ['4', '6', '12', '13', '20', '21'], ['5', '7', '14', '15', '17', '22']]
num_epochs = 200

class CustomDataset(Dataset):  
    def __init__(self, names, emolabels, feats):  
        self.names = names  
        self.emolabels = emolabels  
        self.feats = feats  
  
    def __len__(self):  
        return len(self.names)  
  
    def __getitem__(self, idx):  
        name = self.names[idx]  
        emolabel = self.emolabels[idx]  
        feat = torch.tensor(self.feats[idx])  
        return name, emolabel, feat  

for i in range(4): # I did this in a low-efficient way as it will extract the features four times. The dataset is small so not a big problem, but it is better that you extract the feats/labels first and then split the train/test set according to the fold
    for dirname, dirs, filenames in os.walk(path):
        for filename in filenames:
            emotion = filename[7] # this index represents the emotion label
            file = os.path.join(dirname, filename)
            audioinput, sr = librosa.load(file, sr=16000)

            if emotion == '1':
                emotion_label = 0
            elif emotion == '2':
                emotion_label = 1
            elif emotion == '3':
                emotion_label = 2
            elif emotion == '4':
                emotion_label = 3
            elif emotion == '5':
                emotion_label = 4
            elif emotion == '6':
                emotion_label = 5
            elif emotion == '7':
                emotion_label = 6
            elif emotion == '8':
                emotion_label = 7

        if dirname[-2:] not in folds[i]: #[-2:] represents the actor id
            names_train.append(file)
            emolabels_train.append(emotion_label)
            feats_train.append(audioinput)
        else:
            names_test.append(file)
            emolabels_test.append(emotion_label)
            feats_test.append(audioinput)

        train_dataset = CustomDataset(names_train, emolabels_train, feats_train)  
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = CustomDataset(names_test, emolabels_test, feats_test)  
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  

        for epoch in range(num_epochs):  
            for names, emolabels, feats in train_dataloader:  
                # Train the model here  
            
            for names, emolabels, feats in test_dataloader:  
                # Evaluate the model here  
        
