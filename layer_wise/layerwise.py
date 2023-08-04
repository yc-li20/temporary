import os
import torch
import pwcca
import librosa
import opensmile
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import signal
from transformers import Wav2Vec2Model, Wav2Vec2Processor

#load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")

#load opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

path = '/afs/inf.ed.ac.uk/user/s20/s2057508/Documents/Corpora/IEMOCAP/Data'
os.chdir(path)
files = os.listdir()

w2v_dict = {}
for i in range(13):  
    key = "w2v_{}".format(i)  
    w2v_dict[key] = np.array([])

for file in files:
        audioinput, sr = librosa.load(file_name, sr=16000)
        if len(audioinput) > 400000: #files longer than 25 seconds are not processible given the resources available
            continue

        #extract w2v feats
        with torch.no_grad():
            input_values = processor(audioinput, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            w2v_feats_all = model(input_values, output_hidden_states=True).hidden_states
            w2v_feats_dict = {}
            for layer in range(13):
                w2v_feats_dict["w2v_feats_{}".format(layer)] = w2v_feats_all[layer].to("cpu").squeeze().numpy()
                            
                if len(w2v_dict["w2v_{}".format(layer)]) > 0:
                    w2v_dict["w2v_{}".format(layer)] = np.concatenate((w2v_dict["w2v_{}".format(layer)], w2v_feats_dict["w2v_feats_{}".format(layer)]), axis=0)
                else:
                    w2v_dict["w2v_{}".format(layer)] = w2v_feats_dict["w2v_feats_{}".format(layer)]
            
            torch.cuda.empty_cache()

        #mfcc and reshape to (seq_len, feat_dim)
        mfcc_feats = librosa.feature.mfcc(y=audioinput, sr=sr, n_fft=400, hop_length=160, n_mfcc=20).T

        smile_feats = smile.process_signal(audioinput, sr)
        egmaps1, egmaps2, egmaps3, egmaps4 = [], [], [], []

        #frequency
        egmaps1.append(smile_feats['F0semitoneFrom27.5Hz_sma3nz'].values)
        egmaps1.append(smile_feats['F1frequency_sma3nz'].values)
        egmaps1.append(smile_feats['F1bandwidth_sma3nz'].values)
        egmaps1.append(smile_feats['F2frequency_sma3nz'].values)
        egmaps1.append(smile_feats['F3frequency_sma3nz'].values)

        #energy
        egmaps2.append(smile_feats['Loudness_sma3'].values)
        egmaps2.append(smile_feats['HNRdBACF_sma3nz'].values)

        #spectral
        egmaps3.append(smile_feats['alphaRatio_sma3'].values)
        egmaps3.append(smile_feats['hammarbergIndex_sma3'].values)
        egmaps3.append(smile_feats['slope0-500_sma3'].values)
        egmaps3.append(smile_feats['slope500-1500_sma3'].values)
        egmaps3.append(smile_feats['logRelF0-H1-H2_sma3nz'].values)
        egmaps3.append(smile_feats['logRelF0-H1-A3_sma3nz'].values)
        egmaps3.append(smile_feats['F1amplitudeLogRelF0_sma3nz'].values)
        egmaps3.append(smile_feats['F2amplitudeLogRelF0_sma3nz'].values)
        egmaps3.append(smile_feats['F3amplitudeLogRelF0_sma3nz'].values)

        #quality
        egmaps4.append(smile_feats['jitterLocal_sma3nz'].values)
        egmaps4.append(smile_feats['shimmerLocaldB_sma3nz'].values)

        #reshape to (seq_len, feat_dim)
        egmaps1 = np.array(egmaps1).T
        egmaps2 = np.array(egmaps2).T
        egmaps3 = np.array(egmaps3).T
        egmaps4 = np.array(egmaps4).T

        #resample to the same seq_len
        egmaps1 = signal.resample(egmaps1, len(w2v_feats_dict["w2v_feats_0"]))
        egmaps2 = signal.resample(egmaps2, len(w2v_feats_dict["w2v_feats_0"]))
        egmaps3 = signal.resample(egmaps3, len(w2v_feats_dict["w2v_feats_0"]))
        egmaps4 = signal.resample(egmaps4, len(w2v_feats_dict["w2v_feats_0"]))
        mfcc_feats = signal.resample(mfcc_feats, len(w2v_feats_dict["w2v_feats_0"]))

        #concatanation
        if 'mfcc' in globals():
            mfcc = np.concatenate((mfcc, mfcc_feats), axis=0)
            freq = np.concatenate((freq, egmaps1), axis=0)
            ener = np.concatenate((ener, egmaps2), axis=0)
            spec = np.concatenate((spec, egmaps3), axis=0)
            qual = np.concatenate((qual, egmaps4), axis=0)
        else:
            mfcc, freq, ener, spec, qual = mfcc_feats, egmaps1, egmaps2, egmaps3, egmaps4

#Layer-wise analysis
for layer in range(13):
    #PWCCA calculation for both directions
    pwcca_mfcc_1, wight_mfcc, cca_mfcc = pwcca.compute_pwcca(mfcc.T, w2v_dict["w2v_{}".format(layer)].T, epsilon=1e-10)
    pwcca_freq_1, wight_freq, cca_freq = pwcca.compute_pwcca(freq.T, w2v_dict["w2v_{}".format(layer)].T, epsilon=1e-10)
    pwcca_ener_1, wight_ener, cca_ener = pwcca.compute_pwcca(ener.T, w2v_dict["w2v_{}".format(layer)].T, epsilon=1e-10)
    pwcca_spec_1, wight_spec, cca_spec = pwcca.compute_pwcca(spec.T, w2v_dict["w2v_{}".format(layer)].T, epsilon=1e-10)
    pwcca_qual_1, wight_qual, cca_qual = pwcca.compute_pwcca(qual.T, w2v_dict["w2v_{}".format(layer)].T, epsilon=1e-10)
    #error when changing the order of the input for calculating pwcca in another direction
    # pwcca_mfcc_2, wight_mfcc, cca_mfcc = pwcca.compute_pwcca(w2v_dict["w2v_{}".format(layer)].T, mfcc.T, epsilon=1e-10)
    # pwcca_freq_2, wight_freq, cca_freq = pwcca.compute_pwcca(w2v_dict["w2v_{}".format(layer)].T, freq.T, epsilon=1e-10)
    # pwcca_ener_2, wight_ener, cca_ener = pwcca.compute_pwcca(w2v_dict["w2v_{}".format(layer)].T, ener.T, epsilon=1e-10)
    # pwcca_spec_2, wight_spec, cca_spec = pwcca.compute_pwcca(w2v_dict["w2v_{}".format(layer)].T, spec.T, epsilon=1e-10)
    # pwcca_qual_2, wight_qual, cca_qual = pwcca.compute_pwcca(w2v_dict["w2v_{}".format(layer)].T, qual.T, epsilon=1e-10)

    print('----layer', layer, '----')
    print(np.mean(cca_mfcc), np.mean(cca_freq), np.mean(cca_ener), np.mean(cca_spec), np.mean(cca_qual)) #cca calculation
    
    #error when changing the order of the input. Patterns of the layer-wise trend are not affected no matter cca or pwcca
    # print((pwcca_mfcc_1+pwcca_mfcc_2)/2, (pwcca_freq_1+pwcca_freq_2)/2, (pwcca_ener_1+pwcca_ener_2)/2, (pwcca_spec_1+pwcca_spec_2)/2, (pwcca_qual_1+pwcca_qual_2)/2) #pwcca calculation


#emotion bias
#separate files according to emotion labels and do the same process as above


#hierarchical process
# smile_feats_phone = []
# for i in range(0,len(smile_feats)-3, 3):
#     smile_feats_phone.append((smile_feats[i]+smile_feats[i+1]+smile_feats[i+2])/3.0) #phoneme level   
# smile_feats_word=[]
# for i in range(0, len(smile_feats_phone)-5, 5):
#     smile_feats_word.append((smile_feats_phone[i]+smile_feats_phone[i+1]+smile_feats_phone[i+2]+smile_feats_phone[i+3]+smile_feats_phone[i+4])/5.0) #word level


#SER model
# weighted sum
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet,self).__init__()
#         self.W = torch.nn.Parameter(torch.ones(1, 13), requires_grad=True)
#         self.lstm = nn.LSTM(input_size=768,
#                             hidden_size=64,
#                             num_layers=2,
#                             batch_first=True,
#                             dropout=0.5,
#                             bidirectional=True)
#         self.attn = nn.MultiheadAttention(128, 16, batch_first=True)
#         self.dense1 = nn.Linear(768, 128)
#         self.flat = nn.Flatten()
#         self.dense = nn.Linear(128, 16)
#         self.acti = nn.ReLU()
#         self.out = nn.Linear(16, 4)

#     def forward(self, x):
#         x = torch.matmul(self.W, x)
#         x = self.flat(x)
#         x = torch.div(x, torch.sum(self.W, 1))
#         x = self.dense1(x)
#         x = self.acti(x)
#         x = self.dense(x)
#         res = self.acti(x)
#         emotion = self.out(res)
#         return emotion, self.W
#remove weight parameter for unweighted sum or single layer input