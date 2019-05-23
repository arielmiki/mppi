from pydub import AudioSegment
import pandas as pd
import numpy as np
import librosa
import os
import sys
import pickle as pkl


class AudioExtractor:

    def __init__(self, file):
        self.file = file
        self.features = dict(
            chroma_stft=12,
            chroma_cqt=12,
            spectral_bandwidth=1,
            spectral_contrast=1,
            spectral_rolloff=1,
            mfcc=20,
            zcr=1
        )
        self.stats = (
            'mean',
            'std',
        )

    def read_audio(self):
        data, sampling_rate = librosa.load(
            self.file, res_type='kaiser_fast', duration=30)
        return data, sampling_rate

    def count_stats(self, data):
        all_stats = {}
        for stat in self.stats:
            if stat == 'mean':
                all_stats[stat] = np.mean(data, axis=1)
            elif stat == 'std':
                all_stats[stat] = np.std(data, axis=1)
            elif stat == 'median':
                all_stats[stat] = np.median(data, axis=1)
            elif stat == 'min':
                all_stats[stat] = np.min(data, axis=1)
            elif stat == 'max':
                all_stats[stat] = np.max(data, axis=1)
        return all_stats

    def extract(self):
        data, sampling_rate = self.read_audio()
        N = len(data)//sampling_rate
        extracted = []
        for i in range(0, N, 3):
            S = np.abs(librosa.stft(data[i*sampling_rate:(i+3) * sampling_rate]))
            fixed_features = {}
            for feature in self.features:
                coeff = self.features[feature]
                counted_stats = None
                if feature == "chroma_stft":
                    counted_stats = self.count_stats(librosa.feature.chroma_stft(
                        y=data, sr=sampling_rate, n_chroma=coeff))
                elif feature == "chroma_cqt":
                    counted_stats = self.count_stats(librosa.feature.chroma_cqt(
                        y=data, sr=sampling_rate, n_chroma=coeff))
                elif feature == "chroma_cens":
                    counted_stats = self.count_stats(librosa.feature.chroma_cens(
                        y=data, sr=sampling_rate, n_chroma=coeff))

                elif feature == "spectral_centroid":
                    counted_stats = self.count_stats(
                        librosa.feature.spectral_centroid(y=data, sr=sampling_rate))
                elif feature == "spectral_bandwidth":
                    counted_stats = self.count_stats(
                        librosa.feature.spectral_bandwidth(y=data, sr=sampling_rate))
                elif feature == "spectral_contrast":
                    counted_stats = self.count_stats(
                        librosa.feature.spectral_contrast(S=S, sr=sampling_rate))
                elif feature == "spectral_rolloff":
                    counted_stats = self.count_stats(
                        librosa.feature.spectral_rolloff(y=data, sr=sampling_rate))

                elif feature == "mfcc":
                    counted_stats = self.count_stats(librosa.feature.mfcc(
                        y=data, sr=sampling_rate, n_mfcc=coeff))
                elif feature == "rmse":
                    counted_stats = self.count_stats(librosa.feature.rmse(y=data))
                elif feature == "zcr":
                    counted_stats = self.count_stats(
                        librosa.feature.zero_crossing_rate(y=data))
                for counted_stat in counted_stats:
                    for i in range(len(counted_stats[counted_stat])):
                        value = counted_stats[counted_stat][i]
                        fixed_features[feature+"_" +
                                    counted_stat+"_"+str(i+1)] = value
            extracted.append(fixed_features)
        return extracted
