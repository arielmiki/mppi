import pickle
import pandas as pd
import numpy as np
from ekstraksi import AudioExtractor

with open('model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

KELAS = [
    "classical",
    "country",
    "hip-hop",
    "metal",
]


def predict(filename):
    features = AudioExtractor(filename).extract()
    features = pd.DataFrame(features)
    return KELAS[int(np.mean(MODEL.predict(features)))]
