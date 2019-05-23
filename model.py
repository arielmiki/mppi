import pickle
import pandas as pd
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
    feature = pd.DataFrame(AudioExtractor(filename).extract())
    index = int(MODEL.predict(feature))
    print(index)
    return KELAS[index]

print(predict('classic.mp3'))
