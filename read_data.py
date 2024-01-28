import firebase_admin 
from firebase_admin import credentials 
from firebase_admin import firestore 
from firebase_admin import db
import numpy as np
import pandas as pd
import math
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors


cred = credentials.Certificate('glaucoma-dc019-firebase-adminsdk-pfp4d-120a2945cb.json')
app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://glaucoma-dc019-default-rtdb.firebaseio.com'})
db = firestore.client()

for col in db.collections():
    print(col.id)

    doc_ref = db.collection(col.id)

    col = doc_ref.get()
    values = []
    for document in col:
        values.append(document.to_dict()['value'])
        print(f"Document data: {document.to_dict()['value']}")

    print(values)

    fs = 838000
    f, t, Zxx = signal.stft(values, fs, window='hann', nperseg = 256, noverlap=220, nfft= 512, return_onesided=True)
    sdb = np.abs(Zxx);
    plt.pcolormesh(t, f, sdb)
    cc = max(sdb.flatten())+[-60, 0]
    ax = plt.gca()
    ax.CLim = cc
    c = "jet"
    ax.imshow(np.flip(np.fliplr(sdb)), cmap=c, norm=colors.LogNorm(vmin=sdb.min(), vmax=sdb.max()))
    plt.axis('off')

