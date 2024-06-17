import firebase_admin 
from firebase_admin import credentials 
from firebase_admin import firestore 
from firebase_admin import db
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import shutil
import os
from pathlib import Path




cred = credentials.Certificate('glaucoma-dc019-firebase-adminsdk-pfp4d-120a2945cb.json')
app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://glaucoma-dc019-default-rtdb.firebaseio.com'})
db = firestore.client()


dirpath = Path('test_app/')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs('test_app/500/', exist_ok=True)
os.makedirs('test_app/520/', exist_ok=True)
os.makedirs('test_app/540/', exist_ok=True)
os.makedirs('test_app/560/', exist_ok=True)
os.makedirs('test_app/580/', exist_ok=True)
os.makedirs('test_app/600/', exist_ok=True)

for col in db.collections():
    print(col.id)

    doc_ref = db.collection(col.id)

    col_values = doc_ref.get()
    values = []
    for document in col_values:
        values.append(document.to_dict()['value'])

    fs = 838000
    f, t, Zxx = signal.stft(values, fs, window='hann', return_onesided=True)
    sdb = np.abs(Zxx);
    plt.pcolormesh(t, f, sdb)
    cc = max(sdb.flatten())+[-60, 0]
    ax = plt.gca()
    ax.CLim = cc
    c = "jet"
    ax.imshow(np.flip(np.fliplr(sdb)), cmap=c, norm=colors.LogNorm(vmin=sdb.min(), vmax=sdb.max()))
    plt.axis('off')
    filename = '/Users/mhegde/Downloads/generate_spectograms/test_app/500/' + str(col.id) + '.png'
    plt.savefig(filename)

new_model = tf.keras.models.load_model('my_model_testPDMS',compile = False)
adam = Adam(learning_rate=0.01)
new_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['acc'])
test_datagen = ImageDataGenerator(rescale = 1.0/255.) 
test_dir = '/Users/mhegde/Downloads/generate_spectograms/test_app'
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=1,
                                                 class_mode='categorical',
                                                 target_size=(180,180),
                                                 shuffle=False)
new_model.evaluate(test_generator)