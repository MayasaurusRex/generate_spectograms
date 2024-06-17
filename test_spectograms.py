import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,SGD,Adam

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