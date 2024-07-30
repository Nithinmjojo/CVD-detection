from keras.layers import Input, Lambda, Dense, Flatten,Dropout,BatchNormalization
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from glob import glob
import tensorflow as tf
from keras.models import load_model
IMAGE_SIZE =[192, 128]

train_path = 'data/train'
valid_path = 'data/test'
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
  layer.trainable = False
folders = glob('data/train/*')

x = Flatten()(vgg.output)

# Add a fully connected layer, but with fewer units 
x = Dense(units=256, activation='relu')(x)

# Also, add a dropout layer for regularization.
x = Dropout(0.4)(x)

x = BatchNormalization()(x)  # Add batch normalization

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-5)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,
                                   rotation_range=40
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 64,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('data/test',
                                            target_size = IMAGE_SIZE,
                                            batch_size = 64,
                                            class_mode = 'categorical')
model_history = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=None,
  validation_steps=None,
  callbacks=[early_stopping,reduce_lr]
)

model.save('model_vgg19_final.h5')
#y_pred = model.predict(test_set)