# pip install numpy, pandas, keras, sklearn, tensorflow, matplotlib 
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import numpy as np
from keras.preprocessing import image




def createModel():
      base_dir = 'src\cats_and_dogs_filtered'
      train_dir = os.path.join(base_dir, 'train')
      validation_dir = os.path.join(base_dir, 'validation')

      # Directory with our training cat pictures
      train_cats_dir = os.path.join(train_dir, 'cats')

      # Directory with our training dog pictures
      train_dogs_dir = os.path.join(train_dir, 'dogs')

      # Directory with our validation cat pictures
      validation_cats_dir = os.path.join(validation_dir, 'cats')

      # Directory with our validation dog pictures
      validation_dogs_dir = os.path.join(validation_dir, 'dogs')

      model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
      ])

      model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['acc'])

      train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
      test_datagen = ImageDataGenerator(rescale=1./255)

      # Flow training images in batches of 20 using train_datagen generator
      train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=(150, 150),  # All images will be resized to 150x150
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

      # Flow validation images in batches of 20 using test_datagen generator
      validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

      history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,  # 2000 images = batch_size * steps
            epochs=100,
            validation_data=validation_generator,
            validation_steps=50,  # 1000 images = batch_size * steps
            verbose=2)

      # model.save('../model') #For mac ../model
      # model = keras.models.load_model('../model') #For mac ../model
      model.save(os.path.join('.', 'model')) #These two lines should make the above code work for both windows and mac
model = keras.models.load_model(os.path.join('.', 'model')) 
model.summary()


def attempt_classification(path):
      # predicting images
      # path = "../download.jpg" #path to user image input that needs to be classified #(Luis A Gonzalez) For mac ../download.jpg
      img = image.load_img(path, target_size=(150, 150))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)

      images = np.vstack([x])
      classes = model.predict(images, batch_size=10)
      print(type(classes[0]))
      if classes[0]>0.5:
            print(path + " is a dog")
            
      else:
            print(path + " is a cat")

      return classes[0]

