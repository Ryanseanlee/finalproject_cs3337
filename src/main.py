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

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ = os.path.normpath(os.path.join(__location__, os.pardir))

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

      train_generator = train_datagen.flow_from_directory(
            train_dir, 
            target_size=(150, 150),  
            batch_size=20,
            class_mode='binary')

      validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

      history = model.fit_generator(
            train_generator,
            steps_per_epoch=100, 
            epochs=100,
            validation_data=validation_generator,
            validation_steps=50,  
            verbose=2)

      # model.save('../model') #For mac ../model
      # model = keras.models.load_model('../model') #For mac ../model
      
      model.save(os.path.join(__location__, 'model')) #These two lines should make the above code work for both windows and mac
model = keras.models.load_model(os.path.join(__location__, 'model')) 
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

