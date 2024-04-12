from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

model_name = "VGGNet.keras"
rows = 128
cols = 128
channels = 3
data_directory = './data/nifti-results'

if(path.exists(model_name)) :
    model = keras.models.load_model(model_name)
    print(model.summary())
    print("==========Details==========")
    for layer in model.layers:
        print("Layer Name:", layer.name)
        print("Layer Type:", type(layer).__name__)
        print("Output Shape:", layer.output_shape)
        print("Number of Trainable Parameters:", layer.count_params())
        print()
else :
    train_dir = data_directory
    test_dir = data_directory
    validation_dir = data_directory

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_batch_size = 16

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    #Used to get the same results everytime
    np.random.seed(42)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(rows,cols),
        batch_size=train_batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(rows,cols),
        batch_size=20,
        class_mode='categorical')

    ######################################################################
    #initialize the NN

    #Load the VGG16 model, use the ILSVRC competition's weights
    #include_top = False, means only include the Convolution Base (do not import the top layers or NN Layers)
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(rows,cols,channels))
    conv_base.trainable = False
    model = models.Sequential()

    #Add the VGGNet model
    model.add(conv_base)

    #NN Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dense(2,activation='softmax'))

    print(model.summary())
    ######################################################################

    #Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    #Steps per epoch = Number of images in the training directory / batch_size (of the generator)
    #validation_steps = Number of images in the validation directory / batch_size (of the generator)
    checkpoint_callback = keras.callbacks.ModelCheckpoint("%s" % (model_name), save_best_only=True)

    model_history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint_callback])

    #Plot the model
    pd.DataFrame(model_history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

    model.save(model_name)