import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , Flatten, MaxPool2D
from tensorflow.keras import layers

TRAIN_PATH = "./input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv"
TEST_PATH = "./input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv"
#j and z are skipped as they require motion
LABEL_LIST =['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

class Asl:
    def __init__(self):
        self.model = self._build_model()
        self.x_train, self.y_train, \
        self.x_test, self.y_test = self._import_data()
        self.history = None
        self.checkpoint_path = "asl-model-checkpoint.keras"

    def _import_data(self):
        # reading data
        x_train = pd.read_csv(TRAIN_PATH)
        x_test = pd.read_csv(TEST_PATH)

        # store labels
        y_train = x_train['label']
        y_test = x_test['label']
        # remove labels from cvs import 
        del x_train['label']
        del x_test['label']

        # linearize pixel data
        x_train = x_train.values/255
        x_test = x_test.values/255

        # reshape to 28x28 to reduce complexity
        x_train = x_train.reshape(-1,28,28,1)
        x_test = x_test.reshape(-1,28,28,1)

        # binarize labels 
        label_binarizer = LabelBinarizer()
        y_train = label_binarizer.fit_transform(y_train)
        y_test = label_binarizer.fit_transform(y_test)
        
        return x_train, y_train, x_test, y_test

    # build keras model
    def _build_model(self):  

        model = Sequential()
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(Conv2D(32,(3,3),padding = 'same',activation = 'relu'))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(64,(3,3),padding = 'same',activation = 'relu'))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(128,(3,3),padding = 'same',activation = 'relu'))
        model.add(MaxPool2D((2,2)))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(24,activation="softmax"))
        
        # print model layers
        model.summary()

        # 
        model.compile(loss="binary_crossentropy",
                        optimizer="rmsprop",
                        metrics=["accuracy"])
                        
        return model

    # train model while saving checkpoints
    def train_model(self,num_epochs):
        #save model during training steps
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path,
                save_best_only=True,
                monitor="val_loss")
            ]
        # fit data to model
        self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test,self.y_test), epochs=num_epochs, callbacks=callbacks)

    # plot training process 
    def plot_history(self):
        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history['val_accuracy']
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()

    # print loss and accuracy of model
    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test accuracy: {score[1]}')

    # load saved keras model
    def load_trained_model(self):
        self.model.load_weights(self.checkpoint_path)

    # provide number to predict test img against trained mdeol
    def predict(self,num):
        pred = self.model.predict(np.expand_dims(self.x_test[num],axis=0))
        plt.imshow(self.x_test[num], cmap='gray')
        print("prediction: ")
        print(LABEL_LIST[np.argmax(pred)])
    
    # show asl chart to check if prediciton was correct
    def show_asl_chart(self):
        img = plt.imread("./input/sign-language-mnist/amer_sign2.png")
        plt.imshow(img)

