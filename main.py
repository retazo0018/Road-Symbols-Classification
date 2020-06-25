import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pickle
import cv2
import random
import pandas as pd
np.random.seed(0)


def grayscale(img):
	return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def equalize(img):
	return (cv2.equalizeHist(img))

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255 # Normalization
    return img


class classify_road_symbols:
	def __init__(self):
		self.num_classes = 43

	def import_dataset(self):
		with open('data/german-traffic-signs/train.p', 'rb') as f:
			train_data = pickle.load(f)
		with open('data/german-traffic-signs/valid.p', 'rb') as f:		
			val_data = pickle.load(f)
		with open('data/german-traffic-signs/test.p', 'rb') as f:
			test_data = pickle.load(f)
	    
		self.data = pd.read_csv('data/german-traffic-signs/signnames.csv')

		self.X_train, self.y_train = train_data['features'], train_data['labels']
		self.X_val, self.y_val = val_data['features'], val_data['labels']
		self.X_test, self.y_test = test_data['features'], test_data['labels']

		assert(self.X_train.shape[0] == self.y_train.shape[0]), "The number of images is not equal to number of labels"
		assert(self.X_val.shape[0] == self.y_val.shape[0]), "The number of images is not equal to number of labels"
		assert(self.X_test.shape[0] == self.y_test.shape[0]), "The number of images is not equal to number of labels"
		assert(self.X_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 * 32 * 3"
		assert(self.X_val.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 * 32 * 3"
		assert(self.X_test.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 * 32 * 3"

		#print(self.X_train.shape)
		#print(self.data)

	def visualize_dataset(self):

		num_of_samples=[]
		 
		cols = 5
		 
		fig, axs = plt.subplots(nrows=self.num_classes, ncols=cols, figsize=(5,50))
		fig.tight_layout()
		 
		for i in range(cols):
		    for j, row in self.data.iterrows():
		        x_selected = self.X_train[self.y_train == j]
		        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
		        axs[j][i].axis("off")
		        if i == 2:
		            axs[j][i].set_title(str(j) + "-" + row["SignName"])
		            num_of_samples.append(len(x_selected))

		plt.figure(figsize=(12, 4))
		plt.bar(range(0, self.num_classes), num_of_samples)
		plt.title("Distribution of the train dataset")
		plt.xlabel("Class number")
		plt.ylabel("Number of images")
		plt.show()


	def preprocess_dataset(self):
		self.X_train = np.array(list(map(preprocessing, self.X_train)))
		self.X_val = np.array(list(map(preprocessing, self.X_val)))
		self.X_test = np.array(list(map(preprocessing, self.X_test)))

		# Reshaping
		self.X_train = self.X_train.reshape(34799, 32, 32, 1)
		self.X_test = self.X_test.reshape(12630, 32, 32, 1)
		self.X_val = self.X_val.reshape(4410, 32, 32, 1)

		# One hot Encoding the labels
		self.y_train = to_categorical(self.y_train, self.num_classes)
		self.y_test = to_categorical(self.y_test, self.num_classes)
		self.y_val = to_categorical(self.y_val, self.num_classes)



	def data_augmentation(self):
		self.datagen = ImageDataGenerator(width_shift_range=0.1 , height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
		self.datagen.fit(self.X_train)


	def build_model(self):
		self.preprocess_dataset()
		self.data_augmentation()

		self.model = Sequential()
		self.model.add(Conv2D(60, (5, 5), input_shape = (32, 32, 1), activation="relu"))
		self.model.add(Conv2D(60, (5, 5), activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Conv2D(30, (3, 3), activation="relu"))
		self.model.add(Conv2D(30, (3, 3), activation="relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.5))


		self.model.add(Flatten())
		self.model.add(Dense(500, activation="relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(self.num_classes, activation="softmax"))
		self.model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

		self.model_summary = self.model.summary()
		self.history = self.model.fit_generator(self.datagen.flow(self.X_train, self.y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=(self.X_val, self.y_val), shuffle=1)

		print(self.model_summary)

	def visualize_results(self):
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.legend(['loss', 'val_loss'])
		plt.title('Loss')
		plt.xlabel('epoch')
		plt.figure()

		plt.plot(self.history.history['accuracy'])
		plt.plot(self.history.history['val_accuracy'])
		plt.legend(['acc', 'val_acc'])
		plt.title('Accuracy')
		plt.xlabel('epoch')

		plt.show()

	def evaluate_score(self):
		self.score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
		print("Test Score: ", self.score[0])
		print("Test Accuracy: ", self.score[1])

	def save_model(self):
		self.model.save("model.h5")





ob = classify_road_symbols()
# Dataset taken from - git clone https://bitbucket.org/jadslim/german-traffic-signs
ob.import_dataset()
ob.build_model()
ob.visualize_results()
ob.evaluate_score()
ob.save_model()



