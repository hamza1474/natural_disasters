import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.learningratefinder import LearningRateFinder
from utils.clr_callback import CyclicLR
from utils import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import sys
import os
# argparsing
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.DATASET_PATH))
data = []
labels = []
# get data and convert to arrays and onehot
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)
del imagePaths
print("[INFO] processing data...")
data = np.array(data, dtype="float32")
labels = np.array(labels)
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
data, labels = unison_shuffled_copies(data, labels)
# Onehot
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("debug0.4")
length=len(labels)
testsplit = int(length*(1-config.TEST_SPLIT))
trainX = data[:testsplit]
trainY = labels[:testsplit]
testX = data[testsplit:]
testY = labels[testsplit:]
del data
del labels
length_val = len(trainY)
valsplit = int(length_val*(1-config.VAL_SPLIT))
trainX = trainX[:valsplit]
trainY = trainY[:valsplit]
valX = trainX[valsplit:]
valY = trainY[valsplit:]

# Train Test Val split
#(trainX, testX, trainY, testY) = train_test_split(data, labels, 
#	test_size=config.TEST_SPLIT, random_state=42)
#print(trainX.shape, trainY.shape)
#(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, 
#	test_size=config.VAL_SPLIT, random_state=84)
print("debug0.6")
# data augmentation
aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
print("debug.07")
# Model
base_model = VGG16(include_top=False, weights="imagenet",
	input_tensor=Input(shape=(224, 224, 3)))

for layer in base_model.layers:
    layer.trainable = False

X = base_model.output
X = Flatten(name="flatten")(X)
X = Dense(512, activation="relu")(X)
X = Dropout(0.5)(X)
X = Dense(len(config.CLASSES), activation="softmax")(X)

model = Model(inputs=base_model.input, outputs=X)

print("[INFO] compiling model...")
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Learning Rate Finder
if args["lr_find"] > 0:
	print("[INFO] findinf learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(
		aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil((trainX.shape[0]/float(config.BATCH_SIZE))),
		epochs=20,
		batchSize=config.BATCH_SIZE)
	
	lrf.plot_loss()
	plt.savefig(config.LRFIND_PLOT_PATH)

	print("[INFO] learning rate finder complete...")
	print("[INFO] examine plot and adjust learning rate before training")
	sys.exit(0)

# Continue after adjusting Learning Rates
stepSize = config.STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size=stepSize
)
print(trainX.shape)
# train
print("[INFO] training...")
history = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
	validation_data=(valX, valY),
	steps_per_epoch=trainX.shape[0]//config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	callbacks=[clr],
	verbose=1)

# Evaluate
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=config.CLASSES))

print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

# plot

N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
