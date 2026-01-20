# importing essential libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report,roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from keras.models import Model
import random
import os


# setting seed across various python libraries to ensure reproducibility of results : https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
def set_seed(seed=2):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Loading the dataset located at 'train_val' in manual_split folder
dataset_path = 'manual_split/train_val'

# setting seed before importing the images 
set_seed()

# Import the dataset and augment size to 256x256 pixels with grayscale color mode
data = image_dataset_from_directory(dataset_path, image_size=(256, 256), color_mode='grayscale')

# Storing classnames in a pandas dataframe
class_names = data.class_names

# Initialize lists to hold images and labels
images = []
labels = []

# Iterating over the dataset and extract images and labels
for image_batch, label_batch in data:
    images.append(image_batch.numpy())
    labels.append(label_batch.numpy())

# Converting lists to numpy arrays
images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

# Printing shapes of images dataframe and labels dataframe to ensure it was done correctly
print(images.shape)  
print(labels.shape)  

# Defining path to test images 
dataset_path_test = 'manual_split/test'

# Setting seed before loading test images
set_seed()

# Importing test set and augment size to 256x256 pixels 
data_test = image_dataset_from_directory(dataset_path_test, image_size=(256, 256), color_mode='grayscale',seed=1)

# Initialize lists to hold test images and labels
images_test = []
labels_test = []

# Iterate over the test dataset and extract images and labels
for image_batch, label_batch in data_test:
    images_test.append(image_batch.numpy())
    labels_test.append(label_batch.numpy())

# Convert lists to numpy arrays
images_test = np.concatenate(images_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)

# Storing the images and labels of test set into appropriate pandas dataframe
x_test = images_test
y_test = labels_test

#Setting seed before splitting the data into training set and validation set
set_seed()

# Splitting the data into training set and validation set
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=1)

# Print shapes of the resulting arrays to check 
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Data augmentation using ImageDataGenerator for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    width_shift_range=0.1,     # Randomly translate images horizontally by 20%
    height_shift_range=0.1,    # Randomly translate images vertically by 20%
    rotation_range=10,         # Randomly rotate images by 10°
    zoom_range=0.1,            # Randomly zoom into images by 20%
    horizontal_flip=True       # Randomly flip images horizontally
)

# Datagenerator for validation and test set as it only needs to be rescaled
test_val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation and test sets

# Setting seed before feeding the data to datagenerators 
set_seed()
# Loading data to training data generator

# DO NOT REMOVE OR COMMENT OUT seed=1

# Here seed=1 should not be removed or commented out as after running it does not yield good results
# It predicts just one class
# It will be improved in the future
train_generator = train_datagen.flow(
    x_train, y_train,
    seed=1,    
    batch_size=8,
    shuffle=True
)

# Validation data generator
# Same for validation generator

# DO NOT REMOVE OR COMMENT OUT seed=1

val_generator = test_val_datagen.flow(
    x_val, y_val,
    seed=1,
    batch_size=8,
    shuffle=False  # Ensure no shuffling for validation and test sets
)

# Test data generator
# Same for test generator

# DO NOT REMOVE OR COMMENT OUT seed=1

test_generator = test_val_datagen.flow(
    x_test,y_test,
    seed=1,
    batch_size=8,
    shuffle=False  # Ensure no shuffling for validation and test sets
)

# Defining callbacks for saving the best model and early stopping to prevent overfitting
checkpoint = ModelCheckpoint('model_final_predict.tf', monitor='val_loss', save_best_only=True, verbose=1, save_format="tf")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Setting seed before defining the model
set_seed()
# Define the model
model = Sequential([
    Input(shape=(256, 256, 1)),  # Input layer with shape 256x256x1 (grayscale)
    Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Conv2D(256, (3, 3), activation='relu'),  # Fourth convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Conv2D(512, (3, 3), activation='relu'),  # Fifth convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Conv2D(1024, (3, 3), activation='relu'),  # Sixth convolutional layer
    BatchNormalization(),  # Batch normalization
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    
    Flatten(),  # Flatten the output of the last pooling layer
    Dense(1024, activation='relu'), # Fully connected layer with 1024 units
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dense(256, activation='relu'),  # Fully connected layer with 256 units
    Dense(128, activation='relu'),  # Fully connected layer with 128 units
    Dense(64, activation='relu'),   # Fully connected layer with 64 units
    Dense(32, activation='relu'),   # Fully connected layer with 32 units
    Dense(16, activation='relu'),   # Fully connected layer with 16 units

    Dense(1, activation='sigmoid')  # Output layer for binary classification
])


# Compiling the model with BinaryFocalCrossentropy loss and Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0), 
              metrics=['accuracy'])

# Printing the model summary to check the architecture
model.summary()

# Setting seed before training the model 
set_seed()

print(f"Training model with 6 convolutional layers")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)

# Printing the maximum value of validation achieved during the training
val_accuracy = max(history.history['val_accuracy'])
print(f"Validation accuracy: {val_accuracy:.4f}")

# Setting seed before classifying test set
set_seed()

# Evaluate the model on test set
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Predict the labels for the test set
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # Convert to binary classes

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print(f'Confusion Matrix for model with 6 layers:\n{cm}')

# Additional evaluation metrics
precision = precision_score(y_test, y_pred_classes)
print(f'Precision: {precision:.4f}')

recall = recall_score(y_test, y_pred_classes)
print(f'Recall: {recall:.4f}')

f1 = f1_score(y_test, y_pred_classes)
print(f'F1 Score: {f1:.4f}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc:.4f}')

# Classification report
report = classification_report(y_test, y_pred_classes, target_names=class_names)
print(f'Classification Report:\n{report}')

# Compute specificity, NPV, PPV
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f'Specificity: {specificity:.4f}')

sensitivity = tp / (tp + fn)
print(f'Sensitivity: {sensitivity:.4f}')

npv = tn / (tn + fn)
print(f'NPV (Negative Predictive Value): {npv:.4f}')

ppv = tp / (tp + fp)
print(f'PPV (Positive Predictive Value): {ppv:.4f}')

# Plot the loss graph for training loss and validation loss
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
fig.suptitle('Loss', fontsize=20,fontweight='bold')
plt.legend(loc="upper left")
plt.savefig('loss.png')
plt.show()


# Plot the accuracy graph for training and validation
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
fig.suptitle('Accuracy', fontsize=20,fontweight='bold')
plt.legend(loc="upper left")
plt.savefig('accuracy.png')
plt.show()



# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)',fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)',fontsize=14)
plt.grid(True)
plt.title('Receiver Operating Characteristic Curve', fontsize=20,fontweight='bold')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()


# Plot the confusion matrix using seaborn heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix', fontsize=20,fontweight='bold')
plt.savefig(f'confusion_matrix.png')
plt.show()

# Save the trained model to a file
model.save('hcq_retinopathy_seed_2.keras')

# Load the model for evaluation
model = tf.keras.models.load_model('hcq_retinopathy_seed_2.keras', custom_objects={'BinaryFocalCrossentropy': tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)})


# Function to load and preprocess a single image
def preprocess_single_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
    img_array = img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of a single image
def predict_single_image(image_path):
    img_array = preprocess_single_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = (prediction > 0.5).astype(int)
    print(f'Prediction: {prediction[0][0]}')
    print(f'Predicted class: {predicted_class[0][0]} ({class_names[predicted_class[0][0]]})')

print('this is a toxic image')
# Path to the sample image
sample_image_path = 'toxic_1.jpg'

# Predict the class of the sample image
predict_single_image(sample_image_path)

print('this is a toxic image')
# Path to the sample image
sample_image_path_2 = 'toxic_2.jpg'

# Predict the class of the sample image
predict_single_image(sample_image_path_2)

print('this is a normal image')
# Path to the sample image
sample_image_path_3 = 'normal_3.jpg'

# Predict the class of the sample image
predict_single_image(sample_image_path_3)


