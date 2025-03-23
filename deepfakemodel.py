import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from mtcnn import MTCNN
import cv2
import numpy as np
import os

# Function to detect face using MTCNN and preprocess the image
def detect_and_crop_face(img_path, target_size=(299, 299)):
    detector = MTCNN()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)
    
    if len(detections) == 0:
        # If no face is detected, return None
        return None

    # Extract the first detected face
    x, y, width, height = detections[0]['box']
    face = img_rgb[y:y+height, x:x+width]
    
    # Resize the detected face to match the input shape for the model
    face_resized = cv2.resize(face, target_size)
    
    return face_resized

# Custom data generator that incorporates MTCNN for face detection
class MTCNNImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size=(299, 299), shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.filepaths, self.labels = self._load_filepaths_and_labels()
        self.on_epoch_end()
        
        if augment:
            self.augmenter = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.augmenter = ImageDataGenerator(rescale=1./255)
    
    def _load_filepaths_and_labels(self):
        filepaths = []
        labels = []
        for class_label, class_dir in enumerate(os.listdir(self.directory)):
            class_path = os.path.join(self.directory, class_dir)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    filepaths.append(os.path.join(class_path, filename))
                    labels.append(class_label)
        return np.array(filepaths), np.array(labels)
    
    def __len__(self):
        return len(self.filepaths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_filepaths = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        for img_path in batch_filepaths:
            face = detect_and_crop_face(img_path, target_size=self.target_size)
            if face is not None:
                # Convert face to float32 and normalize if necessary
                face = cv2.resize(face, self.target_size)
                face = np.expand_dims(face, axis=0)
                face = face.astype('float32') / 255.0  # Normalize pixel values if needed

                batch_images.append(face)

        # Convert list to array and augment if needed
        batch_images = np.array(batch_images).squeeze()  # Remove singleton dimensions
        if self.augment:
            batch_images = self.augmenter.flow(batch_images, batch_size=self.batch_size, shuffle=False).next()

        return batch_images, np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.filepaths, self.labels))
            np.random.shuffle(temp)
            self.filepaths, self.labels = zip(*temp)

# Directory paths for training, validation, and testing data
train_dir = r'C:\Users\USER\Desktop\Small SIH project\train2'  
validation_dir = r'C:\Users\USER\Desktop\Small SIH project\val2'  
test_dir = r'C:\Users\USER\Desktop\Small SIH project\test2'  

# Create custom MTCNN-based generators for train, validation, and test datasets
train_generator = MTCNNImageDataGenerator(
    train_dir,
    batch_size=32,
    target_size=(299, 299),
    augment=True
)

validation_generator = MTCNNImageDataGenerator(
    validation_dir,
    batch_size=32,
    target_size=(299, 299)
)

test_generator = MTCNNImageDataGenerator(
    test_dir,
    batch_size=32,
    target_size=(299, 299)
)

# Load the pre-trained Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom layers for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Save the trained model to a file
model.save('deepfake_detection_model_with_mtcnn.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_acc*100:.2f}%")