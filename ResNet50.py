import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_DIR = "C:/Users/Vania/OneDrive/Desktop/Images"  #Please adjust here to load dataset
IMG_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 32  
NUM_CLASSES = 120


if not os.path.exists(DATA_DIR):
    raise ValueError(f"Dataset directory '{DATA_DIR}' does not exist. Check the path.")

# This function is to load images and labels
def load_data(directory):
    images, labels = [], []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Skipping unreadable image: {img_path}")
                        continue  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # This line is to convert to RGB
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images found. Check the dataset directory.")
    return np.array(images), np.array(labels)

# This function is to create ResNet50-based model
def create_model(input_shape, num_classes):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the intial layers 
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# This function is to compute the classification metrics
def compute_metrics(y_test, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average="weighted", zero_division=1)
    recall = recall_score(y_test_classes, y_pred_classes, average="weighted", zero_division=1)
    f1 = f1_score(y_test_classes, y_pred_classes, average="weighted", zero_division=1)
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# This function is to plot the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# This function is to compute the top misclassified breeds
def top_misclassified_breeds(y_true, y_pred, labels, top_n=5):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    misclassified_counts = np.sum(cm, axis=1) - np.diag(cm)
    top_misclassified_indices = np.argsort(misclassified_counts)[-top_n:][::-1]
    
    print("\nTop Misclassified Breeds:")
    misclassified_pairs = []
    
    for idx in top_misclassified_indices:
        most_confused_for = np.argmax(cm[idx])
        if idx != most_confused_for:   
            misclassified_pairs.append((labels[idx], labels[most_confused_for], cm[idx][most_confused_for]))
            print(f"{labels[idx]} misclassified as {labels[most_confused_for]}: {cm[idx][most_confused_for]} times")
    

    misclassified_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nMost Confusing Pair of Dog Breeds:")
    if misclassified_pairs:
        most_confusing = misclassified_pairs[0]
        print(f"{most_confusing[0]} misclassified as {most_confusing[1]} ({most_confusing[2]} times)")



# This function to visualize the correctly and incorrectly classified images 
def analyze_failures(X_test, y_test_classes, y_pred_classes, labels):
    correct = np.where(y_pred_classes == y_test_classes)[0]
    incorrect = np.where(y_pred_classes != y_test_classes)[0]

    plt.figure(figsize=(15, 6))
    plt.suptitle('5 Correctly & 5 Incorrectly Classified Images', fontsize=16)

    for i, idx in enumerate(correct[:5]):
        plt.subplot(2, 5, i + 1)
        img = X_test[idx]  
        img = img.astype(np.float32) / 255.0  
        plt.imshow(img)
        plt.title(f'Actual: {labels[y_test_classes[idx]]}\nPredicted: {labels[y_pred_classes[idx]]}')
        plt.axis('off')

    for i, idx in enumerate(incorrect[:5]):
        plt.subplot(2, 5, i + 6)
        img = X_test[idx]
        img = img.astype(np.float32) / 255.0
        plt.imshow(img)
        plt.title(f'Actual: {labels[y_test_classes[idx]]}\nPredicted: {labels[y_pred_classes[idx]]}', color='red')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    images, labels = load_data(DATA_DIR)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded, NUM_CLASSES)
    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels_categorical, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    # Data Augmentation 
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_generator = val_test_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=True)
    test_generator = val_test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    model = create_model((*IMG_SIZE, CHANNELS), NUM_CLASSES)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint("ResNet50model.h5", monitor="val_loss", save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)
    
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        verbose=2,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Fine tune the model
    model.layers[0].trainable = True
    num_layers = len(model.layers[0].layers)
    for layer in model.layers[0].layers[max(0, num_layers-50):]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_finetune = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        verbose=2,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    y_pred = model.predict(test_generator)
    compute_metrics(y_test, y_pred)
    plot_training_history(history_finetune)

    top_misclassified_breeds(y_test, y_pred, label_encoder.classes_)

    # Analyse Failures
    analyze_failures(X_test, np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), label_encoder.classes_)
    
