import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# This function is to get the labels of the dataset
def get_labels(DATA_DIR):
    return sorted([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])

# This function is to preprocess images (resize and normalize)
def preprocess_image(path_to_image, img_size=128):
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# This function is to load dataset and labels from folders
def load_dataset(DATA_DIR, labels):
    X, Y = [], []
    for label_idx, label in enumerate(labels):
        label_path = os.path.join(DATA_DIR, label)
        for img_file in tqdm(os.listdir(label_path), desc=f"Loading {label}"):
            img_path = os.path.join(label_path, img_file)
            X.append(preprocess_image(img_path)[0])
            Y.append(label_idx)
    return np.array(X), np.array(Y)

# This function is to create VGG16 based model
def create_model(input_shape, num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Unfreeze more layers to allow deeper fine-tuning
    for layer in base_model.layers[:-15]:  
        layer.trainable = True
    
    x = Flatten()(base_model.output)
    
    # Corrected section with activation and dropout layers
    x = Dense(512, kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  # Apply activation after the dense layer
    x = Dropout(0.4)(x)  # Apply dropout after activation

    x = Dense(256, kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  # Apply activation after the dense layer
    x = Dropout(0.4)(x)  # Apply dropout after activation

    x = Dense(128, kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  # Apply activation after the dense layer

    
    output_layer = Dense(num_classes, activation="softmax")(x)
    
    # Compile with a lower initial learning rate
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-5),  # Slower learning rate for fine-tuning
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
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

# This function is to compute top misclassified breeds
def top_misclassified_breeds(y_true, y_pred, labels, top_n=5):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    misclassified_counts = np.sum(cm, axis=1) - np.diag(cm)
    top_misclassified_indices = np.argsort(misclassified_counts)[-top_n:][::-1]
    
    print("\nTop Misclassified Breeds:")
    misclassified_pairs = []
    
    for idx in top_misclassified_indices:
        most_confused_for = np.argmax(cm[idx])
        misclassified_pairs.append((labels[idx], labels[most_confused_for], cm[idx][most_confused_for]))
        print(f"{labels[idx]} misclassified as {labels[most_confused_for]}: {cm[idx][most_confused_for]} times")
    
    # Print most confusing pair
    misclassified_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nMost confusing pair of dog breeds:")
    if misclassified_pairs:
        most_confusing = misclassified_pairs[0]
        print(f"{most_confusing[0]} misclassified as {most_confusing[1]} ({most_confusing[2]} times)")

# This function is to visualize the correctly and incorrectly classified images
def analyze_failures(X_test, y_test, y_pred, labels):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    correct = np.where(y_pred_classes == y_test_classes)[0]
    incorrect = np.where(y_pred_classes != y_test_classes)[0]

    plt.figure(figsize=(15, 6))
    plt.suptitle('5 Correctly & 5 Incorrectly Classified Images', fontsize=16)

    for i, idx in enumerate(correct[:5]):
        plt.subplot(2, 5, i + 1)
        img_rgb = (X_test[idx] * 255).astype(np.uint8)
        plt.imshow(img_rgb)
        plt.title(f'Actual: {labels[y_test_classes[idx]]}\nPredicted: {labels[y_pred_classes[idx]]}')
        plt.axis('off')

    for i, idx in enumerate(incorrect[:5]):
        plt.subplot(2, 5, i + 6)
        img_rgb = (X_test[idx] * 255).astype(np.uint8)
        plt.imshow(img_rgb)
        plt.title(f'Actual: {labels[y_test_classes[idx]]}\nPredicted: {labels[y_pred_classes[idx]]}', color='red')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# This is the main function to run the program
if __name__ == "__main__":
    DATA_DIR = "C:/Users/Vania/OneDrive/Desktop/Images"  # Set your path here
    LABELS = get_labels(DATA_DIR)
    
    X, Y = load_dataset(DATA_DIR, LABELS)
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    num_classes = len(LABELS)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Data augmentation 
    train_datagen = ImageDataGenerator(
        featurewise_center=True, 
        featurewise_std_normalization=True, 
        rotation_range=20, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        horizontal_flip=True
    )

    train_datagen.fit(X_train)


    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Create a model
    model = create_model((128, 128, 3), num_classes)
    model.summary()
    
    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7)

    # Train the model
    history = model.fit(train_generator, validation_data=val_generator, epochs=50, callbacks=[early_stopping, reduce_lr])
    plot_training_history(history)

    # To save the model and load it
    model.save("VGG16model.keras")
    model = load_model("VGG16model.keras")

    # Test the test dataset using the trained model
    y_pred = model.predict(X_test)

    # To call compute metrics function
    compute_metrics(y_test, y_pred)

    # To call analyze_failures function
    analyze_failures(X_test, y_test, y_pred, LABELS)

    # To call top misclassified breeds function
    top_misclassified_breeds(y_test, y_pred, LABELS)
