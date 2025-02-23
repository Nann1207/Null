import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = '/Users/niruba/Desktop/InceptionV3/InceptionV3model.keras'  # Please update this to the path where the Inception V3 model is saved
IMG_SIZE = (224, 224)  
LABELS_PATH = 'labels.npy'  

# To load the label encoder
if os.path.exists(LABELS_PATH):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(LABELS_PATH)  
else:
    raise FileNotFoundError(f"{LABELS_PATH} not found!")

# Load the BEST trained model which is the Inception V3
model = load_model(MODEL_PATH)

# This function is to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, IMG_SIZE)  
    img = img_to_array(img)  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img

# This function is to predict the dog breed
def predict_breed(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction)  
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]  
    return predicted_class

if __name__ == "__main__":
    print("Welcome to our Dog Breed Recognition software!")
    
    while True:

        name = input("What is your name? ")


        print(f"Hi {name}, welcome to the dog breed recognition program!")


        image_path = input("Please paste the full path of the image of the dog breed you want to test: ")
        
        if os.path.exists(image_path):
            breed = predict_breed(image_path)
            print(f"The dog breed for the image is: {breed}")
        else:
            print(f"The file at {image_path} does not exist. Please check the path and try again.")
        
       
       # Ask the user if they want to analyse another image
        another = input("Would you like to find the breed of another dog? (yes/no): ").strip().lower()
        if another != "yes":
            print("Thank you for using the DOG BREED recognition software!")
            break 
