import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


dataset_path = '/Users/niruba/Desktop/Images'  


breed_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

label_encoder = LabelEncoder()
label_encoder.fit(breed_folders)


np.save('labels.npy', label_encoder.classes_)

print("Label encoder classes has been saved to 'labels.npy'.")
