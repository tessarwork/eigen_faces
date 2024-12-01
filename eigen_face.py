# %%
# !pip install opencv-python
# !pip install numpy
# !pip install scikit-learn


# %%
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os

# %%
def load_images(folder, image_size=(112, 112)):
    images = []
    labels = []
    label = 0
    for subfolder in os.listdir(folder): 
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path): 
            for filename in os.listdir(subfolder_path): 
                img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None: 
                    img_resized = cv2.resize(img, image_size)
                    print(f"Loaded image shape: {img_resized.shape}")  # Debug: print image shape
                    images.append(img_resized.flatten())
                    labels.append(label)
            label+=1
    return np.array(images), np.array(labels)

# %%
def apply_pca(images, n_components=4):
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)  # Fit scaler here

    pca = PCA(n_components=n_components)
    eigenfaces = pca.fit_transform(images_scaled)
    
    return pca, eigenfaces, scaler
def train_classifier(eigenfaces, labels): 
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(eigenfaces, labels)
    return knn

def recognize_face(image, pca, knn, scaler):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (112, 112))  # Resize to the same size as training images
    image_flattened = image_resized.flatten()

    # Use the fitted scaler here
    image_scaled = scaler.transform([image_flattened])  # Transform using the fitted scaler
    image_projected = pca.transform(image_scaled)

    distances, indices = knn.kneighbors(image_projected)

    confidence_score = 1 / (1 + distances.mean())  # A simple measure: inverse of average distance
    predicted_label = knn.predict(image_projected)[0]
    print(f"Predicted label: {predicted_label}, Confidence score: {confidence_score}")

    label = knn.predict(image_projected)
    
    return predicted_label
def main():
    folder = '/Users/taufiq/workspace/eigen_faces/test_data'
    images, labels = load_images(folder)
    
    pca, eigenfaces, scaler = apply_pca(images)  # Get fitted scaler here
    
    knn = train_classifier(eigenfaces, labels)

    test_image = '/Users/taufiq/workspace/eigen_faces/neymar-1.jpeg'
    predicted_label = recognize_face(test_image, pca, knn, scaler)  # Pass the fitted scaler
    print(type(predicted_label))

    print(f"Predicted label: {predicted_label}")


# %%
main()


