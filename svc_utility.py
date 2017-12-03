import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import hog_utility


def train_classifier(cspace, orient, pix_per_cell, cells_per_block, hog_channel):
    car_images = glob.glob('training_data/vehicles/**/*.png')
    noncar_images = glob.glob('training_data/non_vehicles/**/*.png')

    car_ind = np.random.randint(0, len(car_images))
    noncar_ind = np.random.randint(0, len(noncar_images))

    # Read in car / noncar images
    car_image = mpimg.imread(car_images[car_ind])
    noncar_image = mpimg.imread(noncar_images[noncar_ind])

    # Plot the examples
    plt.imshow(car_image)
    plt.imshow(noncar_image)
    plt.show()

    car_features = hog_utility.extract_features(car_images, cspace, orient, pix_per_cell, cells_per_block, hog_channel)
    noncar_features = hog_utility.extract_features(noncar_images, cspace, orient, pix_per_cell, cells_per_block,
                                                   hog_channel)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = 0
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For them labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    joblib.dump(svc, "svc.save")

    return svc


def load_classifier():
    return joblib.load("svc.save")
