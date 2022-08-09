"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List
import scipy.linalg
import numpy as np
import collections as coll
import operator

N_DIMENSIONS = 10
NUM_PCA_AXES = 40


# KNN
def KNN(data, labels, test, k):
    d = data - test
    european_distance = np.sum(d ** 2, axis=1) ** 0.5
    distance = european_distance.argsort()

    # Deposit of the final voting results
    classK = {}
    for i in range(k):
        voteclass = labels[distance[i]]
        classK[voteclass] = classK.get(voteclass, 0) + 1

    # return label
    return sorted(classK.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    labels = []
    for t in test:
        labels.append(KNN(train, train_labels, t, 3))
    return np.array(labels)


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


# calculating the difference between two classes,and the two classes are 1-D
def divergence(class_comp, another_class):
    m_comp = np.mean(class_comp, axis=0)
    m_another = np.mean(another_class, axis=0)
    v_comp = np.var(class_comp, axis=0)
    v_another = np.var(another_class, axis=0)
    return 0.5 * (v_comp / v_another + v_another / v_comp - 2.0) + 0.5 * (m_comp - m_another) * (m_comp - m_another) * (
            1.0 / v_comp + 1.0 / v_another)


# calculating the difference between two classes,and the two classes are N-D
# features: the subset of features in class_comp and another_class
def multi_dim_divergence(class_comp, another_class, features):
    mu_comp = np.mean(class_comp[:, features], axis=0)
    mu_another = np.mean(another_class[:, features], axis=0)
    # compute distance between class_comp mean and another_class mean
    distance_mean = mu_comp - mu_another

    # compute covariance
    cov_comp = np.cov(class_comp[:, features], rowvar=0)
    cov_another = np.cov(another_class[:, features], rowvar=0)

    # compute inverse covariance matrices
    inverse_cov_comp = np.linalg.pinv(cov_comp)
    inverse_cov_another = np.linalg.pinv(cov_another)

    return (0.5 * np.trace(np.dot(inverse_cov_comp, cov_another) + np.dot(inverse_cov_another, cov_comp) - 2 * np.eye(len(features)))
            + 0.5 * np.dot(np.dot(distance_mean, inverse_cov_comp + inverse_cov_another), distance_mean))


# selecting the best te by PCA
def feature_selection(pca_train_data, labels_train):
    # arrays of different labels
    unique_labels = np.array(list(set(labels_train)))
    num_labels = len(unique_labels)

    # a list of labels tuple
    compare_list = [(unique_labels[class1], unique_labels[class2])
                    for class1 in range(num_labels)
                    for class2 in range(class1 + 1, num_labels)
                    if np.sum(labels_train == unique_labels[class1]) > 1
                    and np.sum(labels_train == unique_labels[class2]) > 1]

    # compute divergence between two different classes
    divergence_list = []
    for class1, class2 in compare_list:
        char1_data = pca_train_data[labels_train == class1, :]
        class2_data = pca_train_data[labels_train == class2, :]
        d12 = divergence(char1_data, class2_data)
        divergence_list.append(np.array(d12))

    # select 25 features that have highest divergence
    divergence_list = np.sum(np.array(divergence_list), axis=0)
    divergence_list = np.argsort(-divergence_list)[:25]

    best_feature = 1
    features_list = []
    for class1, class2 in compare_list:
        class1_data = pca_train_data[labels_train == class1, :]
        class2_data = pca_train_data[labels_train == class2, :]
        features = [best_feature]
        n_features = [feature for feature in divergence_list if feature not in features]

        # select the best N_DIMENSIONS features 
        while len(features) < N_DIMENSIONS:
            mul_dim = []
            for i in n_features:
                test_features = list(features)
                test_features.append(i)
                mul_dim.append(multi_dim_divergence(class1_data, class2_data, test_features))

            # append the testing features
            index = np.argmax(mul_dim)
            features.append(n_features[index])
            n_features.remove(n_features[index])

        features_list.append(features)

    best_final_features = np.array([feature[0] for feature in coll.Counter(np.array(features_list).flatten()).most_common(N_DIMENSIONS)])

    return best_final_features


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    pca_axes = np.array(model["pca_axes"])
    pca_train_data = PCA(data, pca_axes)
    return pca_train_data[:, model['best_features']]

# projecting data onto the principal components axes
def PCA(fvectors, pca_axes):
    pca_data = np.dot((fvectors - np.mean(fvectors)), pca_axes)
    return pca_data


def generate_principal_components_axes(fvectors_train):
    covx = np.cov(fvectors_train, rowvar=0)
    n = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(n - NUM_PCA_AXES, n - 1))
    v = np.fliplr(v)
    return v


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    pca_axes = generate_principal_components_axes(fvectors_train)
    pca_train_data = PCA(fvectors_train, pca_axes)
    best_features = feature_selection(pca_train_data, labels_train)

    model['pca_axes'] = pca_axes.tolist()
    model['best_features'] = best_features.tolist()

    fvectors_train_reduced = pca_train_data[:, model['best_features']]

    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

        In the feature vector array, each row corresponds to an image in the input list.

        Args:
            images (list[np.ndarray]): A list of input images to convert to feature vectors.

        Returns:
            np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    image_height, image_weight = images[0].shape
    n_features = image_height * image_weight
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        padded_image = np.ones(images[0].shape) * 255
        height, weight = image.shape
        height = min(height, image_height)
        weight = min(weight, image_weight)
        padded_image[0:height, 0:weight] = image[0:height, 0:weight]
        fvectors[i, :] = padded_image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)
