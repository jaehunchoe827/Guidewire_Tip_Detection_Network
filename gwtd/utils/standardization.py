import numpy as np

IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])

def standardize_image(image):
    """
    Standardize the image using mean and std
    Args:
        image: numpy array of shape (height, width, channels)
    Returns:
        standardized image: numpy array of shape (height, width, channels)
    """
    if image.shape[2] == 3:
        # RGB image
        image = (image - IMAGE_MEAN) / IMAGE_STD
    else:
        # Grayscale image
        image = (image - IMAGE_MEAN[0]) / IMAGE_STD[0]
    return image

def destandardize_image(image):
    """
    Destandardize the image using mean and std
    Args:
        image: numpy array of shape (height, width, channels)
    Returns:
        destandardized image: numpy array of shape (height, width, channels)
    """
    if image.shape[2] == 3:
        # RGB image
        image = image * IMAGE_STD + IMAGE_MEAN
    else:
        # Grayscale image
        image = image * IMAGE_STD[0] + IMAGE_MEAN[0]
    return image