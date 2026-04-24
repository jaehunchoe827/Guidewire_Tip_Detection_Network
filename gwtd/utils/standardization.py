import numpy as np

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def standardize_image(image):
    """
    Standardize the image using mean and std.

    Accepts either:
      - (height, width, 3) RGB image, or
      - (height, width, 1) or (height, width) grayscale image.
    Returns the image with the same shape.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB image
        image = (image - IMAGE_MEAN) / IMAGE_STD
    else:
        # Grayscale image (2D or 3D with 1 channel)
        image = (image - IMAGE_MEAN[0]) / IMAGE_STD[0]
    return image

def destandardize_image(image):
    """
    Destandardize the image using mean and std.

    Accepts either:
      - (height, width, 3) RGB image, or
      - (height, width, 1) or (height, width) grayscale image.
    Returns the image with the same shape.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB image
        image = image * IMAGE_STD + IMAGE_MEAN
    else:
        # Grayscale image (2D or 3D with 1 channel)
        image = image * IMAGE_STD[0] + IMAGE_MEAN[0]
    return image