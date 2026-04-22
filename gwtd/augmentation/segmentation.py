# augmentation functions for segmentation mask.
# the input is image, segmentation mask where
# segmentation mask is a 3D tensor of shape (height, width, num_classes)
# here, num_classes is the number of classes in the dataset + 1 (background)
# The input image channels**must be normalized to [0, 1]**!
# the output is the augmented image and segmentation mask

import cv2
import numpy as np

def augment_brightness(image, segmentation_mask, brightness_factor):
    """
    add brightness to the image
    brightness factor should be close to 0.0
    """
    image_brightened = image + brightness_factor
    image_brightened = np.clip(image_brightened, 0, 1)
    return image_brightened, segmentation_mask


def augment_random_brightness(image, segmentation_mask, brightness_factor_range):
    """
    add random brightness (between range) to the image
    """
    brightness_factor = np.random.uniform(brightness_factor_range[0],
                                          brightness_factor_range[1])
    return augment_brightness(image, segmentation_mask, brightness_factor)


def augment_horizontal_flip(image, segmentation_mask):
    """ flip the image and segmentation mask horizontally """
    image_flipped = cv2.flip(image, 1)
    segmentation_mask_flipped = cv2.flip(segmentation_mask, 1)
    return image_flipped, segmentation_mask_flipped


def augment_vertical_flip(image, segmentation_mask):
    """ flip the image and segmentation mask vertically """
    image_flipped = cv2.flip(image, 0)
    segmentation_mask_flipped = cv2.flip(segmentation_mask, 0)
    return image_flipped, segmentation_mask_flipped


def augment_gaussian_noise(image, segmentation_mask, sigma):
    """ add gaussian noise to the image """
    noise = np.random.normal(0, sigma, image.shape)
    image_noised = image + noise
    image_noised = np.clip(image_noised, 0, 1)
    image_noised = image_noised.astype(np.float32)
    return image_noised, segmentation_mask


def augment_random_gaussian_noise(image, segmentation_mask, sigma_range):
    """
    add random gaussian noise (between range) to the image
    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return augment_gaussian_noise(image, segmentation_mask, sigma)


def augment_resize(image, segmentation_mask, size):
    """
    resize the image
    size should be a tuple of (scale_width, scale_height)
    scale_width and scale_height should be close to 1.0
    """
    scale_width, scale_height = size
    new_width = int(image.shape[1] * scale_width)
    new_height = int(image.shape[0] * scale_height)
    image_resized = cv2.resize(image, (new_width, new_height),
                               interpolation=cv2.INTER_LINEAR)
    segmentation_mask_resized = cv2.resize(segmentation_mask,
                                           (new_width, new_height),
                                           interpolation=cv2.INTER_NEAREST)
    return image_resized, segmentation_mask_resized


def augment_random_resize(image, segmentation_mask, width_range, height_range):
    """
    resize the image (between range)
    size should be a tuple of (scale_width, scale_height)
    scale_width and scale_height should be close to 1.0
    """
    scale_width = np.random.uniform(width_range[0], width_range[1])
    scale_height = np.random.uniform(height_range[0], height_range[1])
    return augment_resize(image, segmentation_mask, (scale_width, scale_height))


def augment_shear(image, segmentation_mask, shear_x, shear_y):
    """
    Apply shear transformation with expanded canvas to include all pixels
    shear_x and shear_y should be close to 0.0
    """
    height, width = image.shape[:2]
    # Find corners after transformation to determine new canvas size
    # Use actual pixel boundaries (0 to width-1, 0 to height-1)
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    # Transform corners
    expanded_corners = []
    for corner in corners:
        new_x = corner[0] + shear_x * corner[1]
        new_y = shear_y * corner[0] + corner[1]
        expanded_corners.append([new_x, new_y])
    expanded_corners = np.array(expanded_corners)
    # Find new canvas bounds
    min_x, min_y = expanded_corners.min(axis=0)
    max_x, max_y = expanded_corners.max(axis=0)
    # Use ceiling to ensure we don't truncate any pixels
    new_width = int(np.ceil(max_x - min_x)) + 1
    new_height = int(np.ceil(max_y - min_y)) + 1
    # Adjust transformation matrix to account for translation
    shear_matrix = np.float32([
        [1, shear_x, -min_x],
        [shear_y, 1, -min_y]
    ])
    # Apply transformation with expanded canvas
    image_sheared = cv2.warpAffine(image, shear_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR)
    # Transform mask
    segmentation_mask_sheared = cv2.warpAffine(segmentation_mask, shear_matrix, 
                                               (new_width, new_height),
                                               flags=cv2.INTER_NEAREST, borderValue=0)
    return image_sheared, segmentation_mask_sheared


def augment_random_shear(image, segmentation_mask, shear_x_range, shear_y_range):
    """
    apply random shear (between range) to the image
    """
    shear_x = np.random.uniform(shear_x_range[0], shear_x_range[1])
    shear_y = np.random.uniform(shear_y_range[0], shear_y_range[1])
    return augment_shear(image, segmentation_mask, shear_x, shear_y)


def augment_rotation(image, segmentation_mask, angle):
    """
    Rotate image and segmentation mask by given angle (in radians)
    """
    height, width = image.shape[:2]
    center_x, center_y = (width-1) // 2, (height-1) // 2
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(angle), 1.0)
    
    # Calculate new image dimensions to avoid cropping
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y
    
    # Apply rotation
    image_rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR)
    # Transform mask
    segmentation_mask_rotated = cv2.warpAffine(segmentation_mask, rotation_matrix,
                                               (new_width, new_height),
                                               flags=cv2.INTER_NEAREST, borderValue=0)
    return image_rotated, segmentation_mask_rotated

def augment_random_rotation(image, segmentation_mask, angle_range):
    """
    apply random rotation (between range) to the image
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    return augment_rotation(image, segmentation_mask, angle)


def augment_scale_intensity(image, segmentation_mask, scale_factor):
    """
    Scale image relative to a random reference intensity
    scale factor should be close to 1.0
    """
    reference_intensity = np.random.uniform(0.0, 1.0)
    image_scaled = scale_factor * (image - reference_intensity) + reference_intensity
    image_scaled = np.clip(image_scaled, 0, 1)
    return image_scaled, segmentation_mask


def augment_random_scale_intensity(image, segmentation_mask, scale_factor_range):
    """
    apply random scale intensity (between range) to the image
    """
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    return augment_scale_intensity(image, segmentation_mask, scale_factor)


def augment_saturation(image, segmentation_mask, saturation_factor):
    """
    Adjust saturation for color images (HSV space)
    saturation factor should be close to 1.0
    """
    if len(image.shape) == 2: # Grayscale image
        return image, segmentation_mask
    if len(image.shape) == 3 and image.shape[2] == 1: # Grayscale image
        return image, segmentation_mask
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    # Convert back to RGB
    image_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image_saturated, segmentation_mask


def augment_random_saturation(image, segmentation_mask, saturation_factor_range):
    """
    apply random saturation (between range) to the image
    """
    saturation_factor = np.random.uniform(saturation_factor_range[0], saturation_factor_range[1])
    return augment_saturation(image, segmentation_mask, saturation_factor)


def augment_hue_shift(image, segmentation_mask, hue_shift):
    """
    Shift hue for color images (HSV space)
    hue_shift should be between [0, 1.0]
    """
    if len(image.shape) == 2: # Grayscale image
        return image, segmentation_mask
    if len(image.shape) == 3 and image.shape[2] == 1: # Grayscale image
        return image, segmentation_mask
    
    # Convert to HSV (OpenCV expects uint8 input)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # OpenCV HSV ranges when image is normalized [0,1]
    # H=[0,360], S=[0,1], V=[0,1]
    hue_shift_scaled = hue_shift * 360
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_scaled) % 360
    
    # Convert back to RGB
    image_hue_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image_hue_shifted, segmentation_mask

def augment_random_hue_shift(image, segmentation_mask, hue_shift_range):
    """
    apply random hue shift (between range) to the image
    """
    hue_shift = np.random.uniform(hue_shift_range[0], hue_shift_range[1])
    return augment_hue_shift(image, segmentation_mask, hue_shift)


def augment_gaussian_sharpness(image, segmentation_mask, kernel_size, sigma):
    """
    Apply Gaussian-based blur or sharpening
    - Positive sigma: blur the image (normal Gaussian)
    - Negative sigma: sharpen the image (inverted Gaussian kernel)
    kernel_size should be odd
    sigma should be close to 0.0 (generally between -1.5 and 1.5)
    """
    # Add a check for sigma = 0 to return the original image
    # as sigma = 0 is not a valid input for cv2.getGaussianKernel
    if sigma == 0:
        return image, segmentation_mask

    # Step 1: Get Gaussian kernel using abs(sigma)
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, abs(sigma))
    kernel_2d = gaussian_kernel @ gaussian_kernel.T
    
    if sigma < 0:
        # Step 3: Negative sigma - multiply by -1 and adjust center
        kernel_2d = -kernel_2d
        
        # Step 4: Insert positive number at center to make sum = 1.0
        center_idx = kernel_size // 2
        current_sum = np.sum(kernel_2d)
        center_value = 1.0 - current_sum + kernel_2d[center_idx, center_idx]
        kernel_2d[center_idx, center_idx] = center_value
        
    # Apply the modified kernel
    image_processed = cv2.filter2D(image, -1, kernel_2d)
    image_processed = np.clip(image_processed, 0, 1)
    
    return image_processed, segmentation_mask

def augment_random_gaussian_sharpness(image, segmentation_mask, kernel_size, sigma_range):
    """
    apply random gaussian sharpness (between range) to the image
    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return augment_gaussian_sharpness(image, segmentation_mask, kernel_size, sigma)


def augment_elastic_deformation(image, segmentation_mask, alpha, sigma = 1.0):
    """
    Apply elastic deformation to image and segmentation mask
    alpha should close to 0.0 (generally between 0.0 and 0.002)
    """
    height, width = image.shape[:2]
    
    # Generate random displacement fields
    # (not normalized to [0, 1])
    dx = np.random.randn(height, width) * width * alpha
    dy = np.random.randn(height, width) * height * alpha
    
    # Apply Gaussian filter to smooth the displacement
    # (not normalized to [0, 1])
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    
    # Create coordinate grids (not normalized to [0, 1])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply displacement 
    x_new = x + dx
    y_new = y + dy
    
    # Ensure coordinates are within bounds
    x_new = np.clip(x_new, 0, width - 1)
    y_new = np.clip(y_new, 0, height - 1)
    
    # Apply elastic deformation to image and segmentation mask
    image_elastic = cv2.remap(image, x_new.astype(np.float32), y_new.astype(np.float32), 
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    segmentation_mask_elastic = cv2.remap(segmentation_mask, x_new.astype(np.float32),
                                          y_new.astype(np.float32), cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REFLECT)

    return image_elastic, segmentation_mask_elastic

def augment_random_elastic_deformation(image, segmentation_mask, alpha_range, sigma = 1.0):
    """
    apply random elastic deformation (between range) to the image
    """
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return augment_elastic_deformation(image, segmentation_mask, alpha, sigma)


def augment_cutout(image, segmentation_mask, cutout_size, max_num_cutouts=2):
    """
    Apply cutout (random erasing) augmentation
    cutout_size should be between [0.0, 1.0] (generally around 0.2)
    we first apply the cutout to the image, and then convert the safe region
    back to that part of the original image.
    """
    height, width = image.shape[:2]
    image_cutout = image.copy()
    segmentation_mask_cutout = segmentation_mask.copy()
    cutout_size_in_pixels = int(cutout_size * min(height, width))
    num_cutouts = np.random.randint(1, max_num_cutouts + 1)
    background_one_hot = np.zeros_like(segmentation_mask[0, 0, :])
    background_one_hot[0] = 1.0
    for _ in range(num_cutouts):
        # Random position for cutout
        cutout_h = min(cutout_size_in_pixels, height)
        cutout_w = min(cutout_size_in_pixels, width)
        y = np.random.randint(0, height - cutout_h + 1)
        x = np.random.randint(0, width - cutout_w + 1)
        # Apply cutout (set to mean value or random value)
        if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
            image_cutout[y:y+cutout_h, x:x+cutout_w] = np.random.uniform(0.0, 1.0)  # random value
            segmentation_mask_cutout[y:y+cutout_h, x:x+cutout_w] = background_one_hot
        else:  # Color
            image_cutout[y:y+cutout_h, x:x+cutout_w] = np.random.uniform(0.0, 1.0, 3)  # random value
            segmentation_mask_cutout[y:y+cutout_h, x:x+cutout_w] = background_one_hot
    return image_cutout, segmentation_mask_cutout

def augment_random_cutout(image, segmentation_mask, cutout_size_range, max_num_cutouts=2):
    """
    apply random cutout (between range) to the image
    """
    cutout_size = np.random.uniform(cutout_size_range[0], cutout_size_range[1])
    return augment_cutout(image, segmentation_mask, cutout_size, max_num_cutouts)


def augment_perspective(image, segmentation_mask, perspective_factor):
    """
    Apply perspective transformation to image and segmentation_mask
    - Positive perspective_factor: shrink (move corners inward)
    - Negative perspective_factor: expand (move corners outward)
    Output image has the actual distorted shape (no padding)
    perspective_factor should be close to 0.0 (generally between -0.5 and 0.5)
    """
    height, width = image.shape[:2]
    
    # Define source points (corners of the image)
    src_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    
    # Define destination points with perspective distortion
    offset_h = perspective_factor * height
    offset_w = perspective_factor * width
    dst_points = np.float32([
        [offset_w, offset_h],
        [width-1-offset_w, offset_h],
        [width-1-offset_w, height-1-offset_h],
        [offset_w, height-1-offset_h]
    ])

    # Calculate perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Find the bounding box of the transformed corners
    corners = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]])
    transformed_corners = (perspective_matrix @ corners.T).T
    
    # Convert from homogeneous coordinates
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
    
    # Calculate new canvas size
    min_x, min_y = transformed_corners.min(axis=0)
    max_x, max_y = transformed_corners.max(axis=0)
    new_width = int(np.ceil(max_x - min_x)) + 1
    new_height = int(np.ceil(max_y - min_y)) + 1
    
    # Adjust transformation matrix to account for translation
    perspective_matrix[0, 2] -= min_x
    perspective_matrix[1, 2] -= min_y
    
    # Apply perspective transformation with actual distorted shape
    image_perspective = cv2.warpPerspective(image, perspective_matrix, (new_width, new_height))
    segmentation_mask_perspective = cv2.warpPerspective(segmentation_mask, perspective_matrix, (new_width, new_height))
    
    return image_perspective, segmentation_mask_perspective

def augment_random_perspective(image, segmentation_mask, perspective_factor_range):
    """
    apply random perspective (between range) to the image
    """
    perspective_factor = np.random.uniform(perspective_factor_range[0], perspective_factor_range[1])
    return augment_perspective(image, segmentation_mask, perspective_factor)


def augment_crop(image, segmentation_mask, crop_size_scale, max_out_of_bounds_ratio = 0.1):
    """
    Apply random crop to image and segmentation_mask.
    crop_size should be a relative ratio to the original image size.
    If the cropped area includes area outside the image, then
    pad the image with random values (random rgb if color image,
    random gray if grayscale image). For segmentation mask, we pad
    the background class.
    max_out_of_bounds_ratio represents how much of padding is allowed
    around the cropped area. In the cropped image, the padded area
    on each side must be less than or equal to max_out_of_bounds_ratio * crop_size.
    """
    height, width = image.shape[:2]
    crop_width = round(crop_size_scale * width)
    crop_height = round(crop_size_scale * height)
    safe_region_size = round(max_out_of_bounds_ratio * min(crop_width, crop_height))

    dtype_image = image.dtype
    dtype_segmentation_mask = segmentation_mask.dtype

    # Calculate valid crop positions that preserve the safe region
    min_crop_x = -safe_region_size
    max_crop_x = width + safe_region_size - crop_width # inclusive
    min_crop_y = -safe_region_size
    max_crop_y = height + safe_region_size - crop_height # inclusive

    # pick a random crop position
    crop_x = np.random.randint(min_crop_x, max_crop_x + 1)
    crop_y = np.random.randint(min_crop_y, max_crop_y + 1)

    # Calculate crop boundaries in the original image coordinates
    x1 = crop_x
    y1 = crop_y
    x2 = crop_x + crop_width # exclusive
    y2 = crop_y + crop_height # exclusive
    
    # Handle out-of-bounds cropping by padding with random values
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)
    
    # Adjust crop boundaries to be within image bounds
    x1_crop = max(0, x1)
    y1_crop = max(0, y1)
    x2_crop = min(width, x2)
    y2_crop = min(height, y2)
    
    # Crop the image from the original image
    cropped_image = image[y1_crop:y2_crop, x1_crop:x2_crop]
    cropped_segmentation_mask = segmentation_mask[y1_crop:y2_crop, x1_crop:x2_crop]
    num_classes = segmentation_mask.shape[2]
    background_one_hot = np.zeros((num_classes,), dtype=dtype_segmentation_mask)
    background_one_hot[0] = np.array(1.0, dtype=dtype_segmentation_mask)
    
    # Add padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # Create the final padded image
        final_height = crop_height
        final_width = crop_width
        if len(image.shape) == 2:  # Grayscale image [H, W]
            padded_image = np.full((final_height, final_width), np.random.rand(), dtype=dtype_image)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale image [H, W, 1]
            padded_image = np.full((final_height, final_width, 1), np.random.rand(), dtype=dtype_image)
        else:  # Color image [H, W, 3]
            padded_image = np.full((final_height, final_width, 3), np.random.rand(3), dtype=dtype_image)
        padded_segmentation_mask = np.full((final_height, final_width, num_classes), background_one_hot, dtype=dtype_segmentation_mask)

        # Place the cropped image in the correct position
        start_y = pad_top
        start_x = pad_left
        end_y = start_y + cropped_image.shape[0]
        end_x = start_x + cropped_image.shape[1]
        
        padded_image[start_y:end_y, start_x:end_x] = cropped_image
        padded_segmentation_mask[start_y:end_y, start_x:end_x] = cropped_segmentation_mask
        cropped_image = padded_image
        cropped_segmentation_mask = padded_segmentation_mask

    return cropped_image, cropped_segmentation_mask

def augment_random_crop(image, segmentation_mask, crop_size_scale_range, max_out_of_bounds_ratio = 0.2):
    """
    apply random crop (between range) to the image
    """
    crop_size_scale = np.random.uniform(crop_size_scale_range[0], crop_size_scale_range[1])
    return augment_crop(image, segmentation_mask, crop_size_scale, max_out_of_bounds_ratio)