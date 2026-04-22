# augmentation functions for pixel coordinates.
# the input is image, coords where coords is a list of [x, y]
# The input coords **must be normalized to [0, 1]**
# the output is the augmented image and coords

import cv2
import numpy as np

def augment_brightness(image, coords, brightness_factor):
    """
    add brightness to the image
    brightness factor should be close to 0.0
    """
    image_brightened = image + brightness_factor
    image_brightened = np.clip(image_brightened, 0, 1)
    return image_brightened, coords


def augment_random_brightness(image, coords, brightness_factor_range):
    """
    add random brightness (between range) to the image
    """
    brightness_factor = np.random.uniform(brightness_factor_range[0],
                                          brightness_factor_range[1])
    return augment_brightness(image, coords, brightness_factor)


def augment_horizontal_flip(image, coords):
    """ flip the image and coords horizontally """
    image_flipped = cv2.flip(image, 1)
    coords_flipped = [1.0 - coords[0], coords[1]]
    return image_flipped, coords_flipped


def augment_vertical_flip(image, coords):
    """ flip the image and coords vertically """
    image_flipped = cv2.flip(image, 0)
    coords_flipped = [coords[0], 1.0 - coords[1]]
    return image_flipped, coords_flipped


def augment_gaussian_noise(image, coords, sigma):
    """ add gaussian noise to the image """
    noise = np.random.normal(0, sigma, image.shape)
    image_noised = image + noise
    image_noised = np.clip(image_noised, 0, 1)
    image_noised = image_noised.astype(np.float32)
    return image_noised, coords


def augment_random_gaussian_noise(image, coords, sigma_range):
    """
    add random gaussian noise (between range) to the image
    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return augment_gaussian_noise(image, coords, sigma)


def augment_resize(image, coords, size):
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
    return image_resized, coords


def augment_random_resize(image, coords, width_range, height_range):
    """
    resize the image (between range)
    size should be a tuple of (scale_width, scale_height)
    scale_width and scale_height should be close to 1.0
    """
    scale_width = np.random.uniform(width_range[0], width_range[1])
    scale_height = np.random.uniform(height_range[0], height_range[1])
    return augment_resize(image, coords, (scale_width, scale_height))


def augment_shear(image, coords, shear_x, shear_y):
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
    # apply random color to the border of the image
    # .tolist() converts the numpy array to a list [B, G, R].
    random_bgr_color = np.random.random(3).tolist()
    # Apply transformation with expanded canvas
    image_sheared = cv2.warpAffine(image, shear_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR, borderValue=random_bgr_color)
    # Transform coordinates
    pixel_x = coords[0] * (width - 1)
    pixel_y = coords[1] * (height - 1)
    pixel_x_new = pixel_x + shear_x * pixel_y - min_x
    pixel_y_new = shear_y * pixel_x + pixel_y - min_y
    # Normalize to new canvas size
    coords_sheared = [pixel_x_new / (new_width - 1), pixel_y_new / (new_height - 1)]
    return image_sheared, coords_sheared


def augment_random_shear(image, coords, shear_x_range, shear_y_range):
    """
    apply random shear (between range) to the image
    """
    shear_x = np.random.uniform(shear_x_range[0], shear_x_range[1])
    shear_y = np.random.uniform(shear_y_range[0], shear_y_range[1])
    return augment_shear(image, coords, shear_x, shear_y)


def augment_rotation(image, coords, angle):
    """
    Rotate image and coordinates by given angle (in radians)
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
    
    # apply random color to the border of the image
    # .tolist() converts the numpy array to a list [B, G, R].
    random_bgr_color = np.random.random(3).tolist()

    # Apply rotation
    image_rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR, borderValue=random_bgr_color)
    # Transform coordinates
    pixel_x = coords[0] * (width - 1)
    pixel_y = coords[1] * (height - 1)
    # Apply rotation to coordinates
    coords_homogeneous = np.array([pixel_x, pixel_y, 1])
    rotated_coords = rotation_matrix @ coords_homogeneous
    # Normalize to new image size
    coords_rotated = [rotated_coords[0] / (new_width - 1), rotated_coords[1] / (new_height - 1)]
    return image_rotated, coords_rotated

def augment_random_rotation(image, coords, angle_range):
    """
    apply random rotation (between range) to the image
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    return augment_rotation(image, coords, angle)


def augment_scale_intensity(image, coords, scale_factor):
    """
    Scale image relative to a random intensity
    scale factor should be close to 1.0
    """
    reference_intensity = np.random.uniform(0.0, 1.0)
    image_scaled = scale_factor * (image - reference_intensity) + reference_intensity
    image_scaled = np.clip(image_scaled, 0, 1)
    return image_scaled, coords


def augment_random_scale_intensity(image, coords, scale_factor_range):
    """
    apply random scale intensity (between range) to the image
    """
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    return augment_scale_intensity(image, coords, scale_factor)


def augment_saturation(image, coords, saturation_factor):
    """
    Adjust saturation for color images (HSV space)
    saturation factor should be close to 1.0
    """
    if len(image.shape) == 2: # Grayscale image
        return image, coords
    if len(image.shape) == 3 and image.shape[2] == 1: # Grayscale image
        return image, coords
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Convert back to RGB
    image_saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image_saturated, coords


def augment_random_saturation(image, coords, saturation_factor_range):
    """
    apply random saturation (between range) to the image
    """
    saturation_factor = np.random.uniform(saturation_factor_range[0], saturation_factor_range[1])
    return augment_saturation(image, coords, saturation_factor)


def augment_hue_shift(image, coords, hue_shift):
    """
    Shift hue for color images (HSV space)
    hue_shift should be between [0, 1.0]
    """
    if len(image.shape) == 2: # Grayscale image
        return image, coords
    if len(image.shape) == 3 and image.shape[2] == 1: # Grayscale image
        return image, coords
    
    # Convert to HSV (OpenCV expects uint8 input)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # OpenCV HSV ranges when image is normalized [0,1]
    # H=[0,360], S=[0,1], V=[0,1]
    hue_shift_scaled = hue_shift * 360
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_scaled) % 360
    
    # Convert back to RGB
    image_hue_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image_hue_shifted, coords


def augment_random_hue_shift(image, coords, hue_shift_range):
    """
    apply random hue shift (between range) to the image
    """
    hue_shift = np.random.uniform(hue_shift_range[0], hue_shift_range[1])
    return augment_hue_shift(image, coords, hue_shift)


def augment_gaussian_sharpness(image, coords, kernel_size, sigma):
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
        return image, coords

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
    
    return image_processed, coords


def augment_random_gaussian_sharpness(image, coords, kernel_size, sigma_range):
    """
    apply random gaussian sharpness (between range) to the image
    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return augment_gaussian_sharpness(image, coords, kernel_size, sigma)


def augment_elastic_deformation(image, coords, alpha, sigma = 1.0):
    """
    Apply elastic deformation to image and coordinates
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
    
    # Apply elastic deformation to image
    image_elastic = cv2.remap(image, x_new.astype(np.float32), y_new.astype(np.float32), 
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Transform coordinates
    pixel_x = coords[0] * (width - 1)
    pixel_y = coords[1] * (height - 1)
    
    # Apply displacement to coordinates
    new_pixel_x = pixel_x + dx[round(pixel_y), round(pixel_x)]
    new_pixel_y = pixel_y + dy[round(pixel_y), round(pixel_x)]
    
    # Clip coordinates after applying displacement
    new_pixel_x = np.clip(new_pixel_x, 0, width - 1)
    new_pixel_y = np.clip(new_pixel_y, 0, height - 1)
    
    # Normalize coordinates
    coords_elastic = [new_pixel_x / (width - 1), new_pixel_y / (height - 1)]
    return image_elastic, coords_elastic


def augment_random_elastic_deformation(image, coords, alpha_range, sigma = 1.0):
    """
    apply random elastic deformation (between range) to the image
    """
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return augment_elastic_deformation(image, coords, alpha, sigma)


def augment_cutout(image, coords, cutout_size, num_cutouts=2, safe_reigon=0.1):
    """
    Apply cutout (random erasing) augmentation
    cutout_size should be between [0.0, 1.0] (generally between 0.0 and 0.2)
    we first apply the cutout to the image, and then convert the safe region
    back to that part of the original image.
    """
    height, width = image.shape[:2]
    image_cutout = image.copy()
    cutout_size_in_pixels = int(cutout_size * min(height, width))
    for _ in range(num_cutouts):
        # Random position for cutout
        cutout_h = min(cutout_size_in_pixels, height)
        cutout_w = min(cutout_size_in_pixels, width)
        y = np.random.randint(0, height - cutout_h + 1)
        x = np.random.randint(0, width - cutout_w + 1)
        # Apply cutout (set to mean value or random value)
        if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
            image_cutout[y:y+cutout_h, x:x+cutout_w] = np.random.uniform(0.0, 1.0)  # random value
        else:  # Color
            image_cutout[y:y+cutout_h, x:x+cutout_w] = np.random.uniform(0.0, 1.0, 3)  # random value
    # define safe region (region around the coords)
    safe_region_half_size = int(safe_reigon * min(height, width))
    pixel_y = round(coords[1] * (height - 1))
    pixel_x = round(coords[0] * (width - 1))
    x1 = max(0, pixel_x - safe_region_half_size)
    x2 = min(width - 1, pixel_x + safe_region_half_size + 1)
    y1 = max(0, pixel_y - safe_region_half_size) 
    y2 = min(height - 1, pixel_y + safe_region_half_size + 1)
    image_cutout[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return image_cutout, coords


def augment_random_cutout(image, coords, cutout_size_range, max_num_cutouts=2, safe_reigon=0.1):
    """
    apply random cutout (between range) to the image
    """
    num_cutouts = np.random.randint(1, max_num_cutouts + 1)
    cutout_size = np.random.uniform(cutout_size_range[0], cutout_size_range[1])
    return augment_cutout(image, coords, cutout_size, num_cutouts, safe_reigon)


def augment_perspective(image, coords, perspective_factor):
    """
    Apply perspective transformation to image and coordinates
    - Positive perspective_factor: shrink (move corners inward)
    - Negative perspective_factor: expand (move corners outward)
    Output image has the actual distorted shape (no padding)
    perspective_factor should be close to 0.0 (generally between -0.1 and 0.1)
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
    
    # Transform coordinates
    pixel_x = coords[0] * (width - 1)
    pixel_y = coords[1] * (height - 1)
    
    # Apply perspective transformation to coordinates
    coords_homogeneous = np.array([pixel_x, pixel_y, 1])
    transformed_coords = perspective_matrix @ coords_homogeneous
    
    # Normalize coordinates to new canvas size
    coords_perspective = [np.clip(transformed_coords[0] / transformed_coords[2] / (new_width - 1), 0, 1),
                         np.clip(transformed_coords[1] / transformed_coords[2] / (new_height - 1), 0, 1)]
    
    return image_perspective, coords_perspective


def augment_random_perspective(image, coords, perspective_factor_range):
    """
    apply random perspective (between range) to the image
    """
    perspective_factor = np.random.uniform(perspective_factor_range[0], perspective_factor_range[1])
    return augment_perspective(image, coords, perspective_factor)


def augment_random_crop(image, coords, crop_size, safe_reigon=0.1):
    """
    Apply random crop to image and coordinates.
    crop_size should be a tuple of (crop_width, crop_height).
    Here, crop_size are pixel values, not normalized to [0, 1].
    The crop position is determined so that the safe region
    around the coords is preserved in the cropped image.
    If the cropped area includes area outside the image, then
    pad the image with random values (random rgb if color image,
    random gray if grayscale image).
    """
    height, width = image.shape[:2]
    crop_width = crop_size[0]
    crop_height = crop_size[1]

    # Calculate safe region size
    safe_region_half_size = round(safe_reigon * min(height, width))
    
    # just return the original image if the crop size is too small
    if crop_width < 2 * safe_region_half_size or crop_height < 2 * safe_region_half_size:
        return image, coords

    # Convert coordinates to pixel values
    pixel_x_int = round(coords[0] * (width - 1))
    pixel_y_int = round(coords[1] * (height - 1))

    # Calculate valid crop positions that preserve the safe region
    min_crop_x = pixel_x_int + safe_region_half_size - crop_width + 1
    max_crop_x = pixel_x_int - safe_region_half_size
    min_crop_y = pixel_y_int + safe_region_half_size - crop_height + 1
    max_crop_y = pixel_y_int - safe_region_half_size

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
    
    # Add padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # Create the final padded image
        final_height = crop_height
        final_width = crop_width
        
        if len(image.shape) == 2:  # Grayscale image [H, W]
            padded_image = np.full((final_height, final_width), np.random.rand())
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale image [H, W, 1]
            padded_image = np.full((final_height, final_width, 1), np.random.rand())
        else:  # Color image [H, W, 3]
            padded_image = np.full((final_height, final_width, 3), np.random.rand(3))
        
        # Place the cropped image in the correct position
        start_y = pad_top
        start_x = pad_left
        end_y = start_y + cropped_image.shape[0]
        end_x = start_x + cropped_image.shape[1]
        
        padded_image[start_y:end_y, start_x:end_x] = cropped_image
        cropped_image = padded_image
    
    # Update coordinates relative to the cropped image
    # Account for the original crop position and any padding
    new_pixel_x = pixel_x_int - crop_x
    new_pixel_y = pixel_y_int - crop_y
    
    # Normalize coordinates to cropped image size
    coords_cropped = [new_pixel_x / (crop_width - 1), new_pixel_y / (crop_height - 1)]

    return cropped_image, coords_cropped


def augment_mosaic(image, coords, mosaic_size, num_mosaics=2, safe_reigon=0.1):
    """
    Apply mosaic (random mosaic) augmentation
    mosaic_size should be between [0.0, 1.0] (generally between 0.0 and 0.2)
    we first crop a random region, and then randomly blend it with the
    original image. Then, we convert the safe region back to that part of
    the original image.
    """
    height, width = image.shape[:2]
    image_mosaic = image.copy()
    mosaic_size_in_pixels = int(mosaic_size * min(height, width))
    for _ in range(num_mosaics):
        # mosaic width and height
        mosaic_h = min(mosaic_size_in_pixels, height)
        mosaic_w = min(mosaic_size_in_pixels, width)
        # from
        y_from = np.random.randint(0, height - mosaic_h + 1)
        x_from = np.random.randint(0, width - mosaic_w + 1)
        image_from = image_mosaic[y_from:y_from+mosaic_h, x_from:x_from+mosaic_w]
        # to
        y_to = np.random.randint(0, height - mosaic_h + 1)
        x_to = np.random.randint(0, width - mosaic_w + 1)
        image_to = image_mosaic[y_to:y_to+mosaic_h, x_to:x_to+mosaic_w]
        # Apply mosaic
        alpha_blend = np.random.uniform(0.2, 0.8)
        image_mosaic[y_to:y_to+mosaic_h, x_to:x_to+mosaic_w] = (
            image_from * alpha_blend + image_to * (1-alpha_blend)
        )
    # define safe region (region around the coords)
    safe_region_half_size = int(safe_reigon * min(height, width))
    pixel_y = round(coords[1] * (height - 1))
    pixel_x = round(coords[0] * (width - 1))
    x1 = max(0, pixel_x - safe_region_half_size)
    x2 = min(width - 1, pixel_x + safe_region_half_size + 1)
    y1 = max(0, pixel_y - safe_region_half_size) 
    y2 = min(height - 1, pixel_y + safe_region_half_size + 1)
    image_mosaic[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return image_mosaic, coords


def augment_random_mosaic(image, coords, mosaic_size_range, max_num_mosaics=2, safe_reigon=0.1):
    """
    apply random mosaic (between range) to the image
    """
    num_mosaics = np.random.randint(1, max_num_mosaics + 1)
    mosaic_size = np.random.uniform(mosaic_size_range[0], mosaic_size_range[1])
    return augment_mosaic(image, coords, mosaic_size, num_mosaics, safe_reigon)