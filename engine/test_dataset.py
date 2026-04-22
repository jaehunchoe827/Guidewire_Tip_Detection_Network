import os
import cv2
import sys
import yaml
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
# Add project root to Python path for model loading
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from gwtd.augmentation import pixel_coords as aug_functions
from gwtd.data_loader.guidewire_data_loader import GuidewireDataSet, GuidewireDataPreprocessor
from gwtd.utils import util


def visualize_sample(dataset: GuidewireDataSet, index: int, output_image_name: str):
    image, label = dataset.__getitem__(index, get_heatmap=False)
    heatmap = dataset.convert_pixel_coord_to_heatmap(label)
    # add color channel to heatmap
    heatmap = np.expand_dims(heatmap, axis=2)
    height, width = image.shape[:2]
    plt.figure(figsize=(10, 10), dpi=300)
    # plot original image and marked image side by side, and save the figure
    marked_image = image.copy()
    x = round(label[0] * (width-1))
    y = round(label[1] * (height-1))
    print('x: ', x, 'y: ', y)
    marker_size = 2
    marked_image[y-marker_size:y+marker_size+1, x-marker_size:x+marker_size+1] = 1.0 # white color
    image_heatmap_overlay = 1.0 * image + 0.5 * heatmap
    image_heatmap_overlay = np.clip(image_heatmap_overlay, 0.0, 1.0)
    # plot in grayscale (0~1)
    plt.subplot(1, 2, 1)
    plt.imshow(marked_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image_heatmap_overlay, cmap='gray')
    plt.axis('off')
    plt.savefig(output_image_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def test_augmentation(sample, output_folder: str):
    image, label = sample
    images = []
    labels = []
    aug_names = []
    # 1 brightness
    image_augmented, label_augmented = aug_functions.augment_brightness(
        image, label, 0.2
        )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("brightness")
    print(f"Applied augmentation: brightness")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # 2 horizontal flip
    image_augmented, label_augmented = aug_functions.augment_horizontal_flip(
        image, label,
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("horizontal flip")
    print(f"Applied augmentation: horizontal flip")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # 3 vertical flip
    image_augmented, label_augmented = aug_functions.augment_vertical_flip(
        image, label,
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("vertical flip")
    print(f"Applied augmentation: vertical flip")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # 4 gaussian noise
    image_augmented, label_augmented = aug_functions.augment_gaussian_noise(
        image, label, 0.03
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("gaussian noise")
    print(f"Applied augmentation: gaussian noise")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # resize
    image_augmented, label_augmented = aug_functions.augment_resize(
        image, label, (0.8, 1.2)
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("resize")
    print(f"Applied augmentation: resize")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # shear
    image_augmented, label_augmented = aug_functions.augment_shear(
        image, label, 0.05, -0.05
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("shear")
    print(f"Applied augmentation: shear")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # rotation
    image_augmented, label_augmented = aug_functions.augment_rotation(
        image, label, 3.14159
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("rotation")
    print(f"Applied augmentation: rotation")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # scale intensity
    image_augmented, label_augmented = aug_functions.augment_scale_intensity(
        image, label, 0.8
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("scale intensity")
    print(f"Applied augmentation: scale intensity")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # saturation
    image_augmented, label_augmented = aug_functions.augment_saturation(
        image, label, 1.2
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("saturation")
    print(f"Applied augmentation: saturation")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # hue shift
    image_augmented, label_augmented = aug_functions.augment_hue_shift(
        image, label, 0.05
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("hue shift")
    print(f"Applied augmentation: hue shift")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # gaussian sharpness
    image_augmented, label_augmented = aug_functions.augment_gaussian_sharpness(
        image, label, 5, 1.5
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("gaussian sharpness")
    print(f"Applied augmentation: gaussian sharpness")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # elastic deformation
    image_augmented, label_augmented = aug_functions.augment_elastic_deformation(
        image, label, 0.004, 1.0
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("elastic deformation")
    print(f"Applied augmentation: elastic deformation")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # cutout
    image_augmented, label_augmented = aug_functions.augment_cutout(
        image, label, 0.2
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("cutout")
    print(f"Applied augmentation: cutout")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # perspective
    image_augmented, label_augmented = aug_functions.augment_perspective(
        image, label, -0.1
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("perspective")
    print(f"Applied augmentation: perspective")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # crop
    image_augmented, label_augmented = aug_functions.augment_random_crop(
        image, label, (640, 640)
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("crop")
    print(f"Applied augmentation: crop")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # mosaic
    image_augmented, label_augmented = aug_functions.augment_mosaic(
        image, label, 0.2
    )
    images.append(image_augmented)
    labels.append(label_augmented)
    aug_names.append("mosaic")
    print(f"Applied augmentation: mosaic")
    print(f"image_augmented shape: {image_augmented.shape}, label_augmented: {label_augmented}")
    # create a subplot that has two colums: one column for the original image,
    # and one column for the augmented image
    # each row is image for each augmentation
    # on both columns, plot the image and overlay the label on the image
    for i in range(len(images)):
        plt.figure(figsize=(10, 10), dpi=300)
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        image_original = image.copy()
        image_augmented = images[i].copy()
        marker_size = 2
        x_original = round(label[0] * image.shape[1])
        y_original = round(label[1] * image.shape[0])
        x_augmented = round(labels[i][0] * images[i].shape[1])
        y_augmented = round(labels[i][1] * images[i].shape[0])
        image_original[y_original-marker_size:y_original+marker_size+1, 
                       x_original-marker_size:x_original+marker_size+1] = 1.0 # white color
        image_augmented[y_augmented-marker_size:y_augmented+marker_size+1,
                        x_augmented-marker_size:x_augmented+marker_size+1] = 1.0 # white color
        plt.imshow(image_original, cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(image_augmented, cmap = 'gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, 'augmentation_test_'+aug_names[i]+'.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    # load config
    config_path = os.path.join(project_root, 'config', 'ver3_default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_path = os.path.join(project_root, 'datasets', 'guidewire')
    dataset = GuidewireDataPreprocessor(dir_dataset=dataset_path, split_ratio=[0.8, 0.1, 0.1])
    print(dataset.get_data_sample_names('train')[0])
    print(dataset.get_data_sample_names('val')[0])
    print(dataset.get_data_sample_names('test')[0])
    print(dataset.get_data_sample_names('train')[0])

    util.setup_seed()

    train_dataset = GuidewireDataSet(dataset.get_data_sample_names('train'), apply_augmentation=True, config=config)
    val_dataset = GuidewireDataSet(dataset.get_data_sample_names('val'), apply_augmentation=False, config=config)
    test_dataset = GuidewireDataSet(dataset.get_data_sample_names('test'), apply_augmentation=False, config=config)
    print('number of train samples: ', len(train_dataset))
    print('number of val samples: ', len(val_dataset))
    print('number of test samples: ', len(test_dataset))
    results_dir = os.path.join(project_root, 'results', 'dataset_test')
    os.makedirs(results_dir, exist_ok=True)
    visualize_sample(train_dataset, 0, os.path.join(results_dir, 'train_sample.jpg'))
    visualize_sample(val_dataset, 0, os.path.join(results_dir, 'val_sample.jpg'))
    visualize_sample(test_dataset, 0, os.path.join(results_dir, 'test_sample.jpg'))
    test_augmentation(val_dataset.__getitem__(0, get_heatmap=False, apply_default_noise=False), results_dir)