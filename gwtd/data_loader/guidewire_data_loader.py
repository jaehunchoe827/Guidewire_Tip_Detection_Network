import os
import cv2
import numpy
import random
import numpy as np
import torch
from torch.utils import data
from gwtd.augmentation import pixel_coords as aug_functions
from gwtd.utils.standardization import standardize_image, destandardize_image


class GuidewireDataSet(data.Dataset):
    def __init__(self, data_sample_names: list, apply_augmentation: bool = False, config: dict = None, apply_standardization: bool = True):
        self.data_sample_names = data_sample_names
        self.apply_augmentation = apply_augmentation
        self.config = config
        self.augmentation_config = self.config['dataset']['augmentation']
        self.image_initial_resize_ratio = self.config['dataset']['image_initial_resize_ratio']
        self.image_output_size = self.config['network']['input_image_shape']
        self.heatmap_sigma = self.config['dataset']['heatmap_sigma']
        self.sigma_xray_noise = self.config['dataset']['sigma_xray_noise']
        self.apply_standardization = apply_standardization
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_sample_names)
    
    def __getitem__(self, index, get_heatmap: bool = True, apply_default_noise = True):
        image_path, label_path = self.data_sample_names[index]
        image_cv = cv2.imread(image_path).astype(np.float32)
        # from BGR to RGB, and normalize to [0, 1]
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) / 255.0
        height, width = image.shape[:2]
        # read the label from txt file.
        # inside the txt file, the label is in the format of [1, x, y]
        label = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            info = line.split(' ')
            # the label starts with dummy value 1, so we ignore
            # info[0] and start from info[1]
            # the raw coordinates are normalized 
            # by the width and height of the image
            label.append(int(info[1]) / (width-1))
            label.append(int(info[2]) / (height-1))
        label = numpy.array(label)
        # resize the image to the initial size
        image = cv2.resize(image, (round(width * self.image_initial_resize_ratio), round(height * self.image_initial_resize_ratio)))
        image, label = self.augment_sample(image, label)
        # convert rgb image to grayscale image
        # Some numpy operations can upcast to float64; cv2.cvtColor doesn't support CV_64F.
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # if apply_augmentation = False, apply random gaussian noise to the image
        if not self.apply_augmentation and apply_default_noise:
            image, label = aug_functions.augment_gaussian_noise(image, label,
                                                                sigma=self.sigma_xray_noise)
        if get_heatmap:
            # finally, convert the pixel coord label to heatmap label
            label = self.convert_pixel_coord_to_heatmap(label)
        # add color channel dimension
        image = image.reshape(image.shape[0], image.shape[1], 1)
        # standardize the image
        if self.apply_standardization:
            image = standardize_image(image)
        return image, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)

        # __getitem__ returns images shaped (H, W, 1). Stack -> (B, H, W, 1)
        # and then permute to (B, 1, H, W) which is what the model expects.
        images_np = np.stack(images, axis=0).astype(np.float32, copy=False)
        images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).contiguous()

        # labels: this should work for both pixel coord and heatmap coord
        labels_tensor = torch.stack([
            torch.as_tensor(label, dtype=torch.float32)
            for label in labels
        ], dim=0)

        return images_tensor, labels_tensor

    def convert_pixel_coord_to_heatmap(self, label):
        """
        Returns a (height, width) array G with a Gaussian centered at (x, y).
        """
        height, width = self.image_output_size
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)  # Y: rows, X: cols
        x_truth = label[0] * (width-1)
        y_truth = label[1] * (height-1)
        G = np.exp(-0.5 * ((X - x_truth) ** 2 / (self.heatmap_sigma**2) +
                        (Y - y_truth) ** 2 / (self.heatmap_sigma**2)))
        return G

    def augment_sample(self, image, label):
        """
        Apply augmentations to image and label based on config
        Returns augmented image and label
        """
        image_augmented = image.copy()
        label_augmented = label.copy()
        if self.apply_augmentation:
            # Apply each augmentation with its probability
            for aug_name, aug_params in self.augmentation_config.items():
                if np.random.random() < aug_params['probability']:
                    if aug_name == 'random_brightness':
                        image_augmented, label_augmented = aug_functions.augment_random_brightness(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_horizontal_flip':
                        image_augmented, label_augmented = aug_functions.augment_horizontal_flip(
                            image_augmented, label_augmented,
                        ) # no args
                    elif aug_name == 'random_vertical_flip':
                        image_augmented, label_augmented = aug_functions.augment_vertical_flip(
                            image_augmented, label_augmented,
                        ) # no args
                    elif aug_name == 'random_gaussian_noise':
                        image_augmented, label_augmented = aug_functions.augment_random_gaussian_noise(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_resize':
                        image_augmented, label_augmented = aug_functions.augment_random_resize(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_shear':
                        image_augmented, label_augmented = aug_functions.augment_random_shear(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_rotation':
                        image_augmented, label_augmented = aug_functions.augment_random_rotation(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_scale_intensity':
                        image_augmented, label_augmented = aug_functions.augment_random_scale_intensity(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_saturation':
                        image_augmented, label_augmented = aug_functions.augment_random_saturation(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_hue_shift':
                        image_augmented, label_augmented = aug_functions.augment_random_hue_shift(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_gaussian_sharpness':
                        image_augmented, label_augmented = aug_functions.augment_random_gaussian_sharpness(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_elastic_deformation':
                        image_augmented, label_augmented = aug_functions.augment_random_elastic_deformation(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_mosaic':
                        image_augmented, label_augmented = aug_functions.augment_random_mosaic(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_cutout':
                        image_augmented, label_augmented = aug_functions.augment_random_cutout(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_perspective':
                        image_augmented, label_augmented = aug_functions.augment_random_perspective(
                            image_augmented, label_augmented,
                            **aug_params['args']
                        )
                    elif aug_name == 'random_crop':
                        image_augmented, label_augmented = aug_functions.augment_random_crop(
                            image_augmented, label_augmented,
                             # height, width -> width, height
                            crop_size=(self.image_output_size[1], self.image_output_size[0]),
                            **aug_params['args']
                        )
                    else:
                        raise ValueError(f"Unknown augmentation name: {aug_name}")
        # if the image size is not self.image_output_size, resize it
        if image_augmented.shape[:2] != self.image_output_size:
            image_augmented = cv2.resize(image_augmented, (self.image_output_size[1], self.image_output_size[0]))
        return image_augmented, label_augmented


class GuidewireDataPreprocessor():
    def __init__(self, dir_dataset: str, split_ratio: list):
        self.dir_dataset = dir_dataset
        assert len(split_ratio) == 3
        self.data_sample_names = self._load_data(self.dir_dataset)
        self.n_total_samples = len(self.data_sample_names)
        # shuffle data_sample_names tuple for 1000 times
        # for reproducibility, set random seed here
        random.seed(42)
        for _ in range(100): # random shuffle
            random.shuffle(self.data_sample_names)
        n_train = round(len(self.data_sample_names) * split_ratio[0])
        n_val = round(len(self.data_sample_names) * split_ratio[1])
        n_test = len(self.data_sample_names) - n_train - n_val
        self.train_data_sample_names = self.data_sample_names[:n_train]
        self.val_data_sample_names = self.data_sample_names[n_train:n_train + n_val]
        self.test_data_sample_names = self.data_sample_names[n_train + n_val:]
        
    def _load_data(self, dir_dataset: str):
        data_sample_names = []
        # Enumerate all images and labels with deterministic ordering
        for video_dir in sorted(os.listdir(dir_dataset)):
            video_path = os.path.join(dir_dataset, video_dir)
            labels_dir = os.path.join(video_path, 'Labels')
            images_dir = os.path.join(video_path, 'Images')
            if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
                continue
            for label_name in sorted(os.listdir(labels_dir)):
                if not label_name.endswith('.txt'):
                    continue
                label_path = os.path.join(labels_dir, label_name)
                image_path = os.path.join(images_dir, label_name.replace('.txt', '.jpg'))
                data_sample_names.append((image_path, label_path))
        return data_sample_names
    
    def get_data_sample_names(self, data_type: str):
        assert data_type in ['train', 'val', 'test']
        if data_type == 'train':
            return self.train_data_sample_names
        elif data_type == 'val':
            return self.val_data_sample_names
        elif data_type == 'test':
            return self.test_data_sample_names
        else:
            raise ValueError(f"Invalid data type: {data_type}")