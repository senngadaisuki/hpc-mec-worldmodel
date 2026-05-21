# -*- coding: utf-8 -*-
import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import einops
from torch import stack
import torchvision.transforms.functional as F
import re
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class Adapted_Transform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, values):
        obs, act, mask = values
        t, v, c, h, w = obs.shape
        obs = obs.view(-1, c, h, w)
        transformed_images = stack([self.transform(F.to_pil_image(img)) for img in obs])
        _, c, h, w  = transformed_images.shape
        transformed_images = transformed_images.view(t, v, c, h, w)
        return transformed_images, act, mask


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)

class COIL100_DataLoader(Dataset):
    def __init__(self, img_dir, batch_size, seq_len=10, action_threshold=30, batch_cls=4, shuffle=False, transform=None):
        # Initialize parameters
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_cls = batch_cls
        self.shuffle = shuffle
        self.transform = transform

        # Collect image file information
        self.labels_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.indices = np.arange(1, 101)  # 100 classes
        self.rotation_list = np.arange(0, 360, 5)  # 72 rotations
        self.action_list = np.arange(-action_threshold, action_threshold + 1, 5)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        # Check end of dataset
        if self.current_index >= len(self.indices):
            raise StopIteration

        # Get current batch class indices
        start = self.current_index
        end = min(start + self.batch_cls, len(self.indices))
        self.current_index = end

        cls_indices = self.indices[start:end]
        cls_seq_num = self.batch_size // len(cls_indices)  # Sequences per class

        # Load image sequences
        X = []
        X_rotation = []
        action_seq_list = []
        for cls in cls_indices:
            # Generate rotation and action sequences
            initial_rotation = self.efficient_sample_with_min_distance(self.rotation_list, cls_seq_num, len(self.rotation_list) // cls_seq_num)
            rotation_seq_samples, action_seq_samples = [], []

            for i in range(cls_seq_num):
                action_seq = random.choices(self.action_list.tolist(), k=self.seq_len - 1)
                rotation_seq = np.cumsum(action_seq) + initial_rotation[i]
                rotation_seq = np.insert(rotation_seq, 0, initial_rotation[i]) % 360
                rotation_seq_samples.append(rotation_seq)
                action_seq_samples.append(action_seq)

            rotation_seq_samples = np.array(rotation_seq_samples)
            action_seq_samples = np.array(action_seq_samples)
            X_rotation.append(rotation_seq_samples)
            action_seq_list.append(action_seq_samples)
            for rotation_seq in rotation_seq_samples:
                image_seq = [self._load_image(cls, rotation) for rotation in rotation_seq]
                X.append(image_seq)

        # Create labels
        label = np.repeat(cls_indices[:, np.newaxis] - 1, cls_seq_num * self.seq_len).reshape(-1, self.seq_len)
        
        # Create rotation and action data
        X_rotation = np.stack(X_rotation).reshape(-1, self.seq_len)
        action_seq_list = np.stack(action_seq_list).reshape(-1, self.seq_len-1)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(X_rotation, dtype=torch.float32), \
            torch.tensor(action_seq_list, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(cls_seq_num, dtype=torch.long)

    def _load_image(self, cls, rotation):
        """Load and preprocess a single image."""
        img_path = os.path.join(self.img_dir, f'obj{cls}__{rotation}.png')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image).numpy()
        return image

    def __len__(self):
        return (len(self.indices) + self.batch_cls - 1) // self.batch_cls

    def get_sequence(self, seq_num):
        X = []
        np.random.shuffle(self.indices)
        cls_indices = self.indices[0:seq_num]
        for cls in cls_indices:
            image_seq = [self._load_image(cls, rotation) for rotation in self.rotation_list]
            X.append(image_seq)

        X_rotation = np.tile(self.rotation_list, (len(cls_indices), 1))
        action_seq_list = np.tile(np.ones(len(self.rotation_list)-1) * 5., (len(cls_indices), 1))
        label = np.zeros(len(self.rotation_list)) + cls

        return torch.tensor(np.array(X)), torch.tensor(X_rotation), torch.tensor(action_seq_list), torch.tensor(label)
    
    def efficient_sample_with_min_distance(self, array, num_points, min_distance):
        total_points = len(array)
        if num_points * min_distance > total_points:
            raise ValueError("Please choose a smaller num_points or larger min_distance.")
        
        start_idx = np.random.randint(0, total_points)
        
        sampled_indices = [(start_idx + i * min_distance) % total_points for i in range(num_points)]
        
        sampled_points = [array[idx] for idx in sampled_indices]
        return sampled_points
    

class MIRO_Dataloader(Dataset):
    def __init__(self, img_dir, batch_size, seq_len=10, action_threshold=30, batch_cls=4, shuffle=False, transform=None):
        # Initialize parameters
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_cls = batch_cls
        self.shuffle = shuffle
        self.transform = transform
        self.seq_len = 16

        self.cls_type = ["bus", "car", "cleanser", "clock", "cup", "headphones", "mouse", "scissors", "shoe", "stapler", "sunglasses", "tape_cutter"]

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, root_dir, image_files, idx):
        """Load and preprocess a single image."""
        img_path = os.path.join(root_dir, image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).numpy()
        return image

    def get_sequence(self):
        X = []

        for cls in self.cls_type:
            img_path = os.path.join(self.img_dir, f'{cls}')
            image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.png')])
            image_seq = [self._load_image(img_path, image_files, i) for i in range(self.seq_len)]
            X.append(image_seq)

        X_rotation = np.tile(np.arange(0, 360, 22.5), (len(self.cls_type), 1))
        action_seq_list = np.tile(np.ones(self.seq_len-1) * 22.5, (len(self.cls_type), 1))

        return torch.tensor(np.array(X)), torch.tensor(X_rotation), torch.tensor(action_seq_list)

class SomethingSomethingV2Dataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, frames_per_clip=8, step_between_clips=None, sample_downsample_rate=10, sliding_window=4):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotations_file (str): Path to the annotations file (JSON format).
            transform (callable, optional): Optional transform to be applied on a sample.
            frames_per_clip (int): Number of frames per clip.
            step_between_clips (int): Step in frames between each clip. If None, it will be set to frames_per_clip to minimize overlap.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips if step_between_clips is not None else frames_per_clip
        self.sliding_window = sliding_window

        with open(annotations_file, 'r') as f:
            samples = [sample for sample in json.load(f) if len(os.listdir(os.path.join(root_dir, sample['id']))) >= frames_per_clip]

        # Downsample the number of samples
        self.samples = samples[::sample_downsample_rate]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['id']

        # Get list of all frame files in the video directory
        video_dir = os.path.join(self.root_dir, video_id)
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        num_frames = len(frame_files)

        clip = []
        for frame_idx in np.linspace(0, num_frames, self.frames_per_clip, dtype=int, endpoint=False):
            frame_path = os.path.join(video_dir, frame_files[frame_idx])
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip.append(image)

        clip = torch.stack(clip, dim=0)  # [4, C, H, W]
        # Create sliding window clips
        if self.sliding_window > 1:
            # Ensure the clip is long enough for the sliding window
            if len(clip) < self.sliding_window:
                raise ValueError(f"Clip length {len(clip)} is less than sliding window size {self.sliding_window}.")
            
            # Create sliding window clips
            # clips = [clip[i:i+self.sliding_window] for i in range(self.frames_per_clip - self.sliding_window + 1)]
            if self.frames_per_clip == self.sliding_window:
                clips = [clip]
            else: 
                # clips = [clip[0:self.sliding_window], clip[self.frames_per_clip-self.sliding_window:self.frames_per_clip]]
                # clips = [clip[i:i+self.sliding_window] for i in range(self.frames_per_clip - self.sliding_window + 1)]
                clips = [clip[4:4+self.sliding_window]]

        return torch.stack(clips, dim=0) # [5, 4, C, H, W], video_id

class MultiActionOmniDataset(Dataset):
    def __init__(self, img_dir, seq_len=20, 
                 rotation_actions=np.arange(-30, 30 + 1, 5),
                 plane_rotation_actions=np.arange(-30, 30 + 1, 5),
                 scale_actions=np.arange(0.5, 2 + 0.1, 0.1),
                 translation_x_actions=np.arange(-20, 20 + 1, 5),
                 translation_y_actions=np.arange(-20, 20 + 1, 5),
                 transform=None, continuous_rotation=False,
                 initial_rotation_range=np.arange(0, 360, 5),
                 initial_plane_rotation_range=np.arange(-20, 20 + 1, 5),
                 initial_scale_range=np.arange(0.5, 1.0 + 0.01, 0.1),
                 initial_translation_x_range=np.arange(-30, 30 + 1, 10),
                 initial_translation_y_range=np.arange(-30, 30 + 1, 10),
                 action_types=['rotation', 'plane_rotation', 'scale', 'translation_x', 'translation_y'],
                 single_action_per_step=False,
                 background_path=None,
                 use_mask=True,
                 image_size=(128, 128),
                 allow_out_of_bounds=False):
        """
        Multi-action dataset with improved initialization and boundary constraints.
        
        Args:
            img_dir (str): Directory containing object images.
            seq_len (int): Length of the sequence.
            rotation_actions (np.ndarray): Actions for rotation.
            scale_actions (np.ndarray): Actions for scaling.
            translation_x_actions (np.ndarray): Actions for x translation.
            translation_y_actions (np.ndarray): Actions for y translation.
            plane_rotation_actions (np.ndarray): Actions for plane rotation.
            transform (callable, optional): Transform to apply to images.
            continuous_rotation (bool): Whether to allow continuous rotation.
            initial_rotation_range (np.ndarray): Range of initial rotations.
            initial_scale_range (np.ndarray): Range of initial scales.
            initial_translation_x_range (np.ndarray): Range of initial x translations.
            initial_translation_y_range (np.ndarray): Range of initial y translations.
            initial_plane_rotation_range (np.ndarray): Range of initial plane rotations.
            allow_out_of_bounds (bool): If True, allows objects to go out of image boundaries.If False (default), constrains actions to keep objects within bounds.
            action_types (list): Types of actions to include in the dataset.
            background_path (str, optional): Path to background image(s).
            use_mask (bool): Whether to use masks with images.
            image_size (tuple): Size of the output images.

            # Define some extra transforms
            additional_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Note: do NOT add ToTensor() here; it is handled separately
            ])
        """
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.continuous_rotation = continuous_rotation
        self.action_types = action_types
        self.use_mask = use_mask
        self.background_path = background_path
        self.image_size = image_size
        self.allow_out_of_bounds = allow_out_of_bounds

        # Load background image(s)
        self.background = self._load_background()

        # Collect image file information
        self.all_object_dir = self.count_lowest_level_directories(self.img_dir)
        self.indices = np.arange(len(self.all_object_dir))
        
        # Validate the initial scale range
        if np.min(initial_scale_range) < 0.3:
            print(f"Warning: initial_scale_range minimum {np.min(initial_scale_range)} < 0.3")

        # Initial ranges
        self.initial_rotation_range = initial_rotation_range
        self.initial_plane_rotation_range = initial_plane_rotation_range
        self.initial_scale_range = initial_scale_range
        self.initial_translation_x_range = initial_translation_x_range
        self.initial_translation_y_range = initial_translation_y_range
        
        # Action ranges
        self.rotation_actions = np.array([5]) if self.continuous_rotation else rotation_actions
        self.plane_rotation_actions = plane_rotation_actions
        self.scale_actions = scale_actions
        self.translation_x_actions = translation_x_actions
        self.translation_y_actions = translation_y_actions
        
        # The original image corresponds to the maximum scale
        self.max_scale = 1.0

        # Approximate object size within the image
        self.estimated_object_size = min(self.image_size) * 0.8
        
        # Record category
        category_name_lst = []
        for path in self.all_object_dir:
            category_name, object_name = self.extract_category_from_path(path)
            category_name_lst.append(category_name)
        category_name_lst = set(category_name_lst)
        self.category_index_dict = {category: idx for idx, category in enumerate(sorted(category_name_lst))}
        self.category_num = len(category_name_lst)
        
        self.single_action_per_step = single_action_per_step

    def _load_background(self):
        """Load background image(s)."""
        if self.background_path is None:
            # Use gray background
            return None
        elif os.path.isfile(self.background_path):
            # Single background image
            bg = Image.open(self.background_path).convert('RGB')
            return bg
        elif os.path.isdir(self.background_path):
            # Multiple background images (commented out for future use)
            # bg_files = [f for f in os.listdir(self.background_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            # backgrounds = [Image.open(os.path.join(self.background_path, f)).convert('RGB') for f in bg_files]
            # return backgrounds
            
            # For now, use the first background image
            bg_files = [f for f in os.listdir(self.background_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if bg_files:
                bg = Image.open(os.path.join(self.background_path, bg_files[0])).convert('RGB')
                return bg
            else:
                return None
        else:
            return None

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, key):
        # Slice access: dataset[:] returns a batch accessor
        if isinstance(key, slice):
            return BatchAccessor(self, key)

        # List/array access: dataset[[0,1,2]] returns a batch accessor
        elif isinstance(key, (list, tuple, np.ndarray)):
            return BatchAccessor(self, key)

        # Single-index access: dataset[0]
        else:
            return self._get_single_item(key)

    def _get_single_item(self, idx):
        # Randomly initialize rotation and plane_rotation
        initial_rotation = random.choice(self.initial_rotation_range)
        initial_plane_rotation = random.choice(self.initial_plane_rotation_range)

        initial_scale = random.choice(self.initial_scale_range)
        # Choose the initial-translation sampling strategy based on allow_out_of_bounds

        if self.allow_out_of_bounds:
            # Out-of-bounds allowed: sample from the full range
            initial_translation_x = random.choice(self.initial_translation_x_range)
            initial_translation_y = random.choice(self.initial_translation_y_range)
        else:
            # Compute the valid translation range from the scale
            scaled_object_size = self.estimated_object_size * initial_scale
            max_tx = (self.image_size[0] - scaled_object_size) / 2
            max_ty = (self.image_size[1] - scaled_object_size) / 2

            valid_tx_range = [tx for tx in self.initial_translation_x_range if -max_tx <= tx <= max_tx]
            valid_ty_range = [ty for ty in self.initial_translation_y_range if -max_ty <= ty <= max_ty]

            # Fall back to 0 if no valid value exists
            if not valid_tx_range:
                valid_tx_range = [0]
            if not valid_ty_range:
                valid_ty_range = [0]

            initial_translation_x = random.choice(valid_tx_range)
            initial_translation_y = random.choice(valid_ty_range)

            # Ensure the initial position stays within bounds (accounting for plane rotation)
            initial_translation_x, initial_translation_y = self._clamp_translation(
                initial_translation_x, initial_translation_y, initial_scale, initial_plane_rotation
            )

        # Pass the initial state (incl. plane rotation) to the action generator
        action_seq = self._generate_action_sequence(
            initial_plane_rotation, initial_scale, initial_translation_x, initial_translation_y
        )

        # Compute the transform sequence (incl. plane rotation)
        transform_seq, corrected_action_seq = self._calculate_transform_sequence_with_bounds(
            action_seq, initial_rotation, initial_plane_rotation, initial_scale,
            initial_translation_x, initial_translation_y
        )

        # Use the corrected action sequence
        action_seq = corrected_action_seq
        
        # Load and transform images
        image_seq = []
        for i, transform_params in enumerate(transform_seq):
            image = self.load_and_transform_image(idx, transform_params)
            image_seq.append(image)
        
        # Get category information
        category_name, object_name = self.extract_category_from_path(self.all_object_dir[idx])
        category = [self.category_index_dict[category_name]] * self.seq_len

        item = {
            'image_seq': torch.stack(image_seq).float(),
            'action_seq': torch.tensor(action_seq).float(),
            'transform_seq': torch.tensor(transform_seq).float(),
            'seq_index': torch.arange(0, self.seq_len, step=1).float(),
            'category_name': category_name,
            'object_name': object_name,
            'category': torch.tensor(category).long(),
        }
        return item

    def _clamp_translation(self, tx, ty, scale, plane_rotation=0):
        """
        Clamp translation given the current scale and plane_rotation so the
        object stays within the image bounds, accounting for the object's
        actual footprint after plane rotation.
        """
        scaled_object_size = self.estimated_object_size * scale

        # Account for plane rotation: the object's bounding box grows. At angle
        # theta, the bounding-box side becomes size * (|cos(theta)| + |sin(theta)|).
        angle_rad = np.radians(abs(plane_rotation) % 90)  # only the 0-90 deg range matters
        rotation_factor = abs(np.cos(angle_rad)) + abs(np.sin(angle_rad))

        # Effective footprint after rotation
        effective_size = scaled_object_size * rotation_factor

        max_tx = (self.image_size[0] - effective_size) / 2
        max_ty = (self.image_size[1] - effective_size) / 2

        tx = np.clip(tx, -max_tx, max_tx)
        ty = np.clip(ty, -max_ty, max_ty)
        
        return tx, ty

    def _get_valid_translation_actions(self, current_tx, current_ty, current_scale, current_plane_rotation):
        """
        Return the valid translation actions for the current position, scale
        and plane_rotation.
        """
        if self.allow_out_of_bounds:
            return list(self.translation_x_actions), list(self.translation_y_actions)
        valid_tx_actions = []
        valid_ty_actions = []
        
        for tx_action in self.translation_x_actions:
            new_tx = current_tx + tx_action
            new_tx_clamped, _ = self._clamp_translation(new_tx, current_ty, current_scale, current_plane_rotation)
            if abs(new_tx_clamped - new_tx) < 1e-6:
                valid_tx_actions.append(tx_action)
        
        for ty_action in self.translation_y_actions:
            new_ty = current_ty + ty_action
            _, new_ty_clamped = self._clamp_translation(current_tx, new_ty, current_scale, current_plane_rotation)
            if abs(new_ty_clamped - new_ty) < 1e-6:
                valid_ty_actions.append(ty_action)
        
        # Keep 0 (no-op) if there are no valid actions
        if not valid_tx_actions:
            print(f"WARNING: No valid translation X actions, falling back to 0 (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")

            valid_tx_actions = [0]
        if not valid_ty_actions:
            print(f"WARNING: No valid translation Y actions, falling back to 0 (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")
            valid_ty_actions = [0]
            
        return valid_tx_actions, valid_ty_actions

    def _get_valid_plane_rotation_actions(self, current_plane_rotation, current_scale, current_tx, current_ty):
        """
        Return the valid plane_rotation actions for the current state.
        """
        if self.allow_out_of_bounds:
            return list(self.plane_rotation_actions)
        valid_actions = []
        
        for pr_action in self.plane_rotation_actions:
            new_pr = (current_plane_rotation + pr_action) % 360
            
            # Check (via _clamp_translation) whether the new plane_rotation
            # would push the object out of bounds
            test_tx, test_ty = self._clamp_translation(current_tx, current_ty, current_scale, new_pr)

            # If translation was not clamped, this plane_rotation is valid
            if abs(test_tx - current_tx) < 1e-6 and abs(test_ty - current_ty) < 1e-6:
                valid_actions.append(pr_action)
        
        # Keep 0 (no rotation) if there are no valid actions
        if not valid_actions:
            print(f"WARNING: No valid plane rotation actions, falling back to 0 (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")
            valid_actions = [0]
        
        return valid_actions

    def _generate_action_sequence(self, initial_plane_rotation, initial_scale, initial_tx, initial_ty):
        """
        Generate the action sequence; supports single-action-per-step mode.
        """
        action_seq = []
        action_dim = len(self.action_types)

        current_plane_rotation = initial_plane_rotation
        current_scale = initial_scale
        current_tx = initial_tx
        current_ty = initial_ty

        for step in range(self.seq_len - 1):
            action = np.zeros(action_dim)

            # Predicted scale / plane_rotation for this step, used to check
            # translation validity before translation actions are sampled
            predicted_scale = current_scale
            predicted_plane_rotation = current_plane_rotation

            if self.single_action_per_step:
                action = np.ones(action_dim)
                for i, action_type in enumerate(self.action_types):
                    if action_type != 'scale':
                        action[i] = 0.0

                # Only one action type per step
                active_action_idx = random.choice(range(len(self.action_types)))
                active_action_type = self.action_types[active_action_idx]

                # Stage 1: scale / plane_rotation actions (these change the bounds)
                if active_action_type == 'rotation':
                    action[active_action_idx] = random.choice(self.rotation_actions)
                elif active_action_type == 'plane_rotation':
                    valid_pr_actions = self._get_valid_plane_rotation_actions(
                        current_plane_rotation, current_scale, current_tx, current_ty
                    )
                    action[active_action_idx] = random.choice(valid_pr_actions)
                    predicted_plane_rotation = (current_plane_rotation + action[active_action_idx]) % 360
                elif active_action_type == 'scale':
                    valid_scale_action = self._sample_valid_scale_action(current_scale, current_tx, current_ty, current_plane_rotation)
                    action[active_action_idx] = valid_scale_action
                    predicted_scale = current_scale * valid_scale_action

                # Stage 2: translation actions based on predicted scale / plane_rotation
                elif active_action_type == 'translation_x':
                    valid_tx_actions, _ = self._get_valid_translation_actions(
                        current_tx, current_ty, predicted_scale, predicted_plane_rotation
                    )
                    action[active_action_idx] = random.choice(valid_tx_actions)
                elif active_action_type == 'translation_y':
                    _, valid_ty_actions = self._get_valid_translation_actions(
                        current_tx, current_ty, predicted_scale, predicted_plane_rotation
                    )
                    action[active_action_idx] = random.choice(valid_ty_actions)

            else:
                # Multi-action mode: scale / plane_rotation first, then translation
                for i, action_type in enumerate(self.action_types):
                    if action_type == 'rotation':
                        action[i] = random.choice(self.rotation_actions)
                    elif action_type == 'plane_rotation':
                        valid_pr_actions = self._get_valid_plane_rotation_actions(
                            current_plane_rotation, current_scale, current_tx, current_ty
                        )
                        action[i] = random.choice(valid_pr_actions)
                        predicted_plane_rotation = (current_plane_rotation + action[i]) % 360
                    elif action_type == 'scale':
                        valid_scale_action = self._sample_valid_scale_action(current_scale, current_tx, current_ty, current_plane_rotation)
                        action[i] = valid_scale_action
                        predicted_scale = current_scale * valid_scale_action
                
                # Translation based on predicted scale / plane_rotation
                for i, action_type in enumerate(self.action_types):
                    if action_type == 'translation_x':
                        valid_tx_actions, _ = self._get_valid_translation_actions(
                            current_tx, current_ty, predicted_scale, predicted_plane_rotation
                        )
                        action[i] = random.choice(valid_tx_actions)
                    elif action_type == 'translation_y':
                        _, valid_ty_actions = self._get_valid_translation_actions(
                            current_tx, current_ty, predicted_scale, predicted_plane_rotation
                        )
                        action[i] = random.choice(valid_ty_actions)
            
            action_seq.append(action)
            
            # Update the current state using the actual action values
            for i, action_type in enumerate(self.action_types):
                if action_type == 'plane_rotation':
                    current_plane_rotation = (current_plane_rotation + action[i]) % 360
                elif action_type == 'scale':
                    if action[i] != 0:
                        new_scale = current_scale * action[i]
                        if not (0.3 - 1e-6 <= new_scale <= self.max_scale + 1e-6):
                            print(f"ERROR in action generation: step {step}, current_scale {current_scale}, action {action[i]}, new_scale {new_scale}")
                        current_scale = new_scale
                elif action_type == 'translation_x':
                    current_tx += action[i]
                elif action_type == 'translation_y':
                    current_ty += action[i]
            
            if not self.allow_out_of_bounds:
                current_tx, current_ty = self._clamp_translation(current_tx, current_ty, current_scale, current_plane_rotation)
        
        return np.array(action_seq)
    
    def _sample_valid_scale_action(self, current_scale, current_tx, current_ty, current_plane_rotation):
        """
        Sample a valid scale action, ensuring the object stays within bounds
        after it is applied.
        """
        min_scale = 0.3
        max_scale = self.max_scale
        
        valid_actions = []
        for action in self.scale_actions:
            new_scale = current_scale * action
            
            # Skip if the new scale is out of the valid range
            if not (min_scale - 1e-6 <= new_scale <= max_scale + 1e-6):
                continue

            # If out-of-bounds is allowed, only the scale range needs checking
            if self.allow_out_of_bounds:
                valid_actions.append(action)
            else:
                # Otherwise also check it does not push the object out of bounds
                test_tx, test_ty = self._clamp_translation(current_tx, current_ty, new_scale, current_plane_rotation)

                # Valid if the position was not clamped
                if abs(test_tx - current_tx) < 1e-6 and abs(test_ty - current_ty) < 1e-6:
                    valid_actions.append(action)
         
        if not valid_actions:
            print(f"WARNING: No valid scale actions (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")
            # Return 1.0 (no change) as a safe fallback
            return 1.0
        
        return random.choice(valid_actions)

    def _calculate_transform_sequence_with_bounds(self, action_seq, initial_rotation, initial_plane_rotation, initial_scale, initial_tx, initial_ty):
        """Compute the transform sequence (incl. plane rotation), correcting clamped actions."""
        transform_seq = []
        corrected_action_seq = []

        # Current state (5 params)
        current_rotation = initial_rotation
        current_plane_rotation = initial_plane_rotation
        current_scale = initial_scale
        current_tx = initial_tx
        current_ty = initial_ty
        
        # Add initial state (5-dim)
        transform_seq.append([current_rotation, current_plane_rotation, current_scale, current_tx, current_ty])
        
        # Apply actions cumulatively
        for step, action in enumerate(action_seq):
            corrected_action = action.copy()  # copy so it can be corrected

            # Save the position before applying actions (to compute the real change)
            old_tx = current_tx
            old_ty = current_ty
            
            for i, action_type in enumerate(self.action_types):
                if action_type == 'rotation':
                    current_rotation = (current_rotation + action[i]) % 360
                elif action_type == 'plane_rotation':
                    current_plane_rotation = (current_plane_rotation + action[i]) % 360
                elif action_type == 'scale':
                    old_scale = current_scale
                    current_scale *= action[i]
                    
                    if current_scale < 0.3 - 1e-6 or current_scale > self.max_scale + 1e-6:
                        print(f"ERROR: Invalid scale at step {step}")
                        print(f"  old_scale: {old_scale}")
                        print(f"  action: {action[i]}")
                        print(f"  new_scale: {current_scale}")
                        raise ValueError(f"Scale {current_scale} out of bounds!")
                        
                elif action_type == 'translation_x':
                    current_tx += action[i]
                elif action_type == 'translation_y':
                    current_ty += action[i]
            
            # Apply the boundary constraint (accounting for the current plane_rotation)
            old_tx_before_clamp = current_tx
            old_ty_before_clamp = current_ty
            if not self.allow_out_of_bounds:
                current_tx, current_ty = self._clamp_translation(current_tx, current_ty, current_scale, current_plane_rotation)

            # Check whether plane_rotation forced a translation adjustment
            pr_idx = self.action_types.index('plane_rotation') if 'plane_rotation' in self.action_types else -1
            if pr_idx >= 0 and (abs(current_tx - old_tx_before_clamp) > 1e-6 or abs(current_ty - old_ty_before_clamp) > 1e-6):
                # plane_rotation pushed the position out of bounds
                if abs(action[pr_idx]) > 1e-6:  # only warn when there is an actual rotation
                    print(f"INFO: Plane rotation {action[pr_idx]:.1f}° caused position adjustment at step {step}")
            
            # If clamped, correct the corresponding action
            tx_idx = self.action_types.index('translation_x') if 'translation_x' in self.action_types else -1
            ty_idx = self.action_types.index('translation_y') if 'translation_y' in self.action_types else -1
            
            if tx_idx >= 0 and abs(current_tx - old_tx_before_clamp) > 1e-6:
                # translation_x was clamped; compute the real change (using old_tx saved before the loop)
                actual_tx_change = current_tx - old_tx
                corrected_action[tx_idx] = actual_tx_change
                print(f"WARNING: Translation X clamped at step {step}: action {action[tx_idx]:.1f} -> {actual_tx_change:.1f} current state: (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")
                
            if ty_idx >= 0 and abs(current_ty - old_ty_before_clamp) > 1e-6:
                # translation_y was clamped; compute the real change (using old_ty saved before the loop)
                actual_ty_change = current_ty - old_ty
                corrected_action[ty_idx] = actual_ty_change
                print(f"WARNING: Translation Y clamped at step {step}: action {action[ty_idx]:.1f} -> {actual_ty_change:.1f} current state: (scale={current_scale:.2f}, plane_rot={current_plane_rotation:.1f}°, tx={current_tx:.1f}, ty={current_ty:.1f})")
            
            corrected_action_seq.append(corrected_action)
            transform_seq.append([current_rotation, current_plane_rotation, current_scale, current_tx, current_ty])
        
        return np.array(transform_seq), np.array(corrected_action_seq)


    def load_and_transform_image(self, idx, transform_params):
        """Load image with mask and apply transformations: plane rotation -> scale -> translation."""
        rotation, plane_rotation, scale, tx, ty = transform_params
        
        # Load base image and mask at the specified rotation
        img_path = self.all_object_dir[idx] + f'/{int(rotation)//5:03d}.png'
        mask_path = self.all_object_dir[idx] + f'/{int(rotation)//5:03d}_mask.npy'
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        if self.use_mask and os.path.exists(mask_path):
            # Load mask
            mask = np.load(mask_path)
            mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            
            # 1. Apply plane rotation FIRST
            if plane_rotation != 0:
                image = image.rotate(plane_rotation, expand=False, fillcolor=(0, 0, 0, 0))
                mask = mask.rotate(plane_rotation, expand=False, fillcolor=0)
            
            # 2. Apply scaling SECOND
            if scale != 1.0:
                new_height = int(original_size[1] * scale)
                new_width = int(original_size[0] * scale)
                
                # Resize both image and mask
                image = image.resize((new_width, new_height), Image.LANCZOS)
                mask = mask.resize((new_width, new_height), Image.LANCZOS)
                
                # Create new canvas with original size
                scaled_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
                scaled_mask = Image.new('L', original_size, 0)
                
                # Calculate position to center the scaled object
                paste_x = (original_size[0] - new_width) // 2
                paste_y = (original_size[1] - new_height) // 2
                
                # Paste scaled image and mask at center
                scaled_image.paste(image, (paste_x, paste_y))
                scaled_mask.paste(mask, (paste_x, paste_y))
                
                image = scaled_image.convert('RGB')
                mask = scaled_mask
            
            # 3. Apply translation LAST (in global screen coordinates)
            if tx != 0 or ty != 0:
                # Create new canvas for translation
                translated_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
                translated_mask = Image.new('L', original_size, 0)
                
                # Calculate translation bounds
                src_x = max(0, -int(tx))
                src_y = max(0, -int(ty))
                dst_x = max(0, int(tx))
                dst_y = max(0, int(ty))
                
                width = original_size[0] - abs(int(tx))
                height = original_size[1] - abs(int(ty))
                
                if width > 0 and height > 0:
                    # Crop and paste the translated part
                    cropped_image = image.crop((src_x, src_y, src_x + width, src_y + height))
                    cropped_mask = mask.crop((src_x, src_y, src_x + width, src_y + height))
                    
                    translated_image.paste(cropped_image, (dst_x, dst_y))
                    translated_mask.paste(cropped_mask, (dst_x, dst_y))
                
                image = translated_image.convert('RGB')
                mask = translated_mask
            
            # Composite with background
            if self.background is not None:
                background = self.background.resize(original_size)
            else:
                background = Image.new('RGB', original_size, (128, 128, 128))
            
            final_image = Image.composite(image, background, mask)
            
        else:
            # If not using mask, apply transforms directly
            final_image = image
            
            # Apply same order: plane rotation -> scale -> translation
            if plane_rotation != 0:
                final_image = final_image.rotate(plane_rotation, expand=False, fillcolor=(128, 128, 128))
        
        # Apply user-defined transforms if provided (on PIL image first)
        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                # Separate PIL transforms from tensor transforms
                pil_transforms = []
                tensor_transforms = []
                to_tensor_found = False
                
                for t in self.transform.transforms:
                    if isinstance(t, transforms.ToTensor):
                        to_tensor_found = True
                    elif not to_tensor_found:
                        # transforms before ToTensor (applied to the PIL image)
                        pil_transforms.append(t)
                    else:
                        # transforms after ToTensor (applied to the tensor)
                        tensor_transforms.append(t)

                for t in pil_transforms:
                    final_image = t(final_image)

                image_tensor = transforms.ToTensor()(final_image)

                for t in tensor_transforms:
                    image_tensor = t(image_tensor)

            else:
                # Single-transform case
                if isinstance(self.transform, transforms.ToTensor):
                    image_tensor = self.transform(final_image)
                else:
                    # Assume a PIL transform: apply it, then convert to tensor
                    final_image = self.transform(final_image)
                    image_tensor = transforms.ToTensor()(final_image)
        else:
            # No extra transforms; convert directly to tensor
            image_tensor = transforms.ToTensor()(final_image)
        
        return image_tensor
    def count_lowest_level_directories(self, path):
        """Same as OmniDataset."""
        lowest_level_dirs = []
        for root, dirs, files in os.walk(path):
            if not dirs:
                lowest_level_dirs.append(root)
        return lowest_level_dirs

    def extract_category_from_path(self, path):
        """Same as OmniDataset."""
        parent_dir = os.path.basename(os.path.dirname(path))
        match = re.match(r"^(\D+)", parent_dir)
        if match:
            return match.group(1).rstrip('_'), parent_dir 
        return "", parent_dir

class BatchAccessor:
    """Batch accessor: supports dataset[:]['key'] syntax, mimicking the
    stacking behavior of a DataLoader."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, key):
        """Support dataset[:]['key'] syntax; tensors are stacked automatically."""
        return self._get_key(key)

    def _get_key(self, key):
        """Return the value for a given key, stacking tensors automatically."""
        # Handle the different index types
        if isinstance(self.indices, slice):
            indices = range(*self.indices.indices(len(self.dataset)))
        else:
            indices = self.indices
        
        result = []
        for i in indices:
            item = self.dataset._get_single_item(i)
            if key in item:
                result.append(item[key])
            else:
                raise KeyError(f"Key '{key}' not found in dataset item")
        
        # Stack tensors automatically, mimicking DataLoader behavior
        if result and isinstance(result[0], torch.Tensor):
            return torch.stack(result)
        else:
            # For non-tensor data (e.g. category_name, object_name) return a list
            return result

    def __iter__(self):
        """Support iteration, yielding the full item dict."""
        if isinstance(self.indices, slice):
            indices = range(*self.indices.indices(len(self.dataset)))
        else:
            indices = self.indices
            
        for i in indices:
            yield self.dataset._get_single_item(i)
    
    def __len__(self):
        """Return the batch length."""
        if isinstance(self.indices, slice):
            return len(range(*self.indices.indices(len(self.dataset))))
        else:
            return len(self.indices)
    
    def __getattr__(self, name):
        """Support dataset[:].image_seq syntax."""
        valid_keys = ['image_seq', 'action_seq', 'transform_seq', 'seq_index',
                     'category_name', 'object_name', 'category']
        
        if name in valid_keys:
            return self._get_key(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def keys(self):
        """Return the available keys, mimicking a batch dict."""
        return ['image_seq', 'action_seq', 'transform_seq', 'seq_index',
                'category_name', 'object_name', 'category']
    
    def items(self):
        """Return all key-value pairs, mimicking a batch dict."""
        return [(key, self._get_key(key)) for key in self.keys()]

    def values(self):
        """Return all values, mimicking a batch dict."""
        return [self._get_key(key) for key in self.keys()]
    

def visualize_sequence_with_actions(images, actions, seq_idx=0, action_types=['rotation', 'scale', 'translation_x', 'translation_y', 'plane_rotation']):
    """
    Visualize a sequence; each frame's title shows the corresponding action.

    Args:
        images: torch.Tensor, shape (batch_size, seq_len, 3, H, W)
        actions: torch.Tensor, shape (batch_size, seq_len-1, action_dim)
        seq_idx: int, index of the sequence to visualize
        action_types: list, action type names
    """

    seq_images = images  # (seq_len, 3, H, W)
    seq_actions = actions  # (seq_len-1, action_dim)
    
    seq_len = seq_images.shape[0]
    
    cols = min(seq_len, 10)  # at most 10 columns
    rows = (seq_len + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))

    # Ensure axes is a 2D array even with a single row/column
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(seq_len):
        row = i // cols
        col = i % cols
        
        # Convert image format for display
        img = seq_images[i].permute(1, 2, 0)  # (H, W, 3)

        # If values are in [-1, 1], convert to [0, 1]; if already in [0, 1], display as is
        if img.min() >= -1 and img.max() <= 1:
            img = (img + 1) / 2  # from [-1, 1] to [0, 1]

        img = torch.clamp(img, 0, 1)

        axes[row, col].imshow(img.cpu().numpy())
        axes[row, col].axis('off')

        if i == 0:
            # First frame shows the initial state
            title = f"Frame {i}\nInitial State"
        else:
            # Other frames show the corresponding action
            action = seq_actions[i-1]  # actions[i-1] is the action from frame i-1 to frame i

            action_str = []
            for j, action_type in enumerate(action_types):
                if j < len(action):
                    if action_type == 'rotation':
                        action_str.append(f"rt: {action[j]:.1f}°")
                    elif action_type == 'scale':
                        action_str.append(f"sc: {action[j]:.2f}")
                    elif action_type == 'translation_x':
                        action_str.append(f"tx: {action[j]:.1f}")
                    elif action_type == 'translation_y':
                        action_str.append(f"ty: {action[j]:.1f}")
                    elif action_type == 'plane_rotation':
                        action_str.append(f"pr: {action[j]:.1f}°")
            
            title = f"Frame {i}\n" + ", ".join(action_str)
        
        axes[row, col].set_title(title, fontsize=8, pad=5)
    
    # Hide unused subplots
    for i in range(seq_len, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show() 
