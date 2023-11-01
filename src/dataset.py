# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy import pad


class AIRSDataset(Dataset):
    """
    AIRS Dataset class.
    This class is a PyTorch Dataset for loading images and masks from a directory.
    """
    def __init__(self, image_directory, mask_directory=None, transform=None, inference=False):
        """
        Initialize the dataset.
        Args:
            image_directory (str): Directory where the images are stored.
            mask_directory (str, optional): Directory where the masks are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            inference (bool, optional): If True, masks are not loaded. Default is False.
        """
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.inference = inference
        self.images = os.listdir(image_directory)

    def __len__(self):
        """
        Calculate the total number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        Args:
            index (int): Index of the sample to get.
        Returns:
            tuple: A tuple containing an image and its corresponding mask if not in inference mode.
                   Otherwise, only the image is returned.
        """
        # Load the image file
        image_path = os.path.join(self.image_directory, self.images[index])
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Load masks if any
        mask = None
        if not self.inference:
            # Load the mask file
            mask_path = os.path.join(self.mask_directory, self.images[index])
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0

        # Apply transformations if any
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
            image = augmentations["image"]
            mask = augmentations.get("mask", None)

        return image, mask if mask is not None else torch.Tensor()

    
class AIRSTiledDataset(Dataset):
    """
    AIRS Tiled Dataset class.
    This class is a PyTorch Dataset for loading tiled images and masks from a directory.
    """
    def __init__(self, image_directory, mask_directory=None, transform=None, tile_size=512, image_size=10000, inference=False):
        """
        Initialize the dataset.
        Args:
            image_directory (str): Directory where the images are stored.
            mask_directory (str, optional): Directory where the masks are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            tile_size (int, optional): Size of the tiles to be used. Default is 512.
            image_size (int, optional): Size of the images to be used. Default is 10000.
            inference (bool, optional): If True, masks are not loaded. Default is False.
        """
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.images = os.listdir(image_directory)
        self.tile_size = tile_size
        self.image_size = image_size
        self.inference = inference

    def __len__(self):
        """
        Calculate the total number of tiles in the dataset.
        """
        return len(self.images) * self.tiles_per_image()

    def __getitem__(self, index):
        """
        Get a tile from the dataset.
        Args:
            index (int): Index of the tile to get.
        Returns:
            tuple: A tuple containing an image tile and its corresponding mask tile if not in inference mode.
                   Otherwise, only the image tile is returned.
        """
        # Calculate the index of the image and the tile within the image
        image_index = index // self.tiles_per_image()
        tile_index = index % self.tiles_per_image()
        
        # Calculate the row and column position of the tile within the image
        row, col = self.get_tile_position(tile_index)

        # Construct the paths to the image and mask files
        image_path = os.path.join(self.image_directory, self.images[image_index])
        
        mask_tile = None
        if not self.inference:
            mask_path = os.path.join(self.mask_directory, self.images[image_index])
        
            # Get the specified tile from the image and mask
            image_tile, mask_tile = self.get_tiles(image_path, mask_path, row, col)
            
            # Apply transformations if any
            if self.transform is not None:
                augmentations = self.transform(image=image_tile, mask=mask_tile)
                image_tile = augmentations["image"]
                mask_tile = augmentations.get("mask", None)

            return image_tile, mask_tile if mask_tile is not None else torch.Tensor()

    def tiles_per_image(self):
        """
        Calculate the total number of tiles in an image.
        """
        return ((self.image_size + self.tile_size - 1) // self.tile_size) ** 2

    def get_tile_position(self, tile_index):
        """
        Calculate the row and column position of a tile within an image.
        Args:
            tile_index (int): Index of the tile within an image.
        Returns:
            tuple: A tuple containing the row and column position of a tile within an image.
        """
        tiles_per_row = (self.image_size + self.tile_size - 1) // self.tile_size
        row = (tile_index // tiles_per_row) * self.tile_size
        col = (tile_index % tiles_per_row) * self.tile_size
        
        return row, col

    def get_tiles(self, image_path, mask_path, row, col):
        """
        Get a specific tile from an image and its corresponding mask.
        Args:
            image_path (str): Path to the image file.
            mask_path (str): Path to the mask file.
            row (int): Row position of the tile within an image.
            col (int): Column position of the tile within an image.
        Returns:
            tuple: A tuple containing an image tile and its corresponding mask tile.
        """
        # Load the image from file
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Pad the image if necessary
        padded_image = self.pad_image(image)
        
        # Get the specified tile from the padded image
        image_tile = padded_image[row:row+self.tile_size, col:col+self.tile_size]
        
        mask_tile = None
        if mask_path is not None:
            # Load the mask from file
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            
            # Pad the mask if necessary
            padded_mask = self.pad_image(mask)
            
            # Get the specified tile from the padded mask
            mask_tile = padded_mask[row:row+self.tile_size, col:col+self.tile_size]
            
            # Convert all pixels in the mask with a value of 255.0 to 1.0
            mask_tile[mask_tile == 255.0] = 1.0

        return image_tile, mask_tile

    def pad_image_and_mask(self, image, mask):
        """
        Pad an image and its corresponding mask to the tile size.
        """
        # Calculate the padding size for the height and width
        pad_height = (self.tile_size - image.shape[0] % self.tile_size) % self.tile_size
        pad_width = (self.tile_size - image.shape[1] % self.tile_size) % self.tile_size
        
        # Pad the image and mask
        padded_image = pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        padded_mask = pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
        
        return padded_image, padded_mask