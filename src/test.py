# test.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import warnings
warnings.filterwarnings("ignore")

# Importing necessary modules 
from model import FullyConnectedNetwork
from utils import (
    get_loader,
    load_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

# Importing the configuration parameters from config.py
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function to setup data, model, and initiate the testing process.
    """

    # Define transformations for the test dataset
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Initialize & Load a saved model 
    model = FullyConnectedNetwork().to(DEVICE)
    load_checkpoint(torch.load(MODEL_NAME), model)

    # Get DataLoader for the test dataset
    test_loader = get_loader(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=test_transforms,
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # check validation accuracy
    check_accuracy(test_loader, model, device=DEVICE)

    # Save predictions from the test dataset
    save_predictions_as_imgs(test_loader, model, folder=TEST_PREDICTION_PATH, device=DEVICE)


if __name__ == "__main__":
    main()