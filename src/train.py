# train.py
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")

# Importing necessary modules 
from model import FullyConnectedNetwork
from utils import (
    get_loader,
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs
    )
from metrics import DiceLoss, IoULoss

# Importing the configuration parameters from config.py
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Function for training the model.

    Parameters:
        loader (DataLoader): DataLoader for iteration over the training data
        model (nn.Module): The model to be trained
        optimizer (torch.optim): Optimizer for the model parameters
        loss_fn (nn.Module): Loss function to measure the training error
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training
    """
    
    loop = tqdm(loader)

    # Initialize running loss to zero.
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass with autocasting for mixed precision training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward pass and optimization step with gradient scaling for mixed precision training
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop with current loss value
        loop.set_postfix(loss=loss.item())

        # Add current batch loss to running loss
        running_loss += loss.item()

    # Calculate average loss over all batches in an epoch
    avg_loss = running_loss / len(loader)
    
    return avg_loss


def main():
    """
    Main function to setup data, model, loss function and optimizer. 
    Also initiates the training process which includes training over a specified number of epochs,
    saving the model and optimizer state after each epoch, checking validation accuracy,
    and saving val predictions from the validation dataset after each epoch of training.
    """

    # Define augmentations for the training and validation datasets
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomResizedCrop(IMAGE_HEIGHT, IMAGE_WIDTH, p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
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

    # Initialize the model and move it to the device (GPU/CPU)
    model = FullyConnectedNetwork().to(DEVICE)

    # Define the loss function and optimizer
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Get DataLoaders for the training and validation datasets
    train_loader = get_loader(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform,
                              batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    
    val_loader = get_loader(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transforms,
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Load a saved model if LOAD_MODEL is set to True
    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_NAME), model)

    # Training loop over NUM_EPOCHS epochs 
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):

        # Log each batch and starting training
        logging.info(f"Starting training epoch {epoch+1}/{NUM_EPOCHS}")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Log the average loss at the end of each epoch
        logging.info(f"Average Dice loss in epoch {epoch+1}/{NUM_EPOCHS}: {avg_loss}")

        # Save the current state of the model and optimizer after each epoch of training
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check validation accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # Save predictions from the validation dataset after each epoch of training
        save_predictions_as_imgs(val_loader, model, 
                                folder=VAL_PREDICTION_PATH, device=DEVICE)


if __name__ == "__main__":
    main()