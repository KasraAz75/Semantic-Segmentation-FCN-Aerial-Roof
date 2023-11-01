# utils.py
import os
import torch
from torchvision  import transforms
from torch.utils.data import DataLoader
from dataset import AIRSDataset, AIRSTiledDataset
from metrics import DiceLoss, IoULoss
import logging


def get_loader(
        image_dir, 
        mask_dir=None,
        transform=None,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        inference=False
        ):
    """
    Get a data loader for a dataset.
    
    Args:
        image_dir (str): Directory where the images are stored.
        mask_dir (str): Directory where the masks are stored.
        transform (callable): Transformations to be applied on samples.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to copy tensors into CUDA pinned memory.
        shuffle (bool): Whether to shuffle the dataset after every epoch.
        inference (bool): Whether it is inference or not.

    Returns:
        DataLoader: A DataLoader for the given dataset.
    """
    
    ds = AIRSTiledDataset(
        image_directory=image_dir,
        mask_directory=mask_dir,
        transform=transform,
        tile_size=512, 
        image_size=10000,
        inference=inference
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    return loader


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save the current state of the model training.
    """
    
    logging.info("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """
    Load a checkpoint into a model.
    """
    
    logging.info("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    """
    Check the accuracy of a model on a data loader.
    
    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        model (torch.nn.Module): The model to check the accuracy of.
        device (str, optional): The device to run the model on. Default is "cuda".
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # Convert the batch of images and labels to the specified device
            x = x.to(device)
            y = y.to(device)
            
            # Get the model's predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Calculate the number of correct predictions & total number of pixels
            num_correct = (preds == y.unsqueeze(1)).sum().item()
            num_pixels = torch.numel(preds)

            # Calculate the Dice score
            dice_loss_fn = DiceLoss()
            dice_loss = dice_loss_fn(preds, y)
            dice_score = 1 - dice_loss

            # Calculate the IoU score
            iou_loss_fn = IoULoss()
            iou_loss = iou_loss_fn(preds, y)
            iou_score = 1 - iou_loss

    logging.info(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f} for validation data")
    logging.info(f"Dice score for validation data: {dice_score}")
    logging.info(f"IoU score for validation data: {iou_score}")
    
    model.train()


def save_predictions_as_imgs(loader, model, original_height=10000, original_width=10000, folder="predicted_images/", device="cuda"):
    """
    Save the model's predictions as images.
    
    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        model (torch.nn.Module): The model to get predictions from.
        original_height (int, optional): the original height of the input image
        original_width (int, optional): the original width of the input image
        folder (str, optional): The directory to save the images in. Default is "predicted_images/".
        device (str, optional): The device to run the model on. Default is "cuda".
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        # Convert the batch of images to the specified device
        x = x.to(device=device)
        
        with torch.no_grad():
            # Get the model's predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        for i in range(preds.shape[0]):
            # Save the prediction to a PIL Image and resize it
            pred_pil = transforms.ToPILImage()(preds[i]).resize((original_height, original_width))
            pred_pil.save(f"{folder}/pred_{idx}_{i}.png")
            
            # Save the ground truth to a PIL Image and resize it
            if y.numel() != 0:
                mask_pil = transforms.ToPILImage()(y[i]).resize((original_height, original_width))
                mask_pil.save(f"{folder}/{idx}_{i}.png")

    logging.info(f"prediction images were saved successfully at {folder}.")

    model.train()    