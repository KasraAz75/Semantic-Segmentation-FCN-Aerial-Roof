# config.py
import torch 

# Hyperparameters etc. for training
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_NAME = "checkpoint.pth.tar"
TRAIN_IMG_DIR = "train/image/"
TRAIN_MASK_DIR = "train/label/"
VAL_IMG_DIR = "val/image/"
VAL_MASK_DIR = "val/label/"
VAL_PREDICTION_PATH = "val_predicted_images/"

# Testing 
TEST_IMG_DIR = "test/image/"
TEST_MASK_DIR = "test/label/"
TEST_PREDICTION_PATH = "data/test_predicted_images/"
