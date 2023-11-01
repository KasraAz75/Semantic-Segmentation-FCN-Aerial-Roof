# model.py
import torch
import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    """
    Fully Connected Network (FCN) class.
    This class implements a simple FCN with 5 convolutional layers.
    """
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        # Define the layers of the network
        self.convolution1 = nn.Conv2d(3, 16, 3, padding=1)          # Conv 3x3, 16 filters
        self.convolution2 = nn.Conv2d(16, 16, 5, padding=2)         # Conv 5x5, 16 filters
        self.pooling      = nn.MaxPool2d(2, 2)                      # Max Pooling
        self.convolution3 = nn.Conv2d(16, 32, 3, padding=1)         # Conv 3x3, 32 filters
        self.upsampling   = nn.ConvTranspose2d(32, 32, 2, stride=2) # Transposed Conv 2x2, 32 filters
        self.convolution4 = nn.Conv2d(32, 16, 3, padding=1)         # Conv 3x3, 16 filters
        self.convolution5 = nn.Conv2d(16, 1, 1)                     # Output layer

    def forward(self, input_tensor): 
        """
        Forward pass of the FCN.
        Args:
            input_tensor (torch.Tensor): Input tensor for the forward pass.
        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        # Pass the input through each layer of the network
        tensor = torch.relu(self.convolution1(input_tensor))  
        tensor = torch.relu(self.convolution2(tensor))  
        tensor = self.pooling(tensor)  
        tensor = torch.relu(self.convolution3(tensor))  
        tensor = torch.relu(self.upsampling(tensor))  
        tensor = torch.relu(self.convolution4(tensor))  
        output_tensor = self.convolution5(tensor)  
        
        return output_tensor


def test_network():
    """
    Test function for the FCN.
    This function creates a random input tensor and passes it through the FCN.
    It then prints the shapes of the input and output tensors.
    """
    # Create a random input tensor
    input_tensor = torch.randn((3, 3, 512, 512))
    
    # Initialize the model & pass the input through the model
    model = FullyConnectedNetwork()
    output_tensor = model(input_tensor)
    
    # Print the shapes of the input and output tensors
    print(f"Input tensor shape: {input_tensor.shape}")   # [3, 3, 512, 512]
    print(f"Output tensor shape: {output_tensor.shape}") # [3, 1, 512, 512]


if __name__ == "__main__":
    test_network()
