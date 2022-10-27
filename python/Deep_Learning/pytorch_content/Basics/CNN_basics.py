import torch.nn as nn
import torch.nn.functional as F


##### CREATING a Basic Convolutional Network with MaxPooling Layers and
# Relu Activation Function with a two fully connected layers for the 
# classifier. Creating a class inherited nn.Module ######
class BasicCNN(nn.Module):
    # Example: Tensor torch input: 224x224 with 3 channels

    def __init__(self):
        super(BasicCNN, self).__init__()

        ## Define Layers of a CNN
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) # After pooling: 112x112x8
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # After pooling: 56x56x16
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # After pooling: 28x28x32
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1) # After pooling: 14x14x64

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Classification Layer
        self.fc1 = nn.Linear(14*14*64, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 50) # 50 Classes/landmarks

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        ## Define forward behavior

        # Passing through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flattening the images for the fully connected layers
        x = x.view(x.shape[0], -1)

        # passing through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

#### Helper Functions to create Convolutional Layers with Batchnormalization
#### Bias set to False with batchnorm! 
def create_conv_layer(in_channels, out_channels, filter_size,
                      stride=2, padding=1, batch_norm=True):
    """ Function to create Convolution layer with or without a batch 
        normalization layer if wanted.

    Args:
        in_channels (integer): _description_
        out_channels (integer): _description_
        filter_size (integer or tuple with two integers): _description_
        stride (int, optional): _description_. Defaults to 2.
        padding (int, optional): _description_. Defaults to 1.
        batch_norm (bool, optional): _description_. Defaults to True.

    Returns:
        nn.Sequential: a Sequential Container with the defined layers
    """


    layers = []
    if batch_norm:
        # Creating Conv_2 without bias for batch norm
        conv_layer = nn.Conv2d(in_channels, out_channels, filter_size,
                              stride, padding, bias=False)
        layers.append(conv_layer)

        # Adding Batch norm Layer
        batch_layer = nn.BatchNorm2d(out_channels)
        layers.append(batch_layer)

    else:
        # Conv Layer with bias - no batch norm layer
        conv_layer = nn.Conv2d(in_channels, out_channels, filter_size,
                              stride, padding, bias=True)
        layers.append(conv_layer)

    return nn.Sequential(*layers)


def create_deconv_layer(in_channels, out_channels, kernel_size,
                        stride=2, padding=1, batch_norm=True):
    """Creates a deconvolutional layer, with optional batch normalization.
    :return: 

    Args:
        in_channels (integer): _description_
        out_channels (integer): _description_
        kernel_size (integer or Tuple with two Integers): _description_
        stride (int, optional): _description_. Defaults to 2.
        padding (int, optional): _description_. Defaults to 1.
        batch_norm (bool, optional): _description_. Defaults to True.

    Returns:
        nn.Sequential: a Sequential Container with the defined layers
    """

    layers = []
    t_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)

    # append conv layer
    layers.append(t_conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)

