import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import gc
import streamlit as st

gc.enable()

def conversion_block(input_channel, output_channel, g_pool = False):
    layer = [nn.Conv2d(input_channel, output_channel, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace = True)]
    
    if g_pool:
        layer.append(nn.MaxPool2d(2))
    
    return nn.Sequential(*layer)

class ResNet8(nn.Module):
    def __init__(self, input_channel, num_classes):
        super().__init__()
        # 1 X 28 X 28
        self.conv1 = conversion_block(input_channel, 64) # 64 X 28 X 28
        self.conv2 = conversion_block(64, 128, g_pool = True) # 128 X 14 X 14
        self.res1 = nn.Sequential(conversion_block(128, 128), conversion_block(128, 128)) # 128 X 14 X 14
        
        self.conv3 = conversion_block(128, 256, g_pool = True) # 256 X 7 X 7
        self.res2 = nn.Sequential(conversion_block(256, 256), conversion_block(256, 256))


        self.classifier = nn.Sequential(nn.MaxPool2d(7), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, num_classes))


    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.res1(output) + output
        output = self.conv3(output)
        output = self.res2(output) + output
        output = self.classifier(output)
        return output
    
def transform_image(image):
    stats = ((0.1307), (0.3081))

    transformation = T.Compose([T.ToTensor(), T.Normalize(*stats)])

    return transformation(image)

# https://medium.com/@edwardpie/building
# -a-cnn-for-recognising-mouse-drawn-digits-with-keras-opencv-mnist-72a7ae7a070a

@st.cache
def start_model():

    # Initiate the model
    input_channels = 1
    num_classes = 10
    model = ResNet8(input_channels, num_classes)
    device = torch.device('cpu')
    PATH = 'mnist-resnet.pth'
    model.load_state_dict(torch.load(PATH, map_location = device))
    model.eval()

    return model


def prediction(img):

    # Converting into 1
    n_img = img.unsqueeze(0)

    model = start_model()

    # Get predictions from model
    out_img = model(n_img)
    # Softamx
    out_img_soft = F.softmax(out_img, dim = 1)

    confidence , predictions  = torch.max(out_img_soft, dim=1)
    gc.collect()
    # Retrieve the class label, confidence and probabilities of all classes using sigmoid 
    return predictions[0].item(), math.trunc(confidence.item()*100), torch.sigmoid(out_img).detach()
    






    