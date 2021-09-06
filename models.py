import torch 
import torch.nn as nn 


class NextFrameModel(nn.Module):
    '''A learnable model that given a 256x256 input at z=t,
       predicts an output 256x256 at z=t+1. 
    '''
    def __init__(self):
        # See here for documentation on 2D convolutions
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=7,
                               padding=3)
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=17,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=33,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.act3 = nn.LeakyReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=65,
                               out_channels=128,
                               kernel_size=1)
        self.act4 = nn.LeakyReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.bn4 = nn.BatchNorm2d(128)

        self.deconv1 = nn.ConvTranspose2d(in_channels=129,
                                          out_channels=64,
                                          kernel_size=8)

        self.deconv_act1 = nn.Sigmoid()
        self.bn5 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(in_channels=65,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=4)
        self.deconv_act2 = nn.Sigmoid()
        self.bn6 = nn.BatchNorm2d(32)

        self.deconv3 = nn.ConvTranspose2d(in_channels=33,
                                          out_channels=16,
                                          kernel_size=2,
                                          stride=2)
        self.deconv_act3 = nn.Sigmoid()
        self.bn7 = nn.BatchNorm2d(16)

        self.deconv4 = nn.ConvTranspose2d(in_channels=17,
                                          out_channels=8,
                                          kernel_size=4,
                                          stride=4)
        self.deconv_act4 = nn.Sigmoid()
        self.bn8 = nn.BatchNorm2d(8)

        self.combine = nn.Conv2d(in_channels=9,
                                 out_channels=1,
                                 kernel_size=1)

    def forward(self, input_frame):
        '''
        Completes a forward pass through the model which vaguely mimicks a U-Net
        args:
            input_frame (Tensor): PyTorch tensor with shape (batch_size, 256, 256)
        returns:
            (Tensor): Next predicted frame of shape (batch_size, 256, 256)
        '''
        x = self.conv1(input_frame)  # output shape: (batch_size, 16, 256, 256)
        x = self.act1(x)  # output shape: (batch_size, 16, 256, 256)
        x = self.bn1(x) # output shape: (batch_size, 16, 256, 256)

        x_1 = self.pool1(x)  # output shape: (batch_size, 16, 64, 64)
        
        downsample_IF_1 = self.pool1(input_frame)  # output shape: (batch_size, 1, 64, 64)
        
        x = torch.cat((x_1, downsample_IF_1), dim=1)  # output shape: (batch_size, 17, 64, 64)
        
        x = self.conv2(x)  # output shape: (batch_size, 32, 64, 64)
        x = self.act2(x)  # output shape: (batch_size, 32, 64, 64)
        x = self.bn2(x)  # output shape: (batch_size, 32, 64, 64)

        x_2 = self.pool2(x)  # output shape: (batch_size, 32, 32, 32)
        
        downsample_IF_2 = self.pool2(downsample_IF_1)  # output shape: (batch_size, 1, 32, 32)
    
        x = torch.cat((x_2, downsample_IF_2), dim=1)  # output shape: (batch_size, 33, 32, 32)
        
        x = self.conv3(x)  # output shape: (batch_size, 64, 32, 32)
        x = self.act3(x)  # output shape: (batch_size, 64, 32, 32)
        x = self.bn3(x)  # output shape: (batch_size, 64, 32, 32)
    
        x_3 = self.pool3(x)  # output shape: (batch_size, 64, 8, 8)

        downsample_IF_3 = self.pool3(downsample_IF_2)  # output shape: (batch_size, 1, 8, 8)

        x = torch.cat((x_3, downsample_IF_3), dim=1)  # output shape: (batch_size, 65, 8, 8)

        x = self.conv4(x)  # output shape: (batch_size, 128, 8, 8)
        x = self.act4(x)  # output shape: (batch_size, 128, 8, 8)
        x = self.bn4(x)  # output shape: (batch_size, 64, 32, 32)

        x_4 = self.pool4(x)  # output shape: (batch_size, 128, 1, 1)

        downsample_IF_4 = self.pool4(downsample_IF_3)  # output shape: (batch_size, 1, 1, 1)

        x = torch.cat((x_4, downsample_IF_4), dim=1)  # output shape: (batch_size, 129, 1, 1)

        x = self.bn5(self.deconv_act1(self.deconv1(x)))  # output shape: (batch_size, 64, 8, 8)

        # DEBUG: print(x.shape)
        x = torch.cat((x, downsample_IF_3), dim=1)  # output shape: (batch_size, 65, 8, 8)

        x = self.bn6(self.deconv_act2(self.deconv2(x)))  # output shape: (batch_size, 32, 32, 32)

        # DEBUG: print(x.shape)
        x = torch.cat((x, downsample_IF_2), dim=1)  # output shape: (batch_size, 33, 32, 32)

        x = self.bn7(self.deconv_act3(self.deconv3(x)))  # output shape: (batch_size, 16, 64, 64)

        # DEBUG: print(x.shape)
        x = torch.cat((x, downsample_IF_1), dim=1)  # output shape: (batch_size, 17, 64, 64)

        x = self.bn8(self.deconv_act4(self.deconv4(x)))  # output shape: (batch_size, 8, 256, 256)

        # DEBUG: print(x.shape)
        x = torch.cat((x, input_frame), dim=1)  # output shape: (batch_size, 9, 256, 256)

        x = self.combine(x)  # output shape: (batch_size, 1, 256, 256)

        next_frame = input_frame + x 
        return next_frame


if __name__ == '__main__':
  
  x = torch.rand(32, 1, 256, 256) #  Simulated batch of data
  x = x.cuda()

  model = NextFrameModel().cuda()

  next_frame = model(x)

  print(next_frame.shape)
