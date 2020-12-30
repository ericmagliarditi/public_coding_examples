from torch import nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        
        #Contraction:
        self.contract_1 = self.contracting_block(in_channels=in_channel, out_channels=32, kernel_size=7)
        self.contract_max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.contract_2 = self.contracting_block(in_channels=32, out_channels=64, kernel_size=3)
        self.contract_max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.contract_3 = self.contracting_block(in_channels=64, out_channels=128, kernel_size=3)
        self.contract_max_pool_3 = nn.MaxPool2d(kernel_size=2)
        self.contract_4 = self.contracting_block(in_channels=128, out_channels=256, kernel_size=3)
        self.contract_max_pool_4 = nn.MaxPool2d(kernel_size=2)
        
        #Bottleneck:
        self.bottleneck = self.bottle_neck(in_channels=256, out_channels=512, kernel_size=3)
        
        #Expansion:
        self.expansion_3 = self.expansive_block(in_channels=512, mid_channel=256, out_channels=128)
        self.expansion_2 = self.expansive_block(in_channels=256, mid_channel=128, out_channels=64)
        self.expansion_1 = self.expansive_block(kernel_size=7,
            in_channels=128, mid_channel=64, out_channels=32)
        
        #Final Layer
        self.final_layer = self.final_output_layer(64, 32, out_channel)
        
    def forward(self, x):
        #Contraction AKA Encoding
        encode_block_1 = self.contract_1(x)
        encode_pool_1 = self.contract_max_pool_1(encode_block_1)
        encode_block_2 = self.contract_2(encode_pool_1)
        encode_pool_2 = self.contract_max_pool_2(encode_block_2)
        encode_block_3 = self.contract_3(encode_pool_2)
        encode_pool_3 = self.contract_max_pool_3(encode_block_3)
        encode_block_4 = self.contract_4(encode_pool_3)
        encode_pool_4 = self.contract_max_pool_4(encode_block_4)

        #Bottleneck
        bottleneck = self.bottleneck(encode_pool_4)
        
        #Expansion - AKA Decoding
        cat_layer_3 = self.crop_and_concat(bottleneck, encode_block_4, crop=True)
        decode_block_3 = self.expansion_3(cat_layer_3)
        cat_layer_2 = self.crop_and_concat(decode_block_3, encode_block_3, crop=True)
        decode_block_2 = self.expansion_2(cat_layer_2)
        cat_layer_1 = self.crop_and_concat(decode_block_2, encode_block_2, crop=True)
        decode_block_1 = self.expansion_1(cat_layer_1)
        cat_layer_0 = self.crop_and_concat(decode_block_1, encode_block_1, crop=True)
        final_layer = self.final_layer(cat_layer_0)

        # print(f"Input shape: {x.shape}")
        # print(f" Contract 1 Batch Shape: {encode_block_1.shape}")
        # print(f" Contract 1 Batch Shape After Pooling: {encode_pool_1.shape}")
        # print(f" Contract 2 Batch Shape: {encode_block_2.shape}")
        # print(f" Contract 2 Batch Shape After Pooling: {encode_pool_2.shape}")
        # print(f" Contract 3 Batch Shape: {encode_block_3.shape}")
        # print(f" Contract 3 Batch Shape After Pooling: {encode_pool_3.shape}")
        # print(f" Contract 4 Batch Shape: {encode_block_4.shape}")
        # print(f" Contract 4 Batch Shape After Pooling: {encode_pool_4.shape}")
        # print(f" Batch Shape After Bottlenect: {bottleneck.shape}")
        # print(f"Concatenation 1 Shape: {cat_layer_3.shape}")
        # print(f" Exapansion 1 Batch Shape: {decode_block_3.shape}")
        # print(f"Concatenation 2 Shape: {cat_layer_2.shape}")
        # print(f" Exapansion 2 Batch Shape: {decode_block_2.shape}")
        # print(f"Concatenation 3 Shape: {cat_layer_1.shape}")
        # print(f"Expansion 3 Shape: {decode_block_1.shape}")
        # print(f"Concatenation 4 shape: {cat_layer_0.shape}")
        # print(f"Final Layer Output Shape: {final_layer.shape}")
        
        return final_layer
               
    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
            a = torch.cat((upsampled, bypass), 1)
        return torch.cat((upsampled, bypass), 1)


    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        '''
        Creates a single contracting block
        
        :param in_channels: number of channels in the input image (3 for first input layer - RGB)
        :paramtype in_channels: int
        
        :param out_channels: number of channels produced by the convolution
        :paramtype out_channels: int
        
        :param kernel_size: size of the convolving kernel
        :paramtype kernel_size: int or tuple
        
        ..note: The contraction section is made up of many contraction blocks!
        
        ..future_work: Can also include batch normalization or dropout
        '''
        
        contracting_block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            nn.BatchNorm2d(out_channels)
        )
        
        return contracting_block

    def bottle_neck(self, in_channels=256, out_channels=512, kernel_size=3):
        '''
        Creates the Bottle Neck Block - one 1 needed!
        
        :param in_channels: number of channels in the input image (3 for first input layer - RGB)
        :paramtype in_channels: int
        
        :param out_channels: number of channels produced by the convolution
        :paramtype out_channels: int
        
        :param kernel_size: size of the convolving kernel
        :paramtype kernel_size: int or tuple
        
        ..note: The contraction section is made up of many contraction blocks!
        
        ..future_work: Can also include batch normalization or dropout
        '''
        
        bottle_neck_block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.ELU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            #Upsample
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels,
                              kernel_size=kernel_size, stride=2, padding=0, output_padding=1, dilation=4)
        )
        
        return bottle_neck_block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        '''
        Creates a single expansion block
        
        :param in_channels: number of channels in the input image (3 for first input layer - RGB)
        :paramtype in_channels: int
        
        :param out_channels: number of channels produced by the convolution
        :paramtype out_channels: int
        
        :param kernel_size: size of the convolving kernel
        :paramtype kernel_size: int or tuple
        
        ..note: The contraction section is made up of many contraction blocks!
        
        ..future_work: Can also include batch normalization or dropout
        '''
        
        expansive_block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ELU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ELU(),
            nn.BatchNorm2d(mid_channel),
            #Now need to apply upsampling
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels,
                               kernel_size=kernel_size, stride=2, padding=0, output_padding=1, dilation=4)
        )
        
        return expansive_block
    
    
    
    def final_output_layer(self, in_channels, mid_channel, out_channels, kernel_size=3):
        '''
        Creates the final Layer
        
        :param in_channels: number of channels in the input image (3 for first input layer - RGB)
        :paramtype in_channels: int
        
        :param out_channels: number of channels produced by the convolution
        :paramtype out_channels: int
        
        :param kernel_size: size of the convolving kernel
        :paramtype kernel_size: int or tuple
        
        ..note: The contraction section is made up of many contraction blocks!
        
        ..future_work: Can also include batch normalization or dropout
        '''
        
        final_layer = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ELU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ELU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels,
                     padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
        )
        
        return final_layer
    