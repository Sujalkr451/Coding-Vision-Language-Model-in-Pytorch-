import torch
import torch.nn as nn

class ImageFeatureMap(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv = nn.Sequential(
            # Standard Conv: 224 -> 112
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 112 -> 112
            nn.Conv2d(32, 32, 3, stride=1, padding=1,
                      groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Pointwise Conv: 32 -> 64 channels
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 112 -> 56
            nn.Conv2d(64, 64, 3, stride=2, padding=1,
                      groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Pointwise Conv: 64 -> 128 channels
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Depthwise Conv: 56 -> 56
            nn.Conv2d(128, 128, 3, stride=1, padding=1,
                      groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 

            #PointWise Conv: 128 -> 128 channels 
            nn.Conv2d(in_channels=128, out_channels=128,
                      stride=1, groups=1, bias=False, padding=1, kernel_size=(1,1)), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 


            #DepthWise Conv 128 -> 128 channels

            nn.Conv2d(in_channels=128, out_channels=128, groups=128, 
                      stride=2, padding=1, bias=False, kernel_size=(3,3)), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 


            #Pointwise Conv 128 -> 256 channels, 56x56 -> 28x28 
            nn.Conv2d(
                   in_channels=128,
                   out_channels=256,
                   groups=1,
                   bias=False,
                   stride=1,
                   kernel_size=(1, 1)
                     ),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)
                   )

    def forward(self, x):
        x=self.Conv(x)


class MemeTransformer(nn.Module):
    def __init__(self, patch_size=4, qkv_bias=False, d_in=1024, d_out=1024, embedding_dimension=1024,
                 num_head=16):
        super().__init__()

        self.num_head=num_head
        self.embedding_dimension=embedding_dimension

        self.patch_embed= nn.Conv2d(in_channels=256, out_channels=256,
                                    stride=4, kernel_size=(4,4), padding=0, bias=True)
        self.w_k=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_q=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.pos_embedding=nn.Parameter(torch.arange(1, embedding_dimension))

    def forward(self,x):

        layer_dim=(self.embedding_dimension//self.num_head)
        batch, token, embedding_dim=x.shape

        keys=self.w_k(x)
        queries=self.w_q(x)
        values=self.w_v(x)
        """
        implementing multi-head attention
        """
        #[batch, token, num_head, layer_dim]
        keys= keys.view(batch, token, self.num_head,layer_dim)
        values=values.view(batch,token,self.num_head,layer_dim)
        queries=queries.view(batch,token, self.num_head,layer_dim)
        

        







