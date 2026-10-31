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


class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dimension=256, num_heads=4, dropout=0.1,  qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embedding_dimension // num_heads
        self.embedding_dimension=embedding_dimension

        self.w_k=nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.w_q=nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.w_v=nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.Dropout=nn.Dropout(dropout)

    def forward(self,x):

        b, num_tokens, d_in = x.shape

        keys=self.w_k(x)
        queries=self.w_q(x)
        values=self.w_v(x)
        """
        implementing multi-head attention
        """
        #[batch, token, num_head, layer_dim]
        keys= keys.view(b, num_tokens, self.num_heads,self.head_dim)
        values=values.view(b, num_tokens, self.num_heads,self.head_dim)
        queries=queries.view(b, num_tokens, self.num_heads,self.head_dim)
     
        #[batch, token, num_head, layer_dim] --> [batch,num_heads,num_tokens, head_dim]

        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        queries=queries.transpose(1,2)

        attention_score= queries @ keys.T
        atention_weights= torch.softmax((attention_score/keys.shape[-1]**0.5), dim=-1)
        attention_weights=self.Dropout(attention_weights)

        context_vector= attention_weights@ values

        context_vector= context_vector.contiguous().view(b, num_tokens, self.embedding_dimension)

        return context_vector


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels,out_channels, kernal_size, stride, padding=0, bias=False):
        super().__init__()

        self.conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernal_size,kernal_size),
                  stride=stride, padding=padding , bias=bias)

    def forward(self,x ):
        x=self.conv(x)
        x= torch.flatten(x, start_dim=2, end_dim=-1)
        x= x.transpose(1,2) #[batch, token, embedding_dimension]
        return x
    

        

        







