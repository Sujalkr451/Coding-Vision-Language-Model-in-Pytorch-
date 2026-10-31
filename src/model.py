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
                      stride=1, groups=1, bias=False, padding=0, kernel_size=(1,1)), 
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

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension=256,
        num_heads=4,
        dropout=0.1,
        qkv_bias=False
    ):
        super().__init__()

        assert embedding_dimension % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embedding_dimension // num_heads
        self.embedding_dimension = embedding_dimension

        self.w_k = nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.w_q = nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.w_v = nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.w_k(x)
        queries = self.w_q(x)
        values = self.w_v(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_score = queries @ keys.transpose(-2, -1)

        attention_weights = torch.softmax(
            attention_score / (self.head_dim ** 0.5),
            dim=-1
        )

        attention_weights = self.attn_dropout(attention_weights)

        context_vector = attention_weights @ values

        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(
            b, num_tokens, self.embedding_dimension
        )

        context_vector = self.out_proj(context_vector)
        context_vector = self.proj_dropout(context_vector)

        return context_vector


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=256,out_channels=256, kernal_size=4, stride=4, padding=0, bias=False):
        super().__init__()

        self.conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernal_size,kernal_size),
                  stride=stride, padding=padding , bias=bias)

    def forward(self,x ):
        x=self.conv(x)
        x= torch.flatten(x, start_dim=2, end_dim=-1)
        x= x.transpose(1,2) #[batch, token, embedding_dimension]
        return x

class MLP(nn.Module):
    def __init__(self, embedding_dimension=256, mlp_ratio=4):
        super().__init__()

        hidden_dim = embedding_dimension * mlp_ratio

        self.net = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dimension)
        )

    def forward(self, x):
        return self.net(x)
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.multiheadattention=MultiHeadAttention()
        self.LayerNorm1=nn.LayerNorm(256)
        self.LayerNorm2=nn.LayerNorm(256)
        self.mlp=MLP()

    def forward(self,x):
        residual1 = x
        x=self.LayerNorm1(x)
        x=self.multiheadattention(x)
        x=x+residual1
        residual2=x
        x=self.LayerNorm2(x)
        x=self.mlp(x)
        x=x+residual2

        return x


class VisionTransformerCLIP(nn.Module):
    def __init__(self, embedding_dimension=256, num_layers=12):
        super().__init__()

        self.feature_map=ImageFeatureMap()
        self.patch_embedding=PatchEmbedding()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dimension)* 0.02
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 50, embedding_dimension)* 0.02
        )
        self.encoder = nn.Sequential(
            *[TransformerEncoder() for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embedding_dimension)
        self.image_projection = nn.Linear(
                             embedding_dimension,
                            256,
                            bias=False
        )

    def forward(self,x):
        x= self.feature_map(x)
        x= self.patch_embedding(x)

        B=x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x)
        image_features = x[:, 0, :]
        image_embeddings = self.image_projection(image_features)

        return image_embeddings


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, num_heads,context_size,embedding_dimension, qkv_bias=False, dropout=0.1):
        super().__init__()
        assert embedding_dimension % num_heads == 0

        self.context_size = context_size
        self.num_heads=num_heads
        self.embedding_dimension=embedding_dimension
        self.layer_dim=self.embedding_dimension//self.num_heads

        self.W_q= nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.W_k=nn.Linear(embedding_dimension, embedding_dimension, bias=qkv_bias)
        self.W_v=nn.Linear(embedding_dimension,embedding_dimension, bias=qkv_bias)
        self.Dropout=nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_size,context_size), diagonal=1))

    def forward(self,x):

        b,num_token,embed_dim=x.shape

        queries=self.W_q(x)
        keys=self.W_k(x)
        values=self.W_v(x)

        queries= queries.view(b,num_token,self.num_heads, self.layer_dim)
        keys=keys.view(b,num_token,self.num_heads, self.layer_dim)
        values=values.view(b,num_token,self.num_heads, self.layer_dim)

        queries=queries.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)

        attention_score=queries@ keys.transpose(2,3)

        attention_score.masked_fill_(
           self.mask.bool()[:num_token, :num_token],
           float('-inf'))

        attention_weights = torch.softmax(
           attention_score / (self.layer_dim ** 0.5),
           dim=-1)
        attention_weights = self.Dropout(attention_weights)
        context_vector = attention_weights @ values

        return context_vector







    

        

        







