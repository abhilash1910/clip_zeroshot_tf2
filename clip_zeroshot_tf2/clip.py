# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:46:20 2022

@author: Abhilash
"""

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np

class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, name: str = "bottleneck"):
        super().__init__(name=name)

        with tf.name_scope(name):
            # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
            self.conv1 = tf.keras.layers.Conv2D(planes, 1, use_bias=False, name="conv1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1", epsilon=1e-5)

            self.conv2_padding = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
            self.conv2 = tf.keras.layers.Conv2D(planes, 3, use_bias=False, name="conv2")
            self.bn2 =tf.keras.layers.BatchNormalization(name="bn2", epsilon=1e-5)

            self.avgpool = tf.keras.layers.AveragePooling2D(stride) if stride > 1 else None

            self.conv3 = tf.keras.layers.Conv2D(planes * self.expansion, 1, use_bias=False, name="conv3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3", epsilon=1e-5)

            self.relu =tf.keras.layers.ReLU()
            self.downsample = None
            self.stride = stride

            self.inplanes = inplanes
            self.planes = planes

            if stride > 1 or inplanes != planes * Bottleneck.expansion:
                # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
                self.downsample = keras.Sequential([
                    tf.keras.layers.AveragePooling2D(stride, name=name + "/downsample/avgpool"),
                    tf.keras.layers.Conv2D(planes * self.expansion, 1, strides=1, use_bias=False, name=name + "/downsample/0"),
                    tf.keras.layers.BatchNormalization(name=name + "/downsample/1", epsilon=1e-5)
                ], name="downsample")

    def get_config(self):
        return {
            "inplanes": self.inplanes,
            "planes": self.planes,
            "stride": self.stride,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(self.conv2_padding(out))))
        if self.avgpool is not None:
            out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            # x = tf.nn.avg_pool(x, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(tf.keras.layers.Layer):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None,
                 name="AttentionPool2d"):
        super().__init__(name=name)

        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        with tf.name_scope(name):
            self.positional_embedding = tf.Variable(
                tf.random.normal((spatial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5,
                name="positional_embedding"
            )

        self.num_heads = num_heads
        self._key_dim = embed_dim

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            output_shape=output_dim or embed_dim,
            name="mha"
        )

    def get_config(self):
        return {
            "spatial_dim": self.spatial_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=None):
        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))  # NHWC -> N(HW)C

        x = tf.concat([tf.reduce_mean(x, axis=1, keepdims=True), x], axis=1)  # N(HW+1)C
        x = x + tf.cast(self.positional_embedding[None, :, :], x.dtype)  # N(HW+1)C

        query, key, value = x, x, x
        x = self.multi_head_attention(query, value, key)

        # only return the first element in the sequence
        return x[:, 0, ...]


class ModifiedResNet(keras.Model):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, name="ModifiedResNet"):
        super().__init__(name=name)
        self.layers_config = layers
        self.output_dim = output_dim
        self.heads = heads
        self.input_resolution = input_resolution
        self.width = width

        # the 3-layer stem
        self.conv1_padding = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv1_padding")
        self.conv1 = tf.keras.layers.Conv2D(width // 2, 3, strides=2, use_bias=False, name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1", epsilon=1e-5)
        self.conv2_padding = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv2_padding")
        self.conv2 = tf.keras.layers.Conv2D(width // 2, 3, use_bias=False, name="conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2", epsilon=1e-5)
        self.conv3_padding = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="conv3_padding")
        self.conv3 = tf.keras.layers.Conv2D(width, 3, use_bias=False, name="conv3")
        self.bn3 = tf.keras.layers.BatchNormalization(name="bn3", epsilon=1e-5)
        self.avgpool = tf.keras.layers.AveragePooling2D(2, name="avgpool")
        self.relu = tf.keras.layers.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], name=name + "/layer1")
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, name=name + "/layer2")
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, name=name + "/layer3")
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, name=name + "/layer4")

        embed_dim = width * 32  # the ResNet feature dimension
        with tf.name_scope(name):
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, name="attnpool")

    def get_config(self):
        return {
            "layers": self.layers_config,
            "output_dim": self.output_dim,
            "heads": self.heads,
            "input_resolution": self.input_resolution,
            "width": self.width,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _make_layer(self, planes, blocks, stride=1, name="layer"):
        with tf.name_scope(name):
            layers = [Bottleneck(self._inplanes, planes, stride, name=name + "/0")]

            self._inplanes = planes * Bottleneck.expansion
            for i in range(1, blocks):
                layers.append(Bottleneck(self._inplanes, planes, name=name + f"/{i}"))

            return keras.Sequential(layers, name="bla")

    def call(self, x):
        def stem(x):
            for conv_pad, conv, bn in [
                (self.conv1_padding, self.conv1, self.bn1),
                (self.conv2_padding, self.conv2, self.bn2),
                (self.conv3_padding, self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(conv_pad(x))))
            x = self.avgpool(x)
            return x

        # x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(tf.keras.layers.LayerNormalization):
    
    """
    Uses keras layer norm implementation
    """
    def __init__(self, name="LayerNorm"):
        super(LayerNorm, self).__init__(epsilon=1e-05, name=name)

    def call(self, x: tf.Tensor):
        return super().call(x)
    
class QuickGELU(tf.keras.layers.Layer):
    
    """
    Uses tensorflow nn gelu backend
    """
    
    def __init__(self, name="QuickGELU"):
        super(QuickGELU, self).__init__(name=name)

    def call(self, x: tf.Tensor):
        #return x * tf.sigmoid(1.702 * x)
        return tf.nn.gelu(x)

class ResidualAttentionBlock(tf.keras.layers.Layer):
    
    """
    Residual attention module which uses keras MHSA followed by 
    Sequential model of layernorm and Dense layers
    """
    
    def __init__(self, d_model: int, n_head: int, attn_mask: tf.Tensor = None, name="ResidualAttentionBlock", idx=0):
        super().__init__(name=name)
        self.idx = idx

        self.d_model = d_model
        self.n_head = n_head

        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=n_head, key_dim=d_model // n_head, name="attn")
        self.ln_1 = LayerNorm(name="ln_1")
        self.mlp = keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, name=name + "/mlp/c_fc"),
            QuickGELU(name=name + "/mlp/gelu"),
            tf.keras.layers.Dense(d_model, name=name + "/mlp/c_proj")
        ], name="mlp")
        self.ln_2 = LayerNorm(name="ln_2")
        self.attn_mask = attn_mask

    def attention(self, x: tf.Tensor):
        return self.attn(x, x, x, attention_mask=self.attn_mask)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(keras.Model):
    
    """
    Default Transformer class which uses Residual Attention Block
    """
    
    def __init__(self, width: int, layers: int, heads: int, attn_mask: tf.Tensor = None, name="transformer"):
        super().__init__(name=name)
        self.width = width
        self.num_layers = layers
        self.heads = heads
        self.attn_mask = attn_mask
        self.resblocks = keras.Sequential([
            ResidualAttentionBlock(width, heads, attn_mask, name=name + f".resblocks.{i}", idx=i)
            for i in range(layers)
        ], name=name + ".resblocks")

    def get_config(self):
        return {
            "width": self.width,
            "layers": self.num_layers,
            "heads": self.heads,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
        return self.resblocks(x)
    
class VisualTransformer(keras.Model):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, name="VisualTransformer"):
        super().__init__(name=name)
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.num_layers = layers
        self.heads = heads
        self.output_dim = output_dim

        self.conv1 = tf.keras.layers.Conv2D(width, patch_size, strides=patch_size, use_bias=False, name="conv1")

        scale = width ** -0.5

        self.transformer = Transformer(width, layers, heads, name=name + "/transformer")

        with tf.name_scope(name):
            self.class_embedding = tf.Variable(scale * tf.random.normal((width,)), name="class_embedding")
            self.positional_embedding = tf.Variable(scale * tf.random.normal(((input_resolution // patch_size) ** 2 + 1, width)), name="positional_embedding")
            self.ln_pre = LayerNorm(name="ln_pre")

            self.ln_post = LayerNorm(name="ln_post")
            self.proj = tf.Variable(scale * tf.random.normal((width, output_dim)), name="proj")

    def get_config(self):
        return {
            "input_resolution": self.input_resolution,
            "patch_size": self.patch_size,
            "width": self.width,
            "layers": self.num_layers,
            "heads": self.heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
        x = self.conv1(x)  # shape = [*, grid, grid, width]

        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], x_shape[1]*x_shape[2], x_shape[3]))  # shape = [*, grid ** 2, width]

        x_shape = tf.shape(x)
        x = tf.concat([tf.broadcast_to(tf.cast(self.class_embedding, x.dtype), (x_shape[0], 1, x_shape[-1])), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + tf.cast(self.positional_embedding, x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    
class CLIP(keras.Model):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.image_resolution = image_resolution
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                name="visual"
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                name="visual"
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            name="transformer"
        )

        self.vocab_size = vocab_size
        self.token_embedding = tf.Variable(tf.zeros((vocab_size, transformer_width)), name="token_embedding")
        self.positional_embedding = tf.Variable(tf.zeros((self.context_length, transformer_width)), name="positional_embedding")
        self.ln_final = LayerNorm(name="ln_final")

        self.text_projection = tf.Variable(tf.zeros((transformer_width, embed_dim)), name="text_projection")
        self.logit_scale = tf.Variable(np.ones([]) * np.log(1 / 0.07), dtype=tf.float32, name="logit_scale")

        #self.initialize_parameters() TODO: get this working again

    # def build(self, input_shape):
    #     super(CLIP, self).build(input_shape)
        

    def initialize_parameters(self):
        self.token_embedding.assign(tf.random.normal(self.token_embedding.shape, stddev=0.02))
        self.positional_embedding.assign(tf.random.normal(self.positional_embedding.shape, stddev=0.01))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                self.visual.attnpool.q_proj.weight.assign(tf.random.normal(self.visual.attnpool.q_proj.weight.shape, stddev=std))
                self.visual.attnpool.k_proj.weight.assign(tf.random.normal(self.visual.attnpool.k_proj.weight.shape, stddev=std))
                self.visual.attnpool.v_proj.weight.assign(tf.random.normal(self.visual.attnpool.v_proj.weight.shape, stddev=std))
                self.visual.attnpool.c_proj.weight.assign(tf.random.normal(self.visual.attnpool.c_proj.weight.shape, stddev=std))

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        tf.nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            tf.nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            tf.nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            tf.nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            tf.nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            tf.nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        n_dest = self.context_length
        n_src = self.context_length
        dtype = tf.bool
        batch_size = 1

        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

        mask = np.ones((self.context_length, self.context_length))
        #mask = np.triu(mask, 1)
        mask = tf.constant(mask)
        mask = 1 - tf.linalg.band_part(tf.ones((self.context_length, self.context_length)), -1, 0)


        #mask.fill_(float("-inf"))
        #mask.triu_(1)  # zero out the lower diagonal

        # import torch
        # masko = torch.empty(self.context_length, self.context_length)
        # masko.fill_(float("-inf"))
        # masko.triu_(1)  # zero out the lower diagonal
        # return tf.constant(masko.cpu().detach().numpy())

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = tf.nn.embedding_lookup(self.token_embedding, text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        x_shape = tf.shape(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_token = tf.argmax(text, axis=-1)
        idx = tf.transpose(tf.stack((tf.range(0, x_shape[0], dtype=tf.int64), eot_token), axis=0, name='take_features_idx'))
        x = tf.gather_nd(x, idx) @ self.text_projection

        return x

    @tf.function(input_signature=[(
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name="image"),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int64, name="text")
    )])
    def call(self, input: Tuple[tf.Tensor, tf.Tensor]):
        image, text = input
        image_features = self.encode_image(image)

        text = tf.squeeze(text, axis=0) # TODO: find another way to feed data, but keras requires that all input tensors have to have the same batch size
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / tf.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / tf.norm(text_features, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ tf.transpose(text_features)
        logits_per_text = logit_scale * text_features @ tf.transpose(image_features)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text    