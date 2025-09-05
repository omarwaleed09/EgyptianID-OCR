import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class MinPooling2D(layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides) if strides is not None else tuple(pool_size)
        self.padding = padding.upper()

    def call(self, inputs):
        return -tf.nn.max_pool2d(
            -inputs,
            ksize=self.pool_size,
            strides=self.strides,
            padding=self.padding
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config
