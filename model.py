from tensorflow.keras import layers, models
import tensorflow as tf
from hyper_params import *


class ResidualLayer(layers.Layer):
    def __init__(self, filters, kernel_size, kernel_regularizer):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                   kernel_regularizer=kernel_regularizer)
        self.b_norm1 = layers.BatchNormalization(momentum=BATCH_MOMENTUM)
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                   kernel_regularizer=kernel_regularizer)
        self.b_norm2 = layers.BatchNormalization(momentum=BATCH_MOMENTUM)
        self.add = layers.Add()
        self.relu2 = layers.ReLU()

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.b_norm1(y, training=training)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.b_norm2(y, training=training)
        out = self.add([x, y])
        out = self.relu2(out)
        return out


class ConvolutionalLayer(models.Model):
    def __init__(self, filters, kernel_size, kernel_regularizer):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding='same',
                                  kernel_regularizer=kernel_regularizer)
        self.b_norm = layers.BatchNormalization(momentum=BATCH_MOMENTUM)
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.b_norm(y, training=training)
        out = self.relu(y)
        return out


class ValueHead(layers.Layer):
    def __init__(self, hidden_layer, kernel_regularizer, name='value1_head'):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(4, 1, padding='same',
                                   kernel_regularizer=kernel_regularizer)
        self.b_norm1 = layers.BatchNormalization(momentum=BATCH_MOMENTUM)
        self.relu1 = layers.ReLU()
        self.flatten1 = layers.Flatten()
        self.fc1 = layers.Dense(hidden_layer,
                                kernel_regularizer=kernel_regularizer)
        self.relu2 = layers.ReLU()
        self.fc2 = layers.Dense(1, activation="tanh", name='value_head',
                                kernel_regularizer=kernel_regularizer)

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.b_norm1(y, training=training)
        y = self.relu1(y)
        y = self.flatten1(y)
        y = self.fc1(y)
        y = self.relu2(y)
        out = self.fc2(y)
        return out


class PolicyHead(layers.Layer):
    def __init__(self, output_size, kernel_regularizer, name='policy1_head'):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(4, 1, padding='same', kernel_regularizer=kernel_regularizer)
        self.b_norm1 = layers.BatchNormalization(momentum=BATCH_MOMENTUM)
        self.relu1 = layers.ReLU()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(output_size, kernel_regularizer=kernel_regularizer)

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.b_norm1(y, training=training)
        y = self.relu1(y)
        y = self.flatten(y)
        out = self.dense(y)
        return out


class AlphaZeroNetwork(models.Model):
    def __init__(self, num_actions, residual_layers=RESIDUAL_LAYERS, num_filters=NUM_FILTERS, l2=L2):
        super().__init__()
        self.conv = ConvolutionalLayer(num_filters, 3, kernel_regularizer=tf.keras.regularizers.L2(l2=l2))
        self.residual = ResidualLayer(num_filters, 3, kernel_regularizer=tf.keras.regularizers.L2(l2=l2))
        self.value_head = ValueHead(num_filters, kernel_regularizer=tf.keras.regularizers.L2(l2=l2))
        self.policy_head = PolicyHead(num_actions, kernel_regularizer=tf.keras.regularizers.L2(l2=l2))
        self.residual_layers = residual_layers

    def call(self, inputs, training=False):
        y = self.conv(inputs, training=training)
        for i in range(self.residual_layers):
            y = self.residual(y, training=training)
        value = self.value_head(y, training=training)
        policy = self.policy_head(y, training=training)
        return value, policy

    def get_config(self):
        config = super().get_config()
        return config


if __name__ == '__main__':
    model = AlphaZeroNetwork(3)
    a = tf.constant([[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]])
    print(a, model(a))
