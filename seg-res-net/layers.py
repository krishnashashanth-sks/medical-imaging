from tensorflow.keras import layers

def conv_block(input_tensor, num_filters, kernel_size=3, strides=1, activation='relu', use_bias=False, name=None):
    """Standard convolution block with Pre-activation (BN -> Act -> Conv)."""
    x = layers.BatchNormalization(name=name + '_bn' if name else None)(input_tensor)
    if activation:
        x = layers.Activation(activation, name=name + '_act' if name else None)(x)
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias, name=name + '_conv' if name else None)(x)
    return x

def se_block(input_tensor, ratio=16, name=None):
    """Squeeze-and-Excitation block to recalibrate channel-wise features."""
    channels = input_tensor.shape[-1]
    se_squeeze = layers.GlobalAveragePooling2D(name=name + '_se_squeeze' if name else None)(input_tensor)
    se_excitation = layers.Dense(channels // ratio, activation='relu', name=name + '_se_excitation1' if name else None)(se_squeeze)
    se_excitation = layers.Dense(channels, activation='sigmoid', name=name + '_se_excitation2' if name else None)(se_excitation)
    se_excitation = layers.Reshape((1, 1, channels), name=name + '_se_reshape' if name else None)(se_excitation)
    x = layers.Multiply(name=name + '_se_multiply' if name else None)([input_tensor, se_excitation])
    return x

def advanced_residual_block(input_tensor, num_filters, strides=1, name=None):
    """A pre-activation residual block with two convolutional layers and an SE block."""
    shortcut = input_tensor

    # Shortcut path for dimension mismatch
    if strides != 1 or input_tensor.shape[-1] != num_filters:
        shortcut = layers.Conv2D(num_filters, kernel_size=1, strides=strides, padding='same', use_bias=False, name=name + '_shortcut_conv' if name else None)(input_tensor)
        shortcut = layers.BatchNormalization(name=name + '_shortcut_bn' if name else None)(shortcut)

    # First convolutional layer (BN -> Act -> Conv)
    x = conv_block(input_tensor, num_filters, strides=strides, name=name + '_block1')

    # Second convolutional layer (BN -> Act -> Conv, but no final activation here)
    x = conv_block(x, num_filters, activation=None, name=name + '_block2') # Activation is handled after adding shortcut

    # Apply Squeeze-and-Excitation block
    x = se_block(x, name=name + '_se_block')

    # Add shortcut to the main path
    x = layers.Add(name=name + '_add' if name else None)([x, shortcut])
    return x
