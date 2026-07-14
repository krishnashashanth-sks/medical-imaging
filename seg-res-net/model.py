from tensorflow import keras
from tensorflow.keras import layers
from layers import advanced_residual_block

def AdvancedSegResNet(input_shape, num_classes, filters=(32, 64, 128, 256, 512)):
    """
    Advanced SegResNet architecture with pre-activation residual blocks and Squeeze-and-Excitation modules.
    Uses a U-Net-like encoder-decoder structure with advanced residual blocks.

    Args:
        input_shape (tuple): Shape of the input image (H, W, C).
        num_classes (int): Number of segmentation classes.
        filters (tuple): A tuple specifying the number of filters at each downsampling/upsampling stage.
                         The length of this tuple defines the depth of the U-Net-like structure.
    """
    inputs = keras.Input(input_shape)

    # Initial convolution (post-activation for the very first layer)
    x = layers.Conv2D(filters[0], kernel_size=3, padding='same', use_bias=False, name='init_conv')(inputs)
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.Activation('relu', name='init_act')(x)

    skip_connections = []

    # Encoder path
    for i, f in enumerate(filters):
        # Two advanced residual blocks at each level
        x = advanced_residual_block(x, f, name=f'encoder_res_block{i}_1')
        x = advanced_residual_block(x, f, name=f'encoder_res_block{i}_2')
        skip_connections.append(x) # Store for skip connections BEFORE downsampling

        if i < len(filters) - 1:
            x = layers.MaxPool2D((2, 2), name=f'encoder_pool{i}')(x)

    # Bottleneck (last element in filters is bottleneck filter size)
    x = advanced_residual_block(x, filters[-1], name='bottleneck_res_block1')
    x = advanced_residual_block(x, filters[-1], name='bottleneck_res_block2')

    # Decoder path
    # Iterate from the second to last encoder stage backwards
    for i in reversed(range(len(filters) - 1)):
        f = filters[i] # Target filters for this decoder stage

        # Upsampling
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=2, padding='same', name=f'decoder_upconv{i}')(x)

        # Concatenate with skip connection
        encoder_skip = skip_connections[i]
        x = layers.Concatenate(axis=-1, name=f'decoder_concat{i}')([x, encoder_skip])

        # Two advanced residual blocks
        x = advanced_residual_block(x, f, name=f'decoder_res_block{i}_1')
        x = advanced_residual_block(x, f, name=f'decoder_res_block{i}_2')

    # Output layer
    output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation=output_activation, padding='same', name='output_conv')(x)

    model = keras.Model(inputs, outputs, name='AdvancedSegResNet')
    return model

