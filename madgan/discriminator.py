from tensorflow.keras import layers,Model,Input

def make_discriminator_model(img_shape):
  input_image = Input(shape=img_shape)
  x = layers.Conv2D(64,(5,5),strides=(2,2),padding='same')(input_image)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128,(5,5),strides=(2,2),padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Flatten()(x)
  output = layers.Dense(1)(x)

  model = Model(inputs=input_image, outputs=output)
  return model