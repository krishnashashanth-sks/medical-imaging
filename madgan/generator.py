import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model(latent_dim,img_channels,img_height,img_width):
  model=tf.keras.Sequential()
  model.add(layers.Dense(4*4*256,use_bias=False,input_shape=(latent_dim,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Reshape((4,4,256)))

  assert model.output_shape==(None,4,4,256)

  model.add(layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  assert model.output_shape==(None,8,8,128)

  model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  assert model.output_shape==(None,16,16,64)

  model.add(layers.Conv2DTranspose(img_channels,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))
  assert model.output_shape==(None,img_height,img_width,img_channels)
  return model