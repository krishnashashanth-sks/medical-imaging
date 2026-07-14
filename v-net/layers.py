from tensorflow.keras import layers

def residual_conv_block(input_tensor,num_filters,kernel_size=(3,3,3),batch_norm=True,activation='relu'):
  x=layers.Conv3D(num_filters,kernel_size,padding='same')(input_tensor)
  if batch_norm:
    x=layers.BatchNormalization()(x)
  x=layers.Activation(activation)(x)
  x=layers.Conv3D(num_filters,kernel_size,padding='same')(x)
  if batch_norm:
    x=layers.BatchNormalization()(x)
  if input_tensor.shape[-1]!=num_filters:
    shortcut=layers.Conv3D(num_filters,(1,1,1),padding='same')(input_tensor)
  else:
    shortcut=input_tensor
  x=layers.add([x,shortcut])
  x=layers.Activation(activation)(x)
  return x

def attention_gate(gating_signal,skip_connection_signal,num_filters):
  Wg=layers.Conv3D(num_filters,(1,1,1),padding='same')(gating_signal)
  Wg=layers.BatchNormalization()(Wg)
  Ws=layers.Conv3D(num_filters,(1,1,1),padding='same')(skip_connection_signal)
  Ws=layers.BatchNormalization()(Ws)
  combined_filters=layers.add([Wg,Ws])
  relu=layers.Activation('relu')(combined_filters)
  attention_coeffs=layers.Conv3D(1,(1,1,1),padding='same',activation='sigmoid')(relu)
  attended_skip=layers.multiply([skip_connection_signal,attention_coeffs])
  return attended_skip