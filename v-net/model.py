from tensorflow.keras import layers,Model
from layers import residual_conv_block,attention_gate

def VNet(input_shape=(64,64,64,1),num_classes=1,activation='sigmoid'):
  inputs=layers.Input(input_shape)
  conv1=residual_conv_block(inputs,16)
  pool1=layers.MaxPool3D((2,2,2))(conv1)
  conv2=residual_conv_block(pool1,32)
  pool2=layers.MaxPool3D((2,2,2))(conv2)
  conv3=residual_conv_block(pool2,64)
  pool3=layers.MaxPool3D((2,2,2))(conv3)
  bottleneck=residual_conv_block(pool3,128)
  up3=layers.Conv3DTranspose(64,(2,2,2),strides=(2,2,2),padding='same')(bottleneck)
  attn3=attention_gate(up3,conv3,64)
  concat3=layers.concatenate([up3,attn3],axis=-1)
  up_conv3=residual_conv_block(concat3,64)
  up2=layers.Conv3DTranspose(32,(2,2,2),strides=(2,2,2),padding='same')(up_conv3)
  attn2=attention_gate(up2,conv2,32)
  concat2=layers.concatenate([up2,attn2],axis=-1)
  up_conv2=residual_conv_block(concat2,32)
  up1=layers.Conv3DTranspose(16,(2,2,2),strides=(2,2,2),padding='same')(up_conv2)
  attn1=attention_gate(up1,conv1,16)
  concat1=layers.concatenate([up1,attn1],axis=-1)
  up_conv1=residual_conv_block(concat1,16)
  outputs=layers.Conv3D(num_classes,(1,1,1),activation=activation,padding='same')(up_conv1)
  model=Model(inputs=inputs,outputs=outputs)
  return model