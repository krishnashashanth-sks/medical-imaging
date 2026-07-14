from tensorflow import keras
from tensorflow.keras import layers

def create_deepdta_model(max_drug_len,drug_vocab_size,max_target_len,target_vocab_size,
                         conv_filters=32,kernel_size_drug=8,kernel_size_target=8,
                         pool_size_drug=8,pool_size_target=8,dense_units=1024,
                         dropout_rate=0.1):
  drug_input=keras.Input(shape=(max_drug_len,drug_vocab_size),name='drug_input')
  drug_conv=layers.Conv1D(filters=conv_filters,kernel_size=kernel_size_drug,activation='relu',padding='valid')(drug_input)
  drug_pool=layers.GlobalMaxPooling1D()(drug_conv)
  target_input=keras.Input(shape=(max_target_len,target_vocab_size),name='target_input')
  target_conv=layers.Conv1D(filters=conv_filters,kernel_size=kernel_size_target,activation='relu',padding='valid')(target_input)
  target_pool=layers.GlobalMaxPooling1D()(target_conv)
  merged_features=layers.concatenate([drug_pool,target_pool])
  dense_1=layers.Dense(dense_units,activation='relu')(merged_features)
  dropout_1=layers.Dropout(dropout_rate)(dense_1)
  dense_2=layers.Dense(dense_units//2,activation='relu')(dropout_1)
  dropout_2=layers.Dropout(dropout_rate)(dense_2)
  output=layers.Dense(1,activation='linear')(dropout_2)
  model=keras.Model(inputs=[drug_input,target_input],outputs=output)
  return model