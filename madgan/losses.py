import tensorflow as tf
from main import feature_extractor

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
def diversity_loss(generated_images_list):
  diversity_penalty=0.0
  if len(generated_images_list)<2:
    return diversity_penalty

  extracted_features_list=[]
  for gen_images in generated_images_list:
    features=feature_extractor(gen_images,training=False)
    extracted_features_list.append(tf.reshape(features,[tf.shape(features)[0],-1]))
  for i in range(len(extracted_features_list)):
    for j in range(i+1,len(extracted_features_list)):
      pairwise_feature_distance=tf.reduce_mean(tf.square(extracted_features_list[i]-extracted_features_list[j]))
      diversity_penalty+=pairwise_feature_distance
  num_pairs=len(generated_images_list)*(len(generated_images_list)-1)/2
  if num_pairs==0:
    diversity_penalty=0.0
  else:
    diversity_penalty/=num_pairs
  return -diversity_penalty