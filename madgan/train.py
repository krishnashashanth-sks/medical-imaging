from tqdm.auto import tqdm
import tensorflow as tf

@tf.function
def train_step(images,generators,discriminator,generator_loss,discriminator_loss,diversity_loss,generator_optimizers,discriminator_optimizer,batch_size,latent_dim,num_generators,lambda_diversity):
  noise_vectors=[tf.random.normal([batch_size,latent_dim])for _ in range(num_generators)]
  current_batch_size=tf.shape(images)[0]
  with tf.GradientTape() as disc_tape:
    generated_images_list=[]
    for i in range(num_generators):
      gen_images=generators[i](noise_vectors[i],training=True)
      generated_images_list.append(gen_images)
    all_fake_images=tf.concat(generated_images_list,axis=0)
    real_output=discriminator(images,training=False)
    fake_output=discriminator(all_fake_images,training=False)
    disc_loss=discriminator_loss(real_output,fake_output)
  gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))

  total_gen_loss=0.0
  # Calculate diversity_score once per step, outside the individual generator loops
  # and outside the discriminator's tape, using `training=False` for feature extraction from D.
  diversity_score = diversity_loss(generated_images_list)

  for i in range(num_generators):
    with tf.GradientTape() as gen_tape:
      gen_images_current=generators[i](noise_vectors[i],training=True)
      fake_output_current=discriminator(gen_images_current,training=True)
      gen_adversarial_loss=generator_loss(fake_output_current)
      current_gen_total_loss=gen_adversarial_loss+(lambda_diversity*diversity_score) # diversity_score is shared across generators

    gradients_of_generator=gen_tape.gradient(current_gen_total_loss,generators[i].trainable_variables)
    generator_optimizers[i].apply_gradients(zip(gradients_of_generator,generators[i].trainable_variables))
    total_gen_loss+=current_gen_total_loss
  return disc_loss,total_gen_loss/num_generators,diversity_score
  
def train(dataset, epochs,generators,discriminator,generator_loss,discriminator_loss,diversity_loss,generator_optimizers,discriminator_optimizer,batch_size,latent_dim,num_generators,lambda_diversity):
    for epoch in tqdm(range(epochs)):
        start = tf.timestamp()

        for image_batch in tqdm(dataset):
            d_loss, g_loss, div_loss = train_step(image_batch,generators,discriminator,generator_loss,discriminator_loss,diversity_loss,generator_optimizers,discriminator_optimizer,batch_size,latent_dim,num_generators,lambda_diversity)

        # Print progress
        end = tf.timestamp()
        print(f'Epoch {epoch + 1}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, Div Loss: {div_loss:.4f}, Time: {end - start:.2f}s')

    print("\nTraining complete.")


