from medmnist.dataset import PathMNIST
import tensorflow as tf
import numpy as np

def load_medmnist_pathmnist(root_dir='./data', resize_dim=(32,32)):
    """
    Loads and preprocesses the PathMNIST dataset from MedMNIST.
    """
    print(f"Loading PathMNIST dataset from MedMNIST...")

    # Download and load the PathMNIST dataset
    # The 'download=True' argument will download the dataset if not already present.
    train_dataset = PathMNIST(split='train', root=root_dir, download=True)
    test_dataset = PathMNIST(split='test', root=root_dir, download=True)
    val_dataset = PathMNIST(split='val', root=root_dir, download=True)

    # Concatenate all splits for GAN training as we don't need distinct train/test sets for unsupervised generation
    all_images = np.concatenate([train_dataset.imgs, test_dataset.imgs, val_dataset.imgs], axis=0)

    # Convert to float32 and normalize to [-1, 1]
    # MedMNIST images are typically 28x28x3 and uint8 [0, 255]
    images = all_images.astype('float32')
    images = (images / 127.5) - 1 # Normalize to [-1, 1]

    # Resize images if necessary
    if images.shape[1] != resize_dim[0] or images.shape[2] != resize_dim[1]:
        print(f"Resizing images from {images.shape[1]}x{images.shape[2]} to {resize_dim[0]}x{resize_dim[1]}")
        images_resized = tf.image.resize(images, resize_dim, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
        images = images_resized

    print(f"Loaded and preprocessed PathMNIST dataset shape: {images.shape}")
    return images