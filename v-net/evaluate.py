from sklearn.metrics import jaccard_score
import numpy as np 
import tensorflow.keras.backend as K 
from losses import dice_coef

def evaluate_model(model, test_images, test_masks, threshold=0.5):
    predictions = model.predict(np.expand_dims(test_images, -1))
    binary_predictions = (predictions > threshold).astype(np.uint8)

    dice_scores = []
    jaccard_scores = []

    for i in range(len(test_images)):
        # Dice coefficient for each sample
        # Need to ensure dimensions match for dice_coef. Squeeze predictions if channel dim is kept.
        current_true = test_masks[i].flatten()
        current_pred = binary_predictions[i].flatten()

        # Ensure we don't divide by zero if both are completely empty/full
        if np.sum(current_true) == 0 and np.sum(current_pred) == 0:
            dice = 1.0 # Perfect score if both are empty
        elif np.sum(current_true) + np.sum(current_pred) == 0:
            dice = 0.0 # Avoid NaN if both are empty but the function doesn't handle it
        else:
            # Using K.constant to wrap numpy arrays for Keras backend operations
            dice = K.eval(dice_coef(K.constant(test_masks[i], dtype='float32'), K.constant(binary_predictions[i], dtype='float32')))
        dice_scores.append(dice)

        # Jaccard index (IoU) for each sample
        # Scikit-learn's jaccard_score expects flattened arrays.
        if np.sum(current_true) == 0 and np.sum(current_pred) == 0:
            jaccard = 1.0 # Perfect score if both are empty
        else:
            jaccard = jaccard_score(current_true, current_pred)
        jaccard_scores.append(jaccard)

    print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Average Jaccard Index: {np.mean(jaccard_scores):.4f}")
