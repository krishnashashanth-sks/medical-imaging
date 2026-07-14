import numpy as np
from main import MAX_DRUG_LEN,MAX_TARGET_LEN,drug_chars,target_chars

drug_char_to_int = {char: i for i, char in enumerate(drug_chars)}

target_char_to_int = {char: i for i, char in enumerate(target_chars)}

#  HELPER FUNCTIONS
def one_hot_encode(sequence, char_to_int, max_len):
    encoding = np.zeros((max_len, len(char_to_int)))
    for i, char in enumerate(sequence[:max_len]):
        if char in char_to_int:
            encoding[i, char_to_int[char]] = 1
    return encoding

def preprocess_data(smiles, sequences, affinities):
    print("Encoding data... this may take a minute.")
    drugs_encoded = np.array([one_hot_encode(s, drug_char_to_int, MAX_DRUG_LEN) for s in smiles])
    targets_encoded = np.array([one_hot_encode(t, target_char_to_int, MAX_TARGET_LEN) for t in sequences])
    return drugs_encoded, targets_encoded, np.array(affinities)
