import random
import numpy as np
import tensorflow as tf
from main import SPECIAL_TOKENS,MASK_PROB

# ---  Defining create_pretraining_examples function ---
def create_pretraining_examples(corpus_sentences, tokenizer, max_seq_len, cls_id, sep_id, pad_id, mask_id, nsp_ratio=0.5):
    examples = []

    # Configure tokenizer for truncation and padding ONCE
    tokenizer.enable_truncation(max_length=max_seq_len)
    tokenizer.enable_padding(length=max_seq_len, pad_id=pad_id, pad_token=SPECIAL_TOKENS['pad_token'], pad_type_id=0)

    print(f"Generating pre-training examples (max_seq_len={max_seq_len})...")
    # Ensure there are at least two sentences to form a pair
    if len(corpus_sentences) < 2:
        print("Warning: Not enough sentences in corpus to create pre-training examples.")
        return examples

    for i in range(len(corpus_sentences) - 1):
        sentence_a = corpus_sentences[i]
        nsp_label = 0 # Default to consecutive

        if random.random() < nsp_ratio: # 50% chance for non-consecutive
            # Pick a random sentence for sentence_b that is not consecutive to sentence_a
            while True:
                j = random.randint(0, len(corpus_sentences) - 1)
                if j != i and j != i + 1: # Ensure it's not the current or next sentence
                    sentence_b = corpus_sentences[j]
                    nsp_label = 1 # Non-consecutive
                    break
        else:
            # sentence_b is the next sentence
            sentence_b = corpus_sentences[i+1]
            nsp_label = 0 # Consecutive

        # Tokenize sentence pair (truncation and padding are now handled by the tokenizer's config)
        encoded = tokenizer.encode(sentence_a, sentence_b, add_special_tokens=True)

        token_ids = np.array(encoded.ids)
        segment_ids = np.array(encoded.type_ids)
        attention_mask = np.array(encoded.attention_mask)

        examples.append({
            "input_ids": token_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "nsp_label": nsp_label,
        })

    print(f"Generated {len(examples)} raw pre-training examples.")
    return examples

# --- Defining mask_tokens function ---
def mask_tokens(input_ids, vocab_size, mask_id, cls_id, sep_id, pad_id, unk_id, mask_prob=MASK_PROB):
    labels = np.full(input_ids.shape, -100, dtype=np.int32) # -100 for ignore_index
    masked_input_ids = np.copy(input_ids)

    # Find indices of tokens that are not special tokens (CLS, SEP, PAD)
    special_tokens_mask = np.isin(input_ids, [cls_id, sep_id, pad_id])
    candidate_indices = np.where(~special_tokens_mask)[0]

    num_to_mask = int(len(candidate_indices) * mask_prob)
    if num_to_mask == 0 and len(candidate_indices) > 0:
        num_to_mask = 1 # Ensure at least one token is masked if possible

    # Randomly select tokens to mask
    masked_indices = np.random.choice(candidate_indices, num_to_mask, replace=False)

    for idx in masked_indices:
        original_token = input_ids[idx]
        labels[idx] = original_token # Store original token in labels

        rand = random.random()
        if rand < 0.8: # 80% of the time: replace with [MASK]
            masked_input_ids[idx] = mask_id
        elif rand < 0.9: # 10% of the time: replace with a random token
            masked_input_ids[idx] = random.randint(0, vocab_size - 1)
        # 10% of the time: keep original token (do nothing here)

    return masked_input_ids, labels
    
# Function to prepare each example for the dataset
def prepare_example(example):
    # Ensure NSP label is a scalar tensor
    nsp_label = tf.constant(example["nsp_labels"], dtype=tf.int32)

    inputs = {
        "input_ids": tf.constant(example["input_ids"], dtype=tf.int32),
        "segment_ids": tf.constant(example["segment_ids"], dtype=tf.int32),
        "attention_mask": tf.constant(example["attention_mask"], dtype=tf.int32),
    }
    labels = {
        "mlm_labels": tf.constant(example["mlm_labels"], dtype=tf.int32),
        "nsp_labels": nsp_label,
    }
    return inputs, labels