import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_hub
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_SEQ_LEN = 20
VOCAB_SIZE = 59
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4

char_to_idx = {
    '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, "'": 4, ',': 5, '.': 6,
    **{chr(i + 65): i + 7 for i in range(26)},
    **{chr(i + 97): i + 33 for i in range(26)}
}
idx_to_char = {v: k for k, v in char_to_idx.items()}

class CharacterErrorRate(tf.keras.metrics.Metric):

    def __init__(self, name='character_error_rate', pad_token=0, eos_token=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.error_sum = self.add_weight(name="cumulative_errors", initializer="zeros")
        self.char_count = self.add_weight(name="total_characters", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)

        y_pred_tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        mask = tf.cast(tf.not_equal(y_true, self.pad_token), tf.float32)

        batch_size = tf.shape(y_true)[0]
        seq_len = tf.shape(y_true)[1]

        eos_positions = tf.argmax(tf.cast(tf.equal(y_true, self.eos_token), tf.int32), axis=1)
        eos_positions = tf.cast(eos_positions, tf.int32)

        range_tensor = tf.range(seq_len, dtype=tf.int32)
        range_tensor = tf.expand_dims(range_tensor, 0)
        range_tensor = tf.tile(range_tensor, [batch_size, 1])

        eos_positions_expanded = tf.expand_dims(eos_positions, 1)
        before_eos_mask = tf.cast(tf.less_equal(range_tensor, eos_positions_expanded), tf.float32)

        final_mask = mask * before_eos_mask

        not_equal = tf.cast(tf.not_equal(y_true, y_pred_tokens), tf.float32)
        masked_errors = not_equal * final_mask
        batch_errors = tf.reduce_sum(masked_errors)

        chars_per_sample = tf.reduce_sum(final_mask, axis=1)

        self.error_sum.assign_add(batch_errors)
        self.char_count.assign_add(tf.reduce_sum(chars_per_sample))

    def result(self):
        return tf.math.divide_no_nan(self.error_sum, self.char_count)

    def reset_state(self):
        self.error_sum.assign(0.0)
        self.char_count.assign(0.0)

def load_data_from_folder(data_folder):
    images_folder = os.path.join(data_folder, 'images')
    annotations_file = os.path.join(data_folder, 'annotations.json')

    image_paths = []
    labels = []

    if os.path.exists(annotations_file):
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        for ann in annotations:
            img_path = os.path.join(images_folder, ann['filename'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(ann['text'])
        print(f"Загружено {len(image_paths)} образцов из {annotations_file}")
        return image_paths, labels

    if os.path.exists(images_folder):
        for filename in os.listdir(images_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')) and filename.startswith('word_'):
                if '_' in filename:
                    text_part = filename.split('_', 2)[-1]
                    text_part = text_part.rsplit('.', 1)[0]
                    if text_part:
                        image_paths.append(os.path.join(images_folder, filename))
                        labels.append(text_part)
        return image_paths, labels

    return [], []


def prepare_training_data(image_paths, labels):
    X = []
    decoder_inputs_list = []
    decoder_outputs_list = []

    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Загрузка данных"):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        processed = preprocess_image_for_training(image)
        X.append(processed)

        token_seq = [char_to_idx['<bos>']]
        for ch in label:
            token_seq.append(char_to_idx.get(ch, char_to_idx['<unk>']))
        token_seq.append(char_to_idx['<eos>'])

        if len(token_seq) > MAX_SEQ_LEN:
            token_seq = token_seq[:MAX_SEQ_LEN]
        else:
            token_seq += [char_to_idx['<pad>']] * (MAX_SEQ_LEN - len(token_seq))

        decoder_input = token_seq[:-1]
        decoder_output = token_seq[1:]

        decoder_inputs_list.append(decoder_input)
        decoder_outputs_list.append(decoder_output)

    X = np.array(X, dtype=np.float32)
    decoder_inputs = np.array(decoder_inputs_list, dtype=np.int32)
    decoder_outputs = np.array(decoder_outputs_list, dtype=np.int32)

    return X, decoder_inputs, decoder_outputs

def adaptive_threshold(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 21, 10)
    image = cv2.bitwise_not(image)
    return image


def normalize(image):
    return image.astype(np.float32) / 255.0


def resize_and_reshape(image, target_size=(64, 200)):
    h, w = image.shape
    if h > target_size[0] or w > target_size[1]:
        shrink = min(target_size[0] / h, target_size[1] / w)
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)

    pad_h = target_size[0] - image.shape[0]
    pad_w = target_size[1] - image.shape[1]
    image = cv2.copyMakeBorder(
        image,
        math.ceil(pad_h / 2), math.floor(pad_h / 2),
        math.ceil(pad_w / 2), math.floor(pad_w / 2),
        cv2.BORDER_CONSTANT, value=0
    )
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def preprocess_image_for_training(image):
    binary = adaptive_threshold(image)
    resized = resize_and_reshape(binary, target_size=(64, 200))
    normalized = normalize(resized)
    normalized = normalized[..., np.newaxis]
    return normalized

def load_pretrained_transformer_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            'TokenAndPositionEmbedding': keras_hub.layers.TokenAndPositionEmbedding,
            'TransformerDecoder': keras_hub.layers.TransformerDecoder
        }
    )
    print(f"Предобученная модель загружена из {model_path}")
    return model

def fine_tune_model(data_folder, pretrained_model_path, output_model_path='fine_tuned_transformer.keras'):

    image_paths, labels = load_data_from_folder(data_folder)
    X_images, decoder_inputs, decoder_outputs = prepare_training_data(image_paths, labels)

    X_train, X_val, dec_in_train, dec_in_val, dec_out_train, dec_out_val = train_test_split(
        X_images, decoder_inputs, decoder_outputs,
        test_size=0.2, random_state=42
    )

    model = load_pretrained_transformer_model(pretrained_model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # ← from_logits=True
        metrics=['accuracy', CharacterErrorRate(pad_token=0, eos_token=2)]  # ← CER метрика
    )
    model.summary()

    history = model.fit(
        [X_train, dec_in_train],
        dec_out_train,
        validation_data=([X_val, dec_in_val], dec_out_val),
        epochs=30,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            keras.callbacks.ModelCheckpoint(output_model_path.replace('.keras', '_best.keras'),
                                            save_best_only=True, verbose=1)
        ]
    )
    model.save(output_model_path)
    return True

if __name__ == "__main__":
    DATA_FOLDER = "training_data"
    PRETRAINED_MODEL_PATH = "handwritten_decode_vgg6_last2.keras"
    OUTPUT_MODEL_PATH = "../fine_tuned_transformer_my_handwriting.keras"

    success = fine_tune_model(DATA_FOLDER, PRETRAINED_MODEL_PATH, OUTPUT_MODEL_PATH)

    if success:
        print(f"TRANSFORMER_MODEL_PATH = '{OUTPUT_MODEL_PATH}'")
