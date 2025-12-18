import math
import tensorflow as tf
from tensorflow.keras.backend import get_value, ctc_decode
from tensorflow.keras import backend as K
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import requests
import json
import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime
import uuid
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def log_message(message):
    print(message, flush=True)
    sys.stdout.flush()


# Настройки Yandex Object Storage
S3_ENDPOINT = "https://storage.yandexcloud.net"
BUCKET_NAME = "handwriting-results"
S3_FOLDER = "recognized-texts"

# Настройка S3 клиента с использованием IAM-роли сервисного аккаунта
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        region_name='ru-central1'
    )

    # Проверяем что бакет существует и доступен
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        STORAGE_ENABLED = True
        log_message("Object Storage настроен (сервисный аккаунт)")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            log_message(f"Бакет {BUCKET_NAME} не существует")
        elif error_code == '403':
            log_message(f"Нет доступа к бакету {BUCKET_NAME}. Проверьте права сервисного аккаунта")
        else:
            log_message(f"Ошибка доступа к бакету: {error_code}")
        STORAGE_ENABLED = False

except Exception as e:
    log_message(f"Object Storage не настроен: {e}")
    STORAGE_ENABLED = False

API_GATEWAY_URL = "https://d5dkn2rbfi1k6ga0rfj6.3zvepvee.apigw.yandexcloud.net"
ALPHABET = "',.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def save_to_storage(recognized_text, filename, processing_time, used_api_gateway):
    """
    Сохраняет только распознанный текст в Object Storage
    """
    if not STORAGE_ENABLED:
        return None

    try:
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        record_data = {
            "id": record_id,
            "timestamp": timestamp,
            "original_filename": filename,
            "recognized_text": recognized_text,
            "processing_time_seconds": processing_time,
            "used_api_gateway": used_api_gateway,
            "alphabet": ALPHABET,
            "model_version": "handwritten_last2.keras"
        }

        text_key = f"{S3_FOLDER}/{record_id}.json"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=text_key,
            Body=json.dumps(record_data, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )

        log_message(f"Текст сохранен в Object Storage: {text_key}")
        log_message(f"Распознано: {recognized_text}")

        return record_data

    except Exception as e:
        log_message(f"Ошибка сохранения в Object Storage: {e}")
        return None


def call_decode_api_gateway(predictions, alphabet):
    """
    Вызов функции декодирования через API Gateway
    """
    payload = {
        'predictions': predictions,
        'alphabet': alphabet
    }

    log_message(f"[API Gateway] Отправка запроса к {API_GATEWAY_URL}/decode")
    log_message(f"[API Gateway] Размер данных: {len(predictions)} предсказаний")

    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/decode",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        log_message(f"[API Gateway] Получен ответ: статус {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            texts = result.get('texts', [])
            log_message(f"[API Gateway] Успешный ответ. Распознанные тексты: {texts}")
            return texts
        else:
            log_message(f"[API Gateway] Ошибка HTTP {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        log_message("[API Gateway] Таймаут при запросе (10 секунд)")
        return None
    except requests.exceptions.ConnectionError:
        log_message("[API Gateway] Ошибка подключения - API Gateway недоступен")
        return None
    except requests.exceptions.RequestException as e:
        log_message(f"[API Gateway] Ошибка запроса: {e}")
        return None
    except Exception as e:
        log_message(f"[API Gateway] Неожиданная ошибка: {e}")
        return None


def decode_predictions(predictions):
    """
    Основная функция декодирования через API Gateway
    """
    log_message("Начало декодирования предсказаний...")

    def prepare_predictions_for_api(predictions):
        if len(predictions.shape) == 3:
            pred_indices = []
            for batch_item in predictions:
                item_indices = []
                for timestep in batch_item:
                    max_idx = np.argmax(timestep)
                    if max_idx < len(ALPHABET):
                        item_indices.append(int(max_idx))
                pred_indices.append(item_indices)
        else:
            pred_indices = predictions.tolist()
        return pred_indices

    pred_indices = prepare_predictions_for_api(predictions)

    log_message("Попытка декодирования через API Gateway...")
    decoded_texts = call_decode_api_gateway(pred_indices, ALPHABET)
    used_api_gateway = True
    failure_reason = None

    if decoded_texts is None:
        used_api_gateway = False
        failure_reason = "API Gateway недоступен или вернул ошибку"
        log_message(f"API Gateway недоступен. Причина: {failure_reason}")
    else:
        log_message(f"Использовано API Gateway: {used_api_gateway}")

    return decoded_texts, used_api_gateway, failure_reason


# Функции обработки изображений
def adaptive_threshold(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    image = cv2.bitwise_not(image)
    return image


def normalize(image):
    return image.astype(np.float32) / 255.0


def resize_and_reshape(imaдаваge, target_size=(64, 200)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    if h > target_size[0] or w > target_size[1]:
        shrink_multiplier = min(target_size[0] / h, target_size[1] / w)
        image = cv2.resize(image, None, fx=shrink_multiplier, fy=shrink_multiplier, interpolation=cv2.INTER_AREA)

    pad_height = target_size[0] - image.shape[0]
    pad_width = target_size[1] - image.shape[1]
    top, bottom = math.ceil(pad_height / 2), math.floor(pad_height / 2)
    left, right = math.ceil(pad_width / 2), math.floor(pad_width / 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def preprocess_image(image):
    image = resize_and_reshape(image)
    image = adaptive_threshold(image)
    image = normalize(image)
    return image


# Метрики и загрузка модели
class CharacterErrorRateCalculator(tf.keras.metrics.Metric):
    def __init__(self, name='character_error_score', **kwargs):
        super(CharacterErrorRateCalculator, self).__init__(name=name, **kwargs)
        self.error_sum = self.add_weight(name="cumulative_errors", initializer="zeros")
        self.num_samples = self.add_weight(name="sample_counter", initializer="zeros")

    def update_state(self, ground_truth, predictions, sample_weight=None):
        pred_shape = K.shape(predictions)
        seq_len = tf.ones(shape=pred_shape[0]) * K.cast(pred_shape[1], 'float32')
        decoded_output, _ = K.ctc_decode(predictions, seq_len, greedy=True)
        sparse_decoded = K.ctc_label_dense_to_sparse(decoded_output[0], K.cast(seq_len, 'int32'))
        sparse_truth = K.ctc_label_dense_to_sparse(ground_truth, K.cast(seq_len, 'int32'))
        sparse_truth = tf.sparse.retain(sparse_truth,
                                        tf.not_equal(sparse_truth.values, tf.math.reduce_max(sparse_truth.values)))
        sparse_decoded = tf.sparse.retain(sparse_decoded, tf.not_equal(sparse_decoded.values, -1))
        error_rate = tf.edit_distance(sparse_decoded, sparse_truth, normalize=True)
        self.error_sum.assign_add(tf.reduce_sum(error_rate))
        self.num_samples.assign_add(K.cast(ground_truth.shape[1], 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.error_sum, self.num_samples)

    def reset_state(self):
        self.error_sum.assign(0.0)
        self.num_samples.assign(0.0)


def CTCLoss(targets, outputs):
    batch_size = tf.cast(tf.shape(targets)[0], dtype="int64")
    output_seq_length = tf.cast(tf.shape(outputs)[1], dtype="int64")
    target_seq_length = tf.cast(tf.shape(targets)[1], dtype="int64")
    output_seq_length = output_seq_length * tf.ones(shape=(batch_size, 1), dtype="int64")
    target_seq_length = target_seq_length * tf.ones(shape=(batch_size, 1), dtype="int64")
    calculated_loss = K.ctc_batch_cost(targets, outputs, output_seq_length, target_seq_length)
    return calculated_loss


# Загрузка модели
MODEL_LOADED = False
model = None

try:
    model = load_model('handwritten_last2.keras',
                       custom_objects={
                           'CTCLoss': CTCLoss,
                           'CharacterErrorRateCalculator': CharacterErrorRateCalculator
                       },
                       compile=False)
    MODEL_LOADED = True
    model.compile(optimizer='adam',
                  loss=CTCLoss,
                  metrics=[CharacterErrorRateCalculator()])
    log_message("Модель успешно загружена")
except Exception as e:
    MODEL_LOADED = False
    log_message(f"Ошибка загрузки модели: {e}")


@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED, storage_enabled=STORAGE_ENABLED)


@app.route('/recognize', methods=['POST'])
def recognize():
    start_time = datetime.utcnow()
    log_message("\n" + "=" * 60)
    log_message("НОВЫЙ ЗАПРОС НА РАСПОЗНАВАНИЕ")
    log_message("=" * 60)

    try:
        if not MODEL_LOADED:
            log_message("ОШИБКА: Модель не загружена!")
            return jsonify({'success': False, 'error': 'Модель не загружена'})

        if 'image' not in request.files:
            log_message("ОШИБКА: Файл не выбран в запросе")
            return jsonify({'success': False, 'error': 'Файл не выбран'})

        file = request.files['image']
        if file.filename == '':
            log_message("ОШИБКА: Имя файла пустое")
            return jsonify({'success': False, 'error': 'Файл не выбран'})

        log_message(f"Загрузка файла: {file.filename}")

        image_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            log_message("ОШИБКА: Не удалось загрузить изображение")
            return jsonify({'success': False, 'error': 'Не удалось загрузить изображение'})

        log_message("Предобработка изображения...")
        processed_image = preprocess_image(image)
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        log_message("Прогноз модели...")
        predicts = model.predict(processed_image_batch, verbose=0)

        log_message("Декодирование предсказаний...")
        predicted_texts, used_api_gateway, failure_reason = decode_predictions(predicts)

        recognized_text = predicted_texts[0] if predicted_texts else ""
        log_message(f"Финальный распознанный текст: '{recognized_text}'")

        processing_time = round((datetime.utcnow() - start_time).total_seconds(), 3)

        if recognized_text and STORAGE_ENABLED:
            log_message("Сохранение в Object Storage...")
            storage_result = save_to_storage(
                recognized_text=recognized_text,
                filename=file.filename,
                processing_time=processing_time,
                used_api_gateway=used_api_gateway
            )
            record_id = storage_result['id'] if storage_result else None
        else:
            record_id = None

        log_message(f"Распознавание завершено за {processing_time} секунд")
        log_message(f"Использован API Gateway: {used_api_gateway}")
        if failure_reason:
            log_message(f"Причина использования локального декодирования: {failure_reason}")
        log_message("=" * 60)

        return jsonify({
            'success': True,
            'recognized_text': recognized_text,
            'used_api_gateway': used_api_gateway,
            'failure_reason': failure_reason,
            'processing_time': processing_time,
            'saved_to_storage': STORAGE_ENABLED and record_id is not None,
            'record_id': record_id,
            'storage_enabled': STORAGE_ENABLED
        })

    except Exception as e:
        error_time = (datetime.utcnow() - start_time).total_seconds()
        log_message(f"КРИТИЧЕСКАЯ ОШИБКА в /recognize: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    log_message(f"Запуск приложения с API Gateway: {API_GATEWAY_URL}")
    log_message(f"Алфавит: {ALPHABET}")
    log_message(f"Object Storage: {'Включен' if STORAGE_ENABLED else 'Отключен'}")
    app.run(debug=False, host='0.0.0.0', port=5000)
