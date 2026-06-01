import math
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.backend import get_value, ctc_decode
from tensorflow.keras import backend as K
from keras.models import load_model
import keras_hub
from tkinter import ttk
import threading
import easyocr
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

VGG_MODEL_PATH = "handwritten_VGG16_last.keras"
TRANSFORMER_MODEL_PATH = "handwritten_decode_vgg6_last2.keras"
LAST2_MODEL_PATH = "CTC_handwritten.keras"
BASIC_MODEL_PATH = "handwritten_basic.keras"

alphabet = "',.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

print("Инициализация EasyOCR детектора...")
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR успешно инициализирован")
except Exception as e:
    print(e)
    reader = None

def decode_label(num, alphabet, pad_token=None):
    if pad_token is None:
        pad_token = len(alphabet)
    return ''.join(alphabet[ch] for ch in num if ch != pad_token)


def decode_predictions_ctc(preds, alphabet):
    input_length = np.ones(preds.shape[0]) * preds.shape[1]
    values = get_value(ctc_decode(preds, input_length=input_length, greedy=True)[0][0])
    texts = [decode_label(value[value >= 0], alphabet) for value in values]
    return texts

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


def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    binary = adaptive_threshold(gray)
    resized = resize_and_reshape(binary)
    normalized = normalize(resized)

    normalized = normalized[..., np.newaxis]
    normalized = normalized[np.newaxis, ...]
    return normalized

def detect_text_with_easyocr(image, status_callback=None):
    if reader is None:
        if status_callback:
            status_callback("EasyOCR не инициализирован")
        return [], [], []

    try:
        if status_callback:
            status_callback("Детекция текста на изображении...")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(image_rgb, paragraph=False)

        words = []
        boxes = []
        confidences = []

        for (bbox, text, confidence) in results:
            points = np.array(bbox, dtype=np.int32)
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)

            padding = 5
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(image.shape[1], x2 + padding)
            y2p = min(image.shape[0], y2 + padding)

            word_img = image[y1p:y2p, x1p:x2p]

            if word_img.size > 0 and word_img.shape[0] > 10 and word_img.shape[1] > 10:
                words.append(word_img)
                boxes.append((x1p, y1p, x2p, y2p))
                confidences.append(confidence)

        if status_callback:
            status_callback(f"Найдено {len(words)} слов")

        if len(words) == 0:
            if status_callback:
                status_callback("Повторная детекция...")
            results = reader.readtext(image_rgb, paragraph=False, width_ths=0.7, height_ths=0.7)

            for (bbox, text, confidence) in results:
                points = np.array(bbox, dtype=np.int32)
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x1, x2 = np.min(x_coords), np.max(x_coords)
                y1, y2 = np.min(y_coords), np.max(y_coords)

                padding = 5
                x1p = max(0, x1 - padding)
                y1p = max(0, y1 - padding)
                x2p = min(image.shape[1], x2 + padding)
                y2p = min(image.shape[0], y2 + padding)

                word_img = image[y1p:y2p, x1p:x2p]

                if word_img.size > 0 and word_img.shape[0] > 5 and word_img.shape[1] > 5:
                    words.append(word_img)
                    boxes.append((x1p, y1p, x2p, y2p))
                    confidences.append(confidence)

        if boxes:
            boxes_with_indices = list(enumerate(boxes))
            boxes_with_indices.sort(key=lambda x: x[1][1])
            heights = [box[3] - box[1] for box in boxes]
            median_height = np.median(heights) if heights else 50
            line_threshold = median_height * 0.7

            lines = []
            current_line = []
            current_y = boxes_with_indices[0][1][1] if boxes_with_indices else 0

            for idx, box in boxes_with_indices:
                if abs(box[1] - current_y) > line_threshold:
                    current_line.sort(key=lambda x: x[1][0])
                    lines.append(current_line)
                    current_line = [(idx, box)]
                    current_y = box[1]
                else:
                    current_line.append((idx, box))

            if current_line:
                current_line.sort(key=lambda x: x[1][0])
                lines.append(current_line)

            final_words = []
            final_boxes = []
            line_ids = []

            for line_idx, line in enumerate(lines):
                for idx, box in line:
                    final_words.append(words[idx])
                    final_boxes.append(box)
                    line_ids.append(line_idx)

            if status_callback:
                status_callback(f"Найдено {len(final_words)} слов, {line_idx + 1} строк")

            return final_words, final_boxes, line_ids

        return words, boxes, [0] * len(words)

    except Exception as e:
        if status_callback:
            status_callback(f"Ошибка детекции: {e}")
        return [], [], []


def draw_easyocr_segmentation(image, boxes, line_ids):
    vis = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]

    for i, ((x1, y1, x2, y2), line_id) in enumerate(zip(boxes, line_ids)):
        color = colors[line_id % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, str(i + 1), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis

char_to_idx = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, "'": 4, ',': 5, '.': 6,
               **{chr(i + 65): i + 7 for i in range(26)},
               **{chr(i + 97): i + 33 for i in range(26)}}
idx_to_char = {v: k for k, v in char_to_idx.items()}


def greedy_decode_single_image(model, image, max_len=19):
    bos = char_to_idx['<bos>']
    eos = char_to_idx['<eos>']
    pad = char_to_idx['<pad>']

    decoder_input = [bos] + [pad] * (max_len - 1)
    decoder_input = tf.constant([decoder_input], dtype=tf.int32)

    next_pos = 1

    for _ in range(max_len - 1):
        logits = model([image, decoder_input], training=False)

        if len(logits.shape) == 4:
            logits = tf.squeeze(logits, axis=2)

        next_token_logits = logits[:, next_pos - 1, :]
        next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)

        decoder_input = tf.tensor_scatter_nd_update(
            decoder_input,
            indices=[[0, next_pos]],
            updates=next_token
        )

        if next_token.numpy()[0] == eos:
            break

        next_pos += 1

    tokens = decoder_input.numpy()[0]
    text = ''
    for t in tokens:
        if t in [pad, bos]:
            continue
        if t == eos:
            break
        text += idx_to_char.get(int(t), '')

    return text

print("Загрузка моделей...")

transformer_model = None
vgg_model = None
last2_model = None
basic_model = None

try:
    transformer_model = tf.keras.models.load_model(
        TRANSFORMER_MODEL_PATH,
        compile=False,
        custom_objects={
            'TokenAndPositionEmbedding': keras_hub.layers.TokenAndPositionEmbedding,
            'TransformerDecoder': keras_hub.layers.TransformerDecoder
        }
    )
    print("Transformer 4-я версия загружена")
except Exception as e:
    print(f"Ошибка загрузки Transformer модели: {e}")

try:
    vgg_model = load_model(VGG_MODEL_PATH, compile=False)
    print("VGG модель загружена")
except Exception as e:
    print(f"Ошибка загрузки VGG модели: {e}")

try:
    last2_model = load_model(LAST2_MODEL_PATH, compile=False)
    print("Last2 модель загружена")
except Exception as e:
    print(f"Ошибка загрузки Last2 модели: {e}")

try:
    basic_model = load_model(BASIC_MODEL_PATH, compile=False)
    print("Базовая модель загружена")
except Exception as e:
    print(f"Ошибка загрузки базовой модели: {e}")

def recognize_full_text(image, status_callback=None):
    if status_callback:
        status_callback("Начало распознавания...")

    word_images, boxes, line_ids = detect_text_with_easyocr(image, status_callback)

    if len(word_images) == 0:
        if status_callback:
            status_callback("Текст не найден")
        return "Текст не найден", "Текст не найден", "Текст не найден", "Текст не найден", image

    transformer_result = []
    vgg_result = []
    last2_result = []
    basic_result = []

    total = len(word_images)

    for i, word in enumerate(word_images):
        if status_callback:
            status_callback(f"Обработка слова {i + 1}/{total}")
        if hasattr(recognize_full_text, 'set_current_word'):
            recognize_full_text.set_current_word(i + 1, total)

        try:
            processed = preprocess_image(word)
            processed_tensor = tf.constant(processed, dtype=tf.float32)
            if transformer_model is not None:
                t_text = greedy_decode_single_image(transformer_model, processed_tensor, max_len=19)
                transformer_result.append(t_text)
            else:
                transformer_result.append("[модель не загружена]")
            if vgg_model is not None:
                vgg_pred = vgg_model.predict(processed, verbose=0)
                vgg_result.append(decode_predictions_ctc(vgg_pred, alphabet)[0])
            else:
                vgg_result.append("[модель не загружена]")
            if last2_model is not None:
                last2_pred = last2_model.predict(processed, verbose=0)
                last2_result.append(decode_predictions_ctc(last2_pred, alphabet)[0])
            else:
                last2_result.append("[модель не загружена]")
            if basic_model is not None:
                basic_pred = basic_model.predict(processed, verbose=0)
                basic_result.append(decode_predictions_ctc(basic_pred, alphabet)[0])
            else:
                basic_result.append("[модель не загружена]")

        except Exception as e:
            print(f"Ошибка при обработке слова {i + 1}: {e}")
            transformer_result.append("?")
            vgg_result.append("?")
            last2_result.append("?")
            basic_result.append("?")

    if status_callback:
        status_callback("Распознавание завершено")

    segmentation_image = draw_easyocr_segmentation(image, boxes, line_ids)

    return (
        " ".join(transformer_result),
        " ".join(vgg_result),
        " ".join(last2_result),
        " ".join(basic_result),
        segmentation_image
    )

class LoadingWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.geometry("400x150")
        self.window.title("Распознавание")
        self.window.grab_set()
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - 200
        y = (self.window.winfo_screenheight() // 2) - 75
        self.window.geometry(f'400x150+{x}+{y}')

        tk.Label(
            self.window,
            text="Идет распознавание текста",
            font=("Helvetica", 12)
        ).pack(pady=20)

        status_frame = tk.Frame(self.window, bg="white", relief=tk.SUNKEN, bd=1)
        status_frame.pack(pady=20, padx=20, fill=tk.X)
        self.status_label = tk.Label(
            status_frame,
            text="Подготовка...",
            font=("Courier", 10),
            bg="white",
            wraplength=350,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=10, padx=10)
        self.base_message = "Подготовка..."
        self.dot_count = 0
        self.animating = True
        self.animate_dots()

    def animate_dots(self):
        if hasattr(self, 'window') and self.window.winfo_exists() and self.animating:
            self.dot_count = (self.dot_count % 3) + 1
            dots = "." * self.dot_count
            self.status_label.config(text=f"{self.base_message}{dots}")
            self.window.after(500, self.animate_dots)

    def update_status(self, message):
        if hasattr(self, 'window') and self.window.winfo_exists():
            self.base_message = message
            self.dot_count = 0
            self.status_label.config(text=f"{message}.")
            self.window.update_idletasks()

    def destroy(self):
        self.animating = False
        if hasattr(self, 'window') and self.window.winfo_exists():
            self.window.destroy()


def recognize_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
        return

    loading = LoadingWindow(root)

    def run_recognition():
        try:
            def update_status(msg):
                loading.update_status(msg)

            t_text, vgg_text, last2_text, basic_text, seg_img = recognize_full_text(image, update_status)

            loading.destroy()

            result_window = tk.Toplevel(root)
            result_window.geometry("1200x800")
            result_window.title("Результаты распознавания (4 модели)")

            result_window.update_idletasks()
            x = (result_window.winfo_screenwidth() // 2) - 600
            y = (result_window.winfo_screenheight() // 2) - 400
            result_window.geometry(f'1200x800+{x}+{y}')

            main_frame = tk.Frame(result_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            image_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]

            max_display_width = main_frame.winfo_width() - 40
            if max_display_width <= 0:
                max_display_width = 1000

            if width > max_display_width:
                scale = max_display_width / width
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (max_display_width, new_height))

            img_show = Image.fromarray(image_rgb)
            img_tk = ImageTk.PhotoImage(img_show)

            img_frame = tk.Frame(main_frame)
            img_frame.pack(pady=10, fill=tk.X)

            label_img = tk.Label(img_frame, image=img_tk)
            label_img.image = img_tk
            label_img.pack()

            results_frame = tk.Frame(main_frame)
            results_frame.pack(pady=10, fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(results_frame)
            scrollbar_vert = tk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_vert.set)

            labels = [
                ("Transformer:", t_text, "darkorange"),
                ("VGG16 + CTC:", vgg_text, "blue"),
                ("CTC (handwritten):", last2_text, "purple"),
                ("Базовая модель:", basic_text, "teal")
            ]

            for title, result_text, color in labels:
                tk.Label(scrollable_frame,
                         text=title,
                         font=("Helvetica", 14, "bold"),
                         fg=color).pack(anchor="w", pady=(10, 5))

                text_widget = tk.Text(scrollable_frame, height=4, wrap=tk.WORD,
                                      font=("Helvetica", 11), width=80)
                text_widget.pack(fill=tk.X, pady=(0, 10))
                text_widget.insert("1.0", result_text if result_text else "[пусто]")
                text_widget.config(state=tk.DISABLED)

            def save_results():
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")]
                )
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ (4 модели)\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(f"Transformer: {t_text}\n\n")
                        f.write(f"VGG16 + CTC: {vgg_text}\n\n")
                        f.write(f"CTC: {last2_text}\n\n")
                        f.write(f"Базовая модель: {basic_text}\n")
                        f.write("=" * 60 + "\n")
                    messagebox.showinfo("Успех", "Результаты сохранены!")

            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(pady=20)

            tk.Button(button_frame, text="Сохранить результаты",
                      command=save_results, width=20, height=2,
                      bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)

            tk.Button(button_frame, text="Закрыть",
                      command=result_window.destroy, width=20, height=2,
                      bg="#f44336", fg="white").pack(side=tk.LEFT, padx=10)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar_vert.pack(side="right", fill="y")

        except Exception as e:
            loading.destroy()
            messagebox.showerror("Ошибка", f"Произошла ошибка при распознавании:\n{str(e)}")

    threading.Thread(target=run_recognition).start()

root = tk.Tk()
root.geometry("900x600")
root.title("OCR System - 4 модели распознавания")

root.update_idletasks()
width = 900
height = 600
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f'{width}x{height}+{x}+{y}')

tk.Label(root, text="Распознавание рукописного текста",
         font=("Helvetica", 20, "bold")).pack(pady=40)

tk.Label(root, text="(используется EasyOCR для детекции + 4 нейросетевых модели)",
         font=("Helvetica", 11), fg="gray").pack()

if reader is None:
    tk.Label(root, text="EasyOCR не инициализирован!",
             font=("Helvetica", 10), fg="red").pack(pady=5)
else:
    tk.Label(root, text="EasyOCR готов к работе",
             font=("Helvetica", 10), fg="green").pack(pady=5)

tk.Button(root, text="Выбрать изображение",
          command=recognize_image,
          width=25, height=2,
          bg="#4CAF50", fg="white",
          font=("Helvetica", 12, "bold")).pack(pady=40)

tk.Button(root, text="Выход",
          command=root.quit,
          width=20, height=1,
          bg="#f44336", fg="white",
          font=("Helvetica", 10)).pack(pady=10)

info_text = """Инструкция:
1. Нажмите 'Выбрать изображение'
2. Выберите изображение с рукописным текстом
3. Дождитесь результатов от 4 моделей

Модели:
- Transformer - современная архитектура
- VGG16 + CTC - сеть с предобученными слоями VGG16 с CTC-декодером
- CTC - сеть с СTC-декодером
- Базовая - упрощенная модель"""

tk.Label(root, text=info_text, font=("Helvetica", 9),
         justify=tk.LEFT, fg="gray").pack(side=tk.BOTTOM, pady=20)

root.mainloop()
