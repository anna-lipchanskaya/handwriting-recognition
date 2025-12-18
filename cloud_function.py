import json
import numpy as np

def decode_label(num, alphabet):
    """Декодирование числовой последовательности в текст"""
    result = []
    prev_char = None
    
    for ch in num:
        if ch < len(alphabet):  # игнорируем blank символы
            current_char = alphabet[ch]
            # Убираем повторяющиеся символы (CTC decoding)
            if current_char != prev_char:
                result.append(current_char)
                prev_char = current_char
    
    return ''.join(result)

def decode_predictions(predictions, alphabet):
    """Декодирование предсказаний модели"""
    texts = []
    
    print(f"Decoding {len(predictions)} predictions with alphabet: {alphabet}")
    
    for i, prediction in enumerate(predictions):
        print(f"Prediction {i}: {prediction}")
        
        if isinstance(prediction, list):
            # Если пришли индексы символов
            if all(isinstance(x, int) for x in prediction):
                # Это уже индексы - декодируем напрямую
                text = decode_label(prediction, alphabet)
                print(f"Decoded from indices: '{text}'")
            else:
                # Это вероятности - нужно найти argmax для каждого таймстепа
                char_indices = []
                for timestep in prediction:
                    if isinstance(timestep, list) and timestep:
                        max_idx = np.argmax(timestep)
                        char_indices.append(max_idx)
                text = decode_label(char_indices, alphabet)
                print(f"Decoded from probabilities: '{text}'")
        else:
            text = ""
            print(f"Unknown prediction format: {type(prediction)}")
        
        texts.append(text)
    
    return texts

def handler(event, context):
    """
    Декодирование предсказаний нейросети
    """
    try:
        print("=== DECODE FUNCTION START ===")
        
        # Парсим входные данные
        body = event.get('body', '{}')
        
        if isinstance(body, str):
            body = json.loads(body)
        
        predictions = body.get('predictions', [])
        alphabet = body.get('alphabet', "',.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        
        print(f"Received {len(predictions)} predictions")
        print(f"Alphabet length: {len(alphabet)}")
        print(f"Alphabet: {alphabet}")
        
        if predictions:
            print(f"First prediction type: {type(predictions[0])}")
            print(f"First prediction length: {len(predictions[0]) if hasattr(predictions[0], '__len__') else 'N/A'}")
            if predictions[0]:
                print(f"First element type: {type(predictions[0][0])}")
        
        if not predictions:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'No predictions provided',
                    'received_body': body
                })
            }
        
        # Декодируем предсказания
        decoded_texts = decode_predictions(predictions, alphabet)
        
        print(f"Decoded texts: {decoded_texts}")
        print("=== DECODE FUNCTION END ===")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'texts': decoded_texts,
                'alphabet': alphabet,
                'count': len(decoded_texts),
                'debug': {
                    'predictions_count': len(predictions),
                    'alphabet_length': len(alphabet)
                }
            })
        }
        
    except Exception as e:
        import traceback
        error_msg = f'Decoding failed: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
