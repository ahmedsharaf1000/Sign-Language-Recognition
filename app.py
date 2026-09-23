from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)


# إضافة CORS يدوياً بدون مكتبة
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


# تحميل الموديل
print("Loading model...")
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# إعداد MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# القاموس
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
    31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}


@app.route('/')
def index():
    """عرض الصفحة الرئيسية"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """معالجة الصورة والتنبؤ بالحرف"""
    try:
        # استقبال البيانات
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # فك تشفير الصورة من Base64
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # تحويل لصورة numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        H, W, _ = frame.shape

        # تحويل لـ RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # معالجة الصورة بـ MediaPipe
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({
                'prediction': None,
                'confidence': 0,
                'hand_detected': False,
                'message': 'No hand detected'
            })

        # استخراج البيانات
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # التنبؤ
        if model is not None:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # حساب confidence (إذا كان الموديل يدعمه)
            confidence = 0.85  # قيمة افتراضية
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = float(np.max(proba))
            except:
                pass

            # حساب إحداثيات اليد
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            return jsonify({
                'prediction': predicted_character,
                'confidence': round(confidence * 100, 2),
                'hand_detected': True,
                'bounding_box': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """فحص حالة الخادم"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'supported_characters': list(labels_dict.values())
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """معلومات عن الموديل"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_type': str(type(model).__name__),
        'supported_characters': list(labels_dict.values()),
        'total_classes': len(labels_dict)
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting Sign Language Recognition Server")
    print("=" * 50)
    print(f"✅ Model Status: {'Loaded' if model else 'Not Loaded'}")
    print(f"✅ Supported Characters: {len(labels_dict)}")
    print(f"✅ Server URL: http://localhost:5000")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )