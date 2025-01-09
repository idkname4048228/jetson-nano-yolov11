from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import logging

# 設置 Flask 伺服器
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 加載 YOLO 模型
MODEL_PATH = '/workspace/best.engine'  # 替換為模型的正確路徑

logging.info("Loading model...")
model = YOLO(MODEL_PATH, task="classify")  # 使用 ultralytics 的 YOLO 加載模型
logging.info("Model loaded successfully.")

UPLOAD_FOLDER = '/workspace/uploads'  # 圖片上傳目錄
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request at /predict")
    try:
        # 檢查請求是否包含檔案
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # 儲存檔案
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            logging.info(f"File saved at {filepath}")

            # 推理
            results = model(filepath)  # 使用 YOLO 進行推理
            print(results[0].probs.top1)
            
            # 返回 JSON 格式的預測結果
            return jsonify({'predictions': results[0].probs.top1})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, debug=True)

