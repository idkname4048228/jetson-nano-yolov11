# 使用 YOLO 的基礎映像
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# 設定工作目錄
WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install flask pillow

# 將需要的模型和腳本複製到容器內
# 假設 best.pt 和 inference.sh 位於本地目錄中
COPY best.engine /workspace/best.engine
COPY demo_imgs /workspace/demo_imgs
COPY server.py /workspace/server.py

# 設置 Flask 預設埠（可選）
ENV FLASK_RUN_PORT=5000

# 啟動 Python 伺服器
CMD ["python3", "/workspace/server.py"]

