DATA_DIR=/home/k7/LED_flaw/hw/yolo/crop_imgs
USE_MODEL=yolo11n-cls.pt
EPOCHS=100
IMG_SIZE=640

yolo classify train data=$DATA_DIR model=$USE_MODEL epochs=$EPOCHS imgsz=$IMG_SIZE
