IMGS_DIR=/home/k7/LED_flaw/hw/test_imgs/crop_img
#MODEL=runs/classify/train12/weights/best.pt
MODEL=good_trains/crop_n/weights/best.pt

yolo classify predict model=$MODEL source=$IMGS_DIR
