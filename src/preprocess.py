import os
from PIL import Image, ImageEnhance
import numpy as np
import random

def crop_center(image_path, output_size=(600, 600)):
    img = Image.open(image_path)
    width, height = img.size

    new_width, new_height = output_size
    left = (width - new_width) // 2
    up = (height - new_height) // 2
    right = left + new_width
    bottom = up + new_height

    img_cropped = img.crop((left, up, right, bottom))
    return img_cropped

def adjust_bright(img, file, save_dir):
    step = 0.1
    s, f = 0.8, 1.2
    curr = s
    
    enhancer = ImageEnhance.Brightness(img)
    
    while curr <= f + (step / 2):
        if curr == 1:
            curr += step
            continue
        lighter_img = enhancer.enhance(curr)
        lighter_img.save(f'{save_dir}light{curr:.2f}_{file}')
        
        curr += step

def adjust_saturation(img, file, save_dir):
    enhancer = ImageEnhance.Color(img)

    step = 0.5
    s, f = 0.5, 1.5
    curr = s

    while curr <= f + (step / 2):
        if curr == 1:
            curr += step
            continue
        saturated_img = enhancer.enhance(curr)
        saturated_img.save(f'{save_dir}color{curr:.2f}_{file}')

        curr += step

def add_noise(img, file, save_dir, noise_level=25):
    img_array = np.array(img)
    mean = 0
    stddev = noise_level
    gaussian_noise = np.random.normal(mean, stddev, img_array.shape).astype('int')
    noisy_img_array = np.clip(img_array + gaussian_noise, 0, 255).astype('uint8')
    noisy_img = Image.fromarray(noisy_img_array)
    noisy_img.save(f'{save_dir}noisy_{file}')
    
def process_images(input_dir, save_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg')):
                image_path = os.path.join(root, file)
                new_img = crop_center(image_path)
                adjust_bright(new_img, file, save_dir)
                
                LR_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                LR_img.save(f'{save_dir}LR_{file}')
                if random.random() < 0.5:
                    add_noise(LR_img, f'LR_{file}', save_dir)
                if random.random() < 0.5:
                    adjust_bright(LR_img, f'LR_{file}', save_dir)
                if random.random() < 0.5:
                    adjust_saturation(LR_img, f'LR_{file}', save_dir)

                TB_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)
                TB_img.save(f'{save_dir}TB_{file}')
                if random.random() < 0.5:
                    add_noise(TB_img, f'TB_{file}', save_dir)
                if random.random() < 0.5:
                    adjust_bright(TB_img, f'TB_{file}', save_dir)
                if random.random() < 0.5:
                    adjust_saturation(TB_img, f'TB_{file}', save_dir)

IMGS_ROOT = '../../../img/5raw/'
SAVE_ROOT = '../../img5/'
OK_IMGS = f'{IMGS_ROOT}/OK/'
OK_SAVE = f'{SAVE_ROOT}/OK/'

crease_IMGS = f'{IMGS_ROOT}/crease/'
crease_SAVE = f'{SAVE_ROOT}/crease/'

dusty_break_IMGS = f'{IMGS_ROOT}/dusty_break/'
dusty_break_SAVE = f'{SAVE_ROOT}/dusty_break/'

dusty_inside_IMGS = f'{IMGS_ROOT}/dusty_inside/'
dusty_inside_SAVE = f'{SAVE_ROOT}/dusty_inside/'

tin_IMGS = f'{IMGS_ROOT}/tin/'
tin_SAVE = f'{SAVE_ROOT}/tin/'

process_images(OK_IMGS, OK_SAVE)
process_images(crease_IMGS, crease_SAVE)
process_images(dusty_break_IMGS, dusty_break_SAVE)
process_images(dusty_inside_IMGS, dusty_inside_SAVE)
process_images(tin_IMGS, tin_SAVE)