import os
import torch

from PIL import Image
from torchvision import transforms

from preprocess import crop_center
from model import SimpleCNN

from grad_cam import GradCAM
import matplotlib.pyplot as plt
import numpy as np


IMAGE_DIR = '../../../img/raw'
MODEL_PATH = '../output5/100.00_11-20.pth'

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(model, image_path, class_names):
    img = crop_center(image_path)
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    output = model(img)
    _, prediceted = torch.max(output, 1)

    
    return class_names[prediceted.item()] 

def infer_with_grad_cam(ㄊmodel, image_path, class_names, grad_cam=None, save_dir="gradcam_output"):
    img = crop_center(image_path)
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    # Forward pass
    output = model(img)
    _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]

    # Grad-CAM Visualization
    if grad_cam:
        cam = grad_cam.generate_cam(img, target_class=predicted.item())

        # Save Grad-CAM heatmap
        os.makedirs(save_dir, exist_ok=True)
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()

        cam_resized = np.uint8(255 * cam)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM Heatmap")
        # plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.axis("off")

        plt.savefig(save_path)
        plt.close()
        print(f"Saved Grad-CAM visualization to {save_path}")

    return predicted_class

if __name__  == '__main__':
    print(f'Using model: {MODEL_PATH}')
    class_names = ['crease', 'dusty_break', 'dusty_inside', 'tin', 'OK']

    model = torch.load(MODEL_PATH, weights_only=False).to(device)
    #model = SimpleCNN().to(device)
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Grad-CAM instance
    grad_cam = GradCAM(model, model.conv2)  # 使用最后一个卷积层

    total, correct = 0, 0
    types = [[] for i in range(len(class_names))]

    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith(('.jpg')):
                image_path = os.path.join(root, file)
                # ans = infer(model , image_path, class_names)
                ans = infer_with_grad_cam(model, image_path, class_names, grad_cam=grad_cam)

                actual_class = image_path.split("/")[-2]

                total += 1
                if actual_class == ans:
                    correct += 1
                
                ansIndex = class_names.index(ans)
                types[ansIndex].append(file)
    
    cmp = lambda x:int(x.split(".")[0])

    print('=' * 20 + " result " + '=' * 20)
    for i in range(len(class_names)):
        print(f'{class_names[i]} images: {", ".join(sorted(types[i], key=cmp))}')
        print()
    print(f'Accuracy is {(correct / total) * 100}')
    print('=' * 20 + "========" + '=' * 20)
