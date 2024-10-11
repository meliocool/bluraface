from mtcnn import MTCNN
import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import os

training_images = "C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\HaerinBlur\\training\\Haerin"

haerinFace = []

detector = MTCNN()
MLmodel = InceptionResnetV1(pretrained='vggface2').eval()

def hating():
    for haerin in os.listdir(training_images):
        img_path = os.path.join(training_images, haerin)
        to_array = cv.imread(img_path)
        img_rgb = cv.cvtColor(to_array, cv.COLOR_BGR2RGB)
        haerin_face = detector.detect_faces(img_rgb)
        for face in haerin_face:
            x, y, width, height = face['box']
            facialRegion = img_rgb[y:y+height, x:x+width]
            facialRegion_resized = cv.resize(facialRegion, (160, 160))
            facialRegion_tensor = torch.tensor(facialRegion_resized).float().permute(2, 0, 1) / 255.0
            facialRegion_tensor = facialRegion_tensor.unsqueeze(0)

            with torch.no_grad():
                embedding = MLmodel(facialRegion_tensor)
                
                haerinFace.append(embedding.squeeze().numpy())

hating()
print("Haerin known")

haerinFace = np.array(haerinFace)

np.save('haerin.npy', haerinFace)
            