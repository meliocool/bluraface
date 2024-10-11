from mtcnn import MTCNN
import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import os

training_images = "...\\" # Adjust based on your dataset folder

Face = [] 

detector = MTCNN() # face detector
MLmodel = InceptionResnetV1(pretrained='vggface2').eval() # face recognizer

def training():
    for faces in os.listdir(training_images):
        img_path = os.path.join(training_images, faces)
        to_array = cv.imread(img_path)
        img_rgb = cv.cvtColor(to_array, cv.COLOR_BGR2RGB)
        detected_face = detector.detect_faces(img_rgb)
        for face in detected_face:
            x, y, width, height = face['box']
            facialRegion = img_rgb[y:y+height, x:x+width]
            facialRegion_resized = cv.resize(facialRegion, (160, 160))
            facialRegion_tensor = torch.tensor(facialRegion_resized).float().permute(2, 0, 1) / 255.0
            facialRegion_tensor = facialRegion_tensor.unsqueeze(0)

            with torch.no_grad():
                embedding = MLmodel(facialRegion_tensor)
                Face.append(embedding.squeeze().numpy())

training()
print("Training finished!")

Face = np.array(Face)
face_embeddings = 'C:/Users/Asus VivobookPro/Documents/CODING STUFF/AI/blurAFace/face_embeddings'
file_name = 'embeddings.npy'

save_path = os.path.join(face_embeddings, file_name)
np.save(save_path, Face) # save the embedded faces
            