from mtcnn import MTCNN
import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

trained_model = np.load('C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\HaerinBlur\\face_embeddings\\haerin.npy')

detector = MTCNN()
MLmodel = InceptionResnetV1(pretrained='vggface2').eval()

img_path = 'C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\HaerinBlur\\testing\\group2.jpg'
img = cv.imread(img_path)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

faces = detector.detect_faces(img_rgb)
final_img = img_rgb.copy()

for face in faces:
    x, y, width, height = face['box']
    facialRegion = img_rgb[y:y+height, x:x+width]

    facialRegion_resized = cv.resize(facialRegion, (160, 160))
    facialRegion_tensored = torch.tensor(facialRegion_resized).float().permute(2, 0, 1) / 255
    facialRegion_tensored = facialRegion_tensored.unsqueeze(0)

    with torch.no_grad():
       embedding = MLmodel(facialRegion_tensored).numpy().flatten()

    max_similarity = -1
    match = -1

    for idx, stored_embedding in enumerate(trained_model):
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            match = idx

    similarity_threshold = 0.8

    if max_similarity > similarity_threshold:
        blur = cv.GaussianBlur(facialRegion, (99, 99), 30)
        final_img[y:y+height, x:x+width] = blur
    else:
        continue

final_img_bgr = cv.cvtColor(final_img, cv.COLOR_RGB2BGR)
cv.imshow('Haerin Blur Machine', final_img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()
