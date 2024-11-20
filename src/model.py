from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.model_zoo import get_model
import torch
# from insightface.app import FaceAnalysis
import cv2
import onnx
import onnxruntime
from insightface.model_zoo import get_model
import numpy as np
import torch
from ellzaf_ml.models import GhostFaceNetsV2
# from torchvision.io import read_image
from numpy.linalg import norm

class FaceNet:
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

    def __call__(self, images):
        embedding = self.resnet(images)
        return embedding.detach().cpu()

    def compute_similarities(self, e_i, e_j):
        squared_norms = np.sum(e_i**2, axis=1) 
        dist_squared = squared_norms[:, None] + squared_norms[None, :] - 2 * (e_i @ e_i.T)
        dist_squared = np.maximum(dist_squared, 0)
        distances = np.sqrt(dist_squared)
        return distances


class ArcFace:
    def __init__(self):
        self.arcface_model = get_model('buffalo_l', allow_download=True, download_zip=True)
        self.arcface_model.prepare(ctx_id=0)

    def __call__(self, images):
        emb = []
        for img in images:
            cv2.imwrite('temp.jpg', (img.detach().cpu().numpy().transpose(1, 2, 0)*255))
            img = img.detach().cpu().numpy().transpose(1, 2, 0)*255
            
            embeddings = self.arcface_model.get_feat(img)
            emb.append(embeddings)
        return torch.tensor(emb)

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (norm(e_i) * norm(e_j)) *100

class AdaFace:
    def __init__(self):
        self.adaface_model = onnxruntime.InferenceSession("models/adaface.onnx")
        self.input_name = self.adaface_model.get_inputs()[0].name
        self.output_name = self.adaface_model.get_outputs()[0].name
    
    def __call__(self, images):
        images = images.detach().cpu().numpy().transpose(0,2,3,1)
        norm = ((images) - 0.5) / 0.5
        tensor = norm.transpose(0,3,1,2).astype(np.float32)
        embeddings = self.adaface_model.run([self.output_name], {self.input_name: tensor})[0]

        return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T


class GhostFaceNet:
    def __init__(self):
        self.model = GhostFaceNetsV2(image_size=112, width=1, dropout=0.)
        self.model.eval()
        self.model.cuda()

    def __call__(self, images):
        return self.model(images).detach().cpu().numpy()

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j))*100


def load_model(name: str):
    if name == 'facenet':
        return FaceNet()
    elif name == 'arcface':
        return ArcFace()
    elif name == 'adaface':
        return AdaFace()
    elif name == 'ghostfacenet':
        return GhostFaceNet()
    else:
        raise ValueError(f'Model {name} not supported')

