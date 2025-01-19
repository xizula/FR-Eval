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
from torchvision import transforms
from models.AdaFace.inference import load_pretrained_model

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
            img = img.detach().cpu().numpy().transpose(1, 2, 0)*255
            
            embeddings = self.arcface_model.get_feat(img)
            emb.append(embeddings)
        return torch.tensor(emb)

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (norm(e_i) * norm(e_j)) *100

class AdaFace:
    def __init__(self):
        self.model = load_pretrained_model('ir_50')
        self.model.eval()
        self.model.cuda()

    def __call__(self, images, train=False):
        images = images.detach().cpu().numpy().transpose(0,2,3,1)
        norm = ((images) - 0.5) / 0.5
        tensor = torch.tensor(norm.transpose(0,3,1,2)).float()
        if not train:
          with torch.no_grad():
            embeddings, _ = self.model(tensor.cuda())
        else:
            embeddings, _ = self.model(tensor.cuda())
        return embeddings.detach().cpu()
    
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


class HeadPose:
    def __init__(self):
        self.model_paths = ["models/head_pose/fsanet-1x1-iter-688590.onnx", "models/head_pose/fsanet-var-iter-688590.onnx"]
        self.models = [onnxruntime.InferenceSession(model_path) for model_path in self.model_paths]
    
    def __call__(self, image):
        image = [self.transform(image)]
        # print(image.shape)
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.models
        ]
        # inputs = [{model.get_inputs()[0].name: image} for model in self.models]
        # yaw_pitch_roll_results = [
        #     model.run(["output"], {"input": input})[0] for model, input in zip(self.models, inputs)
        # ]
        yaw, pitch, roll = np.mean(np.vstack(yaw_pitch_roll_results), axis=0)
        return yaw, pitch, roll
    
    def transform(self, image):
        trans = transforms.Compose([
            Resize(64),
            Normalize(mean=127.5,std=128),
            ToTensor()
            ])
        image = trans(image)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img-self.mean)/self.std
        return img

class ToTensor(object):
    def __call__(self, img):
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img)

class Resize(object):
    def __init__(self, size=64):
        self.size = (size, size)

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        return img

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

