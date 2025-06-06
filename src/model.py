import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

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
from torch import nn

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
        self.model = load_pretrained_model('ir_101')
        self.model.eval()
        self.model.cuda()

    def __call__(self, images, train=False):
        # images = images.detach().cpu().numpy().transpose(0,2,3,1)
        # norm = ((images) - 0.5) / 0.5
        # tensor = torch.tensor(norm.transpose(0,3,1,2)).float()
        if not train:
          with torch.no_grad():
            embeddings, _ = self.model(images.cuda())
        else:
            embeddings, _ = self.model(images.cuda())
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
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.models
        ]

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


class QualityFaceNet(nn.Module):
    def __init__(self):
        super(QualityFaceNet, self).__init__()
        self.model = load_model('facenet')
        self.features = nn.Sequential(*list(self.model.model.children())[:-4]).cuda()
        self.dropout = nn.Dropout(p=0.5).cuda()
        self.fc = nn.Linear(1792, 1).cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten before the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class QualityAdaFace(nn.Module):
    def __init__(self, **kwargs):
        super(QualityAdaFace, self).__init__()
        original_model = load_model('adaface', **kwargs)
        self.input_layer = original_model.model.input_layer
        self.body = original_model.model.body
        
        # Add new layers: dropout and a fully connected layer
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1).cuda()  # Assuming 512 features from the body
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = x.mean(dim=(2, 3))  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

def load_quality_model(name: str, **kwargs):
    if name == 'facenet':
        return QualityFaceNet()
    elif name == 'adaface':
        return QualityAdaFace(**kwargs)
    else:
        raise ValueError(f'Quality model {name} not supported')
    
def load_quality_method(name: str):
    name = name.lower()
    model = QualityAdaFace()
    ckpt_path = f'models/quality/adaface_{name}.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model