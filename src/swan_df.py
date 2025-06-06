import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

from model import load_model
from prep_vid_quality import get_images_tensor
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

model = load_model('adaface')
path = Path('data/SwanDF')

videos = list(path.rglob('*.mp4'))
data = {'embedding': [], 'class': [], 'path': []}

for video in tqdm(videos):
    # print(video)
    images = get_images_tensor(video)
    label = video.parent.name
    if images is None or images.shape == torch.Size([0]):
        continue
    embeddings = model(images.float().cuda())
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().numpy()
    for i, embedding in enumerate(embeddings):
        embedding = embedding.squeeze()
        data['class'].append(label)
        data['embedding'].append(embedding)
        data['path'].append(video.as_posix())

df = pd.DataFrame(data)
df.to_parquet(f'data/demo/swan_sdd_mod.parquet')


