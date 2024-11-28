import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

from dataset import get_dataset
from model import load_model
import yaml
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle


config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

models = config['model']
datasets = config['data']


for dataset_name in datasets:
    for model_name in models:
        data_loader = get_dataset(dataset_name)
        print(model_name)
        print(dataset_name)
        embeddings = []
        paths = []
        labels = []
        wrong_paths = []
        model = load_model(model_name)
        for image, label, path in tqdm(data_loader):
            try:
                embedding = model(image.cuda())
                embedding = embedding.squeeze()
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0)
                embeddings.append(embedding)
                paths.extend(path)
                labels.extend(list(label.numpy()))
            except Exception as e:
                wrong_paths.extend(path)
                print(f"Error: {e}")
            
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = np.vstack(embeddings)
        embeddings = embeddings.tolist()
        df = {'embedding': embeddings, 'label': labels, 'path': paths}
        embeddings_df = pd.DataFrame(df)
        embeddings_df.to_csv(f'data/embeddings/{model_name}_{dataset_name}_embeddings.csv', index=False)
