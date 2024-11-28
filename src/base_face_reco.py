import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(parent_dir)
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

import itertools
import pandas as pd
import yaml
import ast
from tqdm import tqdm
import torch
from model import load_model
import numpy as np

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

datasets = config['data']
models = config['model']


for model_name in tqdm(models):
    for dataset in tqdm(datasets):
        model = load_model(model_name)
        file_path = f'data/embeddings/{model_name}_{dataset}_embeddings.csv'
        df = pd.read_csv(file_path)

        labels = df['label'].values
        embeddings = df['embedding'].values
        embeddings = np.array([np.array(ast.literal_eval(e)) for e in embeddings])
        num_samples = len(embeddings)
        scores = []
        class_labels = []
        chunk_size = 5000
        for i in range(0, num_samples, chunk_size):
            for j in range(i + 1, num_samples, chunk_size):
                print("Processing chunk", i, j)

                emb_chunk = embeddings[i:i + chunk_size]
                emb_chunk_2 = embeddings[j:j + chunk_size]

                score_chunk = model.compute_similarities(emb_chunk, emb_chunk_2)

                labels_chunk = labels[i:i + chunk_size]
                labels_chunk_2 = labels[j:j + chunk_size]
                labels_matrix = np.equal(labels_chunk[:, None], labels_chunk_2)
                np.fill_diagonal(labels_matrix, False)

                upper_triangle_indices = np.triu_indices_from(score_chunk, k=1)
                try:
                    class_label = labels_matrix[upper_triangle_indices]
                    score = score_chunk[upper_triangle_indices]
                    class_labels.extend(class_label)
                    scores.extend(score)
                except:
                    continue

        data = pd.DataFrame({
            'label': class_labels,
            'score': scores
        })
        data.to_csv(f'data/scores/{model_name}_{dataset}_base_scores.csv', index=False)