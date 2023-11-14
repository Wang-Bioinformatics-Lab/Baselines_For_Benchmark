import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import pandas as pd

from tqdm import tqdm

from time import time

# Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, depth=10):
        super(LinearModel, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.feature_extractor = nn.ModuleList([nn.ReLU(nn.Linear(hidden_size, hidden_size)) for _ in range(depth)])
        self.inference_head = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.first_layer(x)
        out = self.feature_extractor(out)
        out = self.inference_head(out)
        return out

def train_epoch(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Define the training function
def train(model, train_loader, criterion, epochs, optimizer, device):
    model.train()
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print("Epoch: {}, Loss: {}".format(epoch, train_loss))
    

# Define the testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_loader)

def get_optimizer(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def get_criterion():
    criterion = nn.MSELoss()
    return criterion

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, similarity_path, subsample=False):
        self.data_path = data_path
        paths = sorted(os.listdir(data_path))
        self.data_points = [os.path.join(data_path, x) for x in paths]
        self.spectrum_ids = set([x.rsplit('.',1)[0] for x in paths]) # Covnert to set for faster lookups
        #self.similarities = pd.read_csv(similarity_path, names=['spectrum_id1', 'spectrum_id2', 'similarity']).groupby('spectrum_id1').sample(n=max_number_similar, replace=True)
        self.subsample = subsample  # Subsample will ensure roughly uniform distribution over the similarity scores
        
        # Verify the similarities are in the dataset
        # self.similarities = self.similarities.filter(lambda x: x['spectrum_id1'].isin(self.spectrum_ids),axis=0)
        # self.similarities = self.similarities.filter(lambda x: x['spectrum_id2'].isin(self.spectrum_ids),axis=0)
        print("\tLoading similarities...")
        start_time = time()
        temp_similarities = json.load(open(similarity_path, 'r'))
        print(len(temp_similarities))
        # Ensure key, related spectrum_ids in temp_similarities are in the dataset
        temp_similarities = {k: [{'Tanimoto_Similarity':x['Tanimoto_Similarity'], 'spectrumid2':[y for y in x['spectrumid2'] if y in self.spectrum_ids]}  for x in v]  for k, v in temp_similarities.items()  if k in self.spectrum_ids}
        print(len(temp_similarities))
        # Remove all 'Tanimoto_Similarity' keys that have no spectra in dataset
        temp_similarities = {k: [{'Tanimoto_Similarity':x['Tanimoto_Similarity'], 'spectrumid2':x['spectrumid2']}  for x in v if len(x['spectrumid2']) > 0]  for k, v in temp_similarities.items()  if k in self.spectrum_ids}
        print(len(temp_similarities))
        
        # Remove all spectrum_ids that have no similarities
        temp_similarities = {k: v for k, v in temp_similarities.items() if len(v) > 0}
        print(temp_similarities.keys())
        print(len(temp_similarities))
        self.similarities = temp_similarities
        
        print(f"Dataset contains {len(self.spectrum_ids)} spectra, but only has similarities for {len(self.similarities)} spectra.")
        print(f"In total there are {sum([len(v) for _,v in self.similarities.items()])} tanimoto similarities.")
        
        print(f"\tDone loading similarities in {time() - start_time:2f} seconds.")
        
    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        file_name = self.data_points[idx]
        spectrum_id = file_name.split('/')[-1].split('.')[0]
        # similarities = self.similarities.get_group(spectrum_id)
        # With the current setup the maximum number of similar spectra is 20 * 10 = 200. <-- this is not true, we allow duplicate spectra
        # TODO: Fix above and add option for subsampling
        # similar_spectra = similarities['spectrum_id2']
        # similaritiy_scores = torch.Tensor(similarities['similarity'])
        
        
        """
        JSON format:
        {'b': [{'Tanimoto_Similarity': 1.0, 'spectrumid2': ['d']},
                {'Tanimoto_Similarity': 0.03571428571428571, 'spectrumid2': ['a', 'x', 'y']},
                {'Tanimoto_Similarity': 0.037037037037037035, 'spectrumid2': ['c']}
                ],
        'a': [{'Tanimoto_Similarity': 0.03571428571428571, 'spectrumid2': ['b']},
                {'Tanimoto_Similarity': 1.0, 'spectrumid2': ['a', 'x', 'y']},
                {'Tanimoto_Similarity': 0.08888888888888889, 'spectrumid2': ['c']}
                ]
        }
        If we choose to sample the json, we should sample per similarity, each similarity represents 
        one structure. This will constrain us to 20 * 10 = 200 spectra per struucture.
        """
        
        if self.subsample:
            # We will randomly sample one spectra from this structure
            random_indices      = torch.Tensor([random.choice(range(len(x['spectrumid2']))) for x in self.similarities[spectrum_id]])
            similar_spectra     = sum([x['spectrumid2'][random_indices[idx]] for idx, x in enumerate(self.similarities[spectrum_id])], [])
            similarity_scores   = torch.Tensor([[x['Tanimoto_Similarity']] * len(x['spectrumid2']) for x in self.similarities[spectrum_id]])
        else:
            # We will take all spectra from this structure            
            similar_spectra = sum([x['spectrumid2'] for x in self.similarities[spectrum_id]], [])
            similarity_scores = torch.Tensor([[x['Tanimoto_Similarity']] * len(x['spectrumid2'])  for x in self.similarities[spectrum_id]]).flatten()
        
        this_spectra = np.load(file_name)
        other_spectra = torch.Tensor([np.load(os.path.join(self.data_path, x)) for x in similar_spectra])
        
        return this_spectra, other_spectra, similarity_scores
    
def get_train_loader(data_path, similarity_path, batch_size=10, shuffle=True):
    dataset = SpectralDataset(data_path, similarity_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader    

def main():
    parser = argparse.ArgumentParser(description='Train a simple model.')
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--similarity_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default = 75)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    
    if args.mode == 'train':
        model = LinearModel(512, 256, 1)
        print("Loading data...")
        train_loader = get_train_loader(args.train_data, args.similarity_path, args.batch_size)
        print("Done Loading Data")
        criterion = get_criterion()
        optimizer = get_optimizer(model, 0.001)
        
        print("Training...")
        train(model, train_loader, criterion, args.num_epochs, optimizer, args.device)
        print("Done Training")
        
        # Save the model
        print(f"Saving model to {args.model_path}...",flush=True)
        torch.save(model.state_dict(), args.model_path)
        print("Done Saving",flush=True)
    elif args.mode == 'test':
        # Load the model
        model = LinearModel(512, 256, 1)
        model.load_state_dict(torch.load(args.model_path))
        test_loader = get_train_loader(args.test_data)
        criterion = get_criterion()
        test_loss = test(model, test_loader, criterion, args.device)
        print("Test Loss: {}".format(test_loss))
    else:
       print("Unknown mode. Exiting.",flush=True) 

if __name__ == "__main__":
    main()