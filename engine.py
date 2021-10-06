import torch
import torch.nn as nn
import tqdm
import math
import numpy as np  
from model import BaselineModel
from torch.utils.data import DataLoader
 

def do_epochs(epochs, train_loader, valid_loader, model, optimizer,  metrics = [], X_prepro = None, y_prepro = None, device = 'cuda' ):
    '''
    训练+评判模型
    '''

    baseline_model = BaselineModel(device)
    train_baseline(loader=train_loader, model=baseline_model, y_prepro=None )
    outputs, targets = evaluate(loader=valid_loader, model=baseline_model, X_preprocessor=X_prepro, y_preprocessor=None, device=device)

    baseline_valid_scores = []
    for m in metrics:    
        score = m(targets, outputs)
        baseline_valid_scores.append((m.__name__, score))
        print(f' baseline {m.__name__}: {score}')

    for epoch in range(epochs): 
        train_loss = train(
            train_loader,
            model, 
            optimizer=optimizer,  
            device=device, 
            X_preprocessor=X_prepro,
            y_preprocessor=y_prepro)

        outputs, targets = evaluate(
            valid_loader,
            model, 
            device=device, 
            X_preprocessor=X_prepro, 
            y_preprocessor=y_prepro)

        valid_scores = []
        for m in metrics:    
            score = m(targets, outputs)  
            valid_scores.append((m.__name__, score))
        
        print(f"epoch {epoch}/{epochs}, train loss: {train_loss}")
        for i in range(len(metrics)):
            _, score_bl = baseline_valid_scores[i]
            score_name, score = valid_scores[i]
            print(f'\tvalid {score_name}: {score} (baseline: {score_bl})')


def train_baseline(loader: DataLoader, model, y_prepro ):
    model.train()

    y = np.array([])

    for data in tqdm.tqdm(loader, 'training baseline'):

        targets = data['target']

        if y_prepro is not None:
            targets = y_prepro.transform(targets)
         
        y = np.vstack((y.reshape(-1, 1), targets.reshape(-1, 1)))
    
    model.fit(y) 

        

def train(loader, model, optimizer,  X_preprocessor = None, y_preprocessor = None, device='cuda'):
    """
    返回均误差(L2 norm)。
    """
    model.train()
    epoch_loss = 0
    for data in tqdm.tqdm(loader, 'training'):
        
        inputs = data["image"]
        targets = data["target"]

        if X_preprocessor is not None:
            inputs = X_preprocessor.transform(inputs)
        
        if y_preprocessor is not None:
            targets = y_preprocessor.transform(targets)

        inputs = inputs.to(device, dtype=torch.float)
 
        targets = torch.flatten(targets).to(device) 

        if optimizer is not None:
            optimizer.zero_grad()

        outputs = torch.flatten(model(inputs),1)
 
        loss = model.get_loss_fn()(outputs , targets)
        epoch_loss += loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return math.sqrt(epoch_loss / len(loader))


def evaluate(loader, model, X_preprocessor = None, y_preprocessor = None, device='cuda'):
    model.eval()
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for data in tqdm.tqdm(loader, 'evaluating'):
            
            inputs = data['image']
            targets = data['target']
            
            if X_preprocessor is not None:
                inputs = X_preprocessor.transform(inputs)
            
            if y_preprocessor is not None:
                targets = y_preprocessor.transform(targets)
     
            inputs = inputs.to(device, dtype= torch.float)
            targets = torch.flatten(targets.to(device, dtype= torch.float))

            outputs = torch.flatten(model(inputs).flatten())
        
            targets = targets.detach().cpu().numpy().tolist()
            output = outputs.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(output)
    return final_outputs, final_targets
