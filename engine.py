import torch
import torch.nn as nn
import tqdm
import math
import numpy as np
from model import BaselineModel
from torch.utils.data import DataLoader
from colorama import Fore, Style


def get_name(m):
    if hasattr(m, 'name'):
        name = m.name
    else:
        name = m.__name__
    return name


def get_delta_value_str(last_value, curr_value, bigger_better=True):
    if last_value is None:
        return ''
    delta_value = curr_value - last_value

    if curr_value > last_value and bigger_better == True or curr_value < last_value and bigger_better == False:
        return f'({Fore.GREEN}{delta_value:+.6f}{Fore.RESET})'
    return f'({Fore.RED}{delta_value:+.6f}{Fore.RESET})'


def get_value_str(value):
    if value is None:
        return ''

    return f'{Fore.CYAN}{value:.6f}{Fore.RESET}'


def do_epochs(epochs, train_loader, valid_loader, model, optimizer, baseline_model, metrics=[], X_prepro=None, y_prepro=None, device='cuda'):
    '''
    训练+评判模型
    '''

    train_baseline(loader=train_loader, model=baseline_model, y_prepro=None)
    outputs, targets = evaluate(loader=valid_loader, model=baseline_model,
                                X_preprocessor=X_prepro, y_preprocessor=None, device=device)

    baseline_valid_scores = []
    for m in metrics:
        score = m(targets, outputs)
        name = get_name(m)
        baseline_valid_scores.append((name, score))
        print(f'    baseline {name}: {get_value_str(score)}')

    last_epoch_scores = None
    last_epoch_train_loss = None
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
            valid_scores.append((get_name(m), score))

        delta_str = get_delta_value_str(
            last_epoch_train_loss, train_loss, bigger_better=False)

        print(
            f"epoch {epoch}/{epochs}, train loss: {get_value_str(train_loss)} {delta_str}")
        last_epoch_train_loss = train_loss

        for i in range(len(metrics)):
            _, score_bl = baseline_valid_scores[i]
            score_name, score = valid_scores[i]

            if last_epoch_scores is not None:
                _, last_score = last_epoch_scores[i]
            else:
                last_score = None

            delta_str = get_delta_value_str(
                last_score, score, bigger_better=True)

            if score < score_bl:
                xxx_baseline_str = f'{Fore.RED}below baseline{Fore.RESET}'
            else:
                xxx_baseline_str = f'{Fore.GREEN}above baseline{Fore.RESET}'

            print(
                f'    valid {score_name}: {get_value_str(score)} {delta_str}, {xxx_baseline_str} ({get_value_str(score_bl)})')

        last_epoch_scores = valid_scores


def train_baseline(loader: DataLoader, model, y_prepro):
    model.train()

    y = np.array([])

    for _, targets in tqdm.tqdm(loader, 'training baseline'):

        if y_prepro is not None:
            targets = y_prepro.transform(targets)

        y = np.vstack((y.reshape(-1, 1), targets.reshape(-1, 1)))

    model.fit(y)


def train(loader, model, optimizer,  X_preprocessor=None, y_preprocessor=None, device='cuda'):
    """
    返回均误差(L2 norm)。
    """
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm.tqdm(loader, 'training'):

        if X_preprocessor is not None:
            inputs = X_preprocessor.transform(inputs)

        if y_preprocessor is not None:
            targets = y_preprocessor.transform(targets)

        inputs = inputs.to(device, dtype=torch.float)

        targets = torch.flatten(targets).to(device)
 
        optimizer.zero_grad()

        outputs = torch.flatten(model(inputs), 1)

        loss = model.get_loss_fn()(outputs, targets)
        epoch_loss += loss
 
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader) 


def evaluate(loader, model, X_preprocessor=None, y_preprocessor=None, device='cuda'):
    model.eval()
    final_targets = None
    final_outputs = None
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(loader, 'evaluating'):

            if X_preprocessor is not None:
                inputs = X_preprocessor.transform(inputs)

            if y_preprocessor is not None:
                targets = y_preprocessor.transform(targets)

            inputs = inputs.to(device, dtype=torch.float)
            targets = torch.reshape(targets.to(
                device, dtype=torch.float), (-1, 1))

            # 模型的输出可能为(N,),(N,1),（N,classes)
            outputs: torch.Tensor = model(inputs)

            if len(outputs.shape) == 1:
                outputs = outputs.reshape((-1, 1))

            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()

            if final_targets is None:
                final_targets = targets
            else:
                final_targets = np.vstack((final_targets, targets))

            if final_outputs is None:
                final_outputs = outputs
            else:
                final_outputs = np.vstack((final_outputs, outputs))

    return final_outputs, final_targets
