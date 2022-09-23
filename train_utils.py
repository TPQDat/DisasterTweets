# Each epoch: train, evaluate
# Loop n_epoch
import os
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import CFG
from dataset import create_dataloader
from models import create_model, create_tokenizer

def train_epoch(model, dataloader, optimizer):
    """
    1. model = model.train
    2. (model, data) = (model, data).to(device)  - device: cuda, cpu, ...
    3. optimizer.zero_grad()
    4. outputs = model(preprocessed input data)
    5. loss = Loss-function(ouputs, target)
    6. loss.backward()
    7. (optimizer.step(), scheduler.step())
    8. Cache outputs, target, loss, ....
    9. Calculate metric: MSE, MAE, RMSE, accuracy, ...
    """
    model = model.train()
    losses = []
    labels = None
    predictions = None

    for data in tqdm(dataloader):
        input_ids = data['input_ids'].to(CFG.device)
        attention_mask = data['attention_mask'].to(CFG.device)
        targets = data['label'].to(CFG.device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print('outputs: ', outputs)
        print('target: ', targets)
        loss = F.binary_cross_entropy(outputs, targets)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        targets = targets.numpy()
        labels = np.atleast_1d(targets) if labels is None else np.concatenate([labels, np.atleast_1d(targets)])
        outputs = outputs.numpy()
        predictions = np.atleast_1d(outputs) if predictions is None else np.concatenate([predictions, np.atleast_1d(outputs)])

    loss = np.mean(losses)
    predictions = 1*(predictions > CFG.threshold)
    accuracy = (sum(predictions == labels)/predictions.size)
    return loss, accuracy

def eval_model(model, dataloader):
    """
    1. model = model.eval()
    2. (model, data) = (model, data).to(device)  - device: cuda, cpu, ...
    4. outputs = model(preprocessed input data)
    5. loss = Loss-function(ouputs, target)
    8. Cache outputs, target, loss, ....
    9. Calculate metric: MSE, MAE, RMSE, accuracy, ...
    """
    model = model.eval()
    losses = []
    labels = None
    predictions = None

    for data in tqdm(dataloader):
        input_ids = data['input_ids'].to(CFG.device)
        attention_mask = data['attention_mask'].to(CFG.device)
        targets = data['label'].to(CFG.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.binary_cross_entropy(outputs, targets)
        losses.append(loss.item())

        targets = targets.numpy()
        labels = np.atleast_1d(targets) if labels is None else np.concatenate([labels, np.atleast_1d(targets)])
        outputs = outputs.numpy()
        predictions = np.atleast_1d(outputs) if predictions is None else np.concatenate([predictions, np.atleast_1d(outputs)])

    loss = np.mean(losses)
    predictions = 1*(predictions > CFG.threshold)
    accuracy = (sum(predictions == labels)/predictions.size)
    return loss, accuracy

def train_fold(model_ckpt, train_df, val_df, save_model_ckpt):
    """
    1. Create model -> model.to(device)
    2. Create dataloader(train_dataloader, val_dataloader)
    3. Create optimizer, scheduler
    4. Train:
        for epoch in n_epochs:
            train_epochs -> loss, metric
            eval_epochs -> loss, metric
            choose_best_epoch or early stopping based on (loss, metric)

    """
    model = create_model(model_ckpt)
    model = model.to(CFG.device)

    tokenizer = create_tokenizer(model_ckpt)
    train_dataloader = create_dataloader(train_df, tokenizer)
    val_dataloader = create_dataloader(val_df, tokenizer)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

    best_accuracy = -1
    best_epoch = 0
    for epoch in range(CFG.num_epochs):
        print('-'*5 + f'Epoch {epoch+1}/{CFG.num_epochs}' + '-'*5)
        # Train phase
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer)
        print(f'Train_loss: {train_loss:.4f}  Train_accuracy: {train_accuracy:.4f}')
        # Evaluation phase
        val_loss, val_accuracy = eval_model(model, val_dataloader)
        print(f'Valid_loss: {val_loss:.4f}   Valid_accuracy: {val_accuracy:.4f}')
        # Save best model
        if val_accuracy > best_accuracy:
            print(f'Improve accuracy from {best_accuracy:.4f} -> {val_accuracy:.4f}')
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            save_path = os.path.join(save_model_ckpt, 'best_model.bin')
            torch.save(model.state_dict(), save_path)
    print(f'Training completed! \nBest val accuracy: {best_accuracy:.4f} at epoch {best_epoch}')


        