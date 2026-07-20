import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error


def pinball_loss(outputs, targets, quantiles, reduction="mean"):
    assert len(quantiles) == outputs.shape[-1], "Number of quantiles should match number of outputs"
    assert outputs.shape[:-1] == targets.shape, "Output and target shape mismatch"
    loss = 0
    for i, q in enumerate(quantiles):
        err = targets - outputs[:, :, i]
        loss += torch.mean(torch.max((q-1) * err, q * err))
    if reduction == "mean":
        loss = loss / len(quantiles)
    elif reduction == "sum":
        pass
    elif reduction == "none":
        return loss

    return loss


def quantile_tcn_train(device, model, optimizer, quantiles, reduction, train_loader, val_loader, epochs, patience, filename=None):
    results = []
    train_loss = []
    val_loss = []

    best_val_loss = float('inf')
    quantiles = quantiles
    reduction = reduction
    counter = 0
    model.to(device)

    print("🚀 Training started 🚀\n")


    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        for x_con_batch, x_cat_batch, y_batch in train_loader:
            x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            # print(x_con_batch.shape, x_cat_batch.shape, y_batch.shape)
            optimizer.zero_grad()
            outputs = model(x_con_batch, x_cat_batch)
            loss = pinball_loss(outputs, y_batch, quantiles, reduction)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_loss.append(avg_train_loss)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_con_batch, x_cat_batch, y_batch in val_loader:
                x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                outputs = model(x_con_batch, x_cat_batch)
                loss = pinball_loss(outputs, y_batch, quantiles, reduction)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs, stopping training.\n')
            break

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(model.state_dict(), filename)

    return train_loss, val_loss
