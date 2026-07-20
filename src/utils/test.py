import numpy as np
import torch
import torch.nn as nn

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

def test(device, model, test_loader, loss_fn):
    test_loss = 0.0
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.permute(0, 2, 1))
            loss = loss_fn(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            preds.append(outputs.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}\n')
    return test_loss, preds
    
    
# def quantile_test(device, model, test_loader, quantiles):
#     test_loss = 0.0
#     quantiles = quantiles
#     preds = []
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs.permute(0, 2, 1))
#             loss = pinball_loss(outputs, targets, quantiles)
#             test_loss += loss.item() * inputs.size(0)
#             preds.append(outputs.cpu().numpy())
#     preds = np.concatenate(preds, axis=0)
#     test_loss /= len(test_loader.dataset)
#     print(f'Test Loss: {test_loss:.4f}')
#     return test_loss, preds

def quantile_test(device, model, test_loader, quantiles):
    test_losses = []  
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.permute(0, 2, 1))
            loss = pinball_loss(outputs, targets, quantiles)
            test_loss = loss.item() * inputs.size(0)
            test_losses.append(test_loss)  
            preds.append(outputs.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    test_loss = np.sum(test_losses) / len(test_loader.dataset)
    test_losses = [loss / len(test_loader) for loss in test_losses]
    print(f'Test Loss: {test_loss:.4f}')
    print(f"Individual Quantile Loss: {test_losses}")
    return test_loss, test_losses, preds


def test_task_tcn(device, model, test_loader, loss_fn):
    test_loss = 0.0
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x_con_batch, x_cat_batch, y_batch in test_loader:
            x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            outputs = model(x_con_batch, x_cat_batch)
            loss = loss_fn(outputs, y_batch)
            test_loss += loss.item() * x_con_batch.size(0)
            preds.append(outputs.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}\n')
    return test_loss, preds
