import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from tqdm import tqdm
import os

best_val_loss = float('inf')

def train(device, model, optimizer, loss_fn, train_loader, val_loader, epochs, patience, filename=None):
    results = []
    train_loss = []
    val_loss = []
    best_val_loss = np.inf
    counter = 0  
    
    print("Training started........................\n")
    
    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss.append(sum(train_losses) / len(train_losses))

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.permute(0, 2, 1))
                loss = loss_fn(outputs, targets)
                val_losses.append(loss.item())
        val_loss.append(sum(val_losses) / len(val_losses))

        
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            counter = 0  
        else:
            counter += 1  

        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs, stopping training.\n')
            break

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}')

        results.append((epoch, train_loss[-1], val_loss[-1]))

    print("\n")

    if filename is not None:
        with open(filename, 'wb') as f:
            pkl.dump(results, f)
    print(f"Train and validation losses have been saved to: {filename}\n")

    return train_loss, val_loss


# def pinball_loss(outputs, targets, quantiles):
    
#     assert len(quantiles) == outputs.shape[-1], "Number of quantiles should match number of outputs"
#     assert outputs.shape[:-1] == targets.shape, "Output and target shape mismatch"
#     loss = 0
#     # print(outputs.shape, targets.shape, len(quantiles))
#     for i, q in enumerate(quantiles):
#         err = targets - outputs[:,:,i]
#         a = torch.max((q-1) * err, q * err)
#         print()
#         loss += torch.mean(torch.max((q-1) * err, q * err))
#     return loss / len(quantiles)


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
    best_val_loss = np.inf
    quantiles = quantiles
    reduction = reduction
    counter = 0 
    
    print("Training started........................\n")
    
    model.to(device)
    
    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))
            loss = pinball_loss(outputs, targets, quantiles, reduction)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss.append(sum(train_losses) / len(train_losses))

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.permute(0, 2, 1))
                loss = pinball_loss(outputs, targets, quantiles, reduction)
                val_losses.append(loss.item())
        val_loss.append(sum(val_losses) / len(val_losses))

        
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            counter = 0 
        else:
            counter += 1  

        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs, stopping training.\n')
            break

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}')

        results.append((epoch, train_loss[-1], val_loss[-1]))

    print("\n")

    if filename is not None:
        with open(filename, 'wb') as f:
            pkl.dump(results, f)
    print(f"Train and validation losses have been saved to: {filename}\n")

    return train_loss, val_loss


def train_task_tcn(device, model, optimizer, loss_fn, train_loader, val_loader, epochs, patience, filename=None):
    global best_val_loss

    results = []
    train_loss = []
    val_loss = []
    no_improvement_count = 0

    for epoch in tqdm(range(epochs), desc="Training", dynamic_ncols=True, leave=False):
        model.train()
        epoch_train_loss = []
        for x_con_batch, x_cat_batch, y_batch in train_loader:
            x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_con_batch, x_cat_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_loss.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for x_con_batch, x_cat_batch, y_batch in val_loader:
                x_con_batch, x_cat_batch, y_batch = x_con_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                outputs = model(x_con_batch, x_cat_batch)
                loss = loss_fn(outputs, y_batch)
                epoch_val_loss.append(loss.item())
        avg_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            # Save the best model
            best_val_loss = avg_val_loss
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            torch.save(model.state_dict(), filename)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping after {patience} epochs of no improvement in validation loss.")
            break

    return train_loss, val_loss