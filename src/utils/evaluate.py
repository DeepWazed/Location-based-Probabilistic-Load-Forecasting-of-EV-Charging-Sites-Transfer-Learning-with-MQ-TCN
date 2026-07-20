import torch

def evaluate_model(device, model, data_loader, loss_fn):
    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.permute(0, 2, 1))
            loss = loss_fn(outputs, targets)
            epoch_loss.append(loss.item())
    avg_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_loss