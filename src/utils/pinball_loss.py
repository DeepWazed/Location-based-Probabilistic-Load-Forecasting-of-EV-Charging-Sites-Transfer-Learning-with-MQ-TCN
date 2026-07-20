import torch

class PinballLoss:
    def __init__(self, quantiles, reduction='mean'):
        self.quantiles = quantiles
        assert all(0 < q < 1 for q in quantiles), "Quantiles should be between 0 and 1"
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape[:-1] == target.shape, "Output and target shape mismatch"
        batch_size, seq_len, num_quantiles = output.shape
        total_loss = 0

        for i, q in enumerate(self.quantiles):
            error = output[:, :, i] - target
            loss = torch.zeros_like(target, dtype=torch.float)
            loss[error < 0] = q * (-error[error < 0])
            loss[error >= 0] = (1 - q) * error[error >= 0]
            if self.reduction == 'sum':
                total_loss += loss.sum()
            elif self.reduction == 'mean':
                total_loss += loss.mean()
            elif self.reduction == 'none':
                total_loss += loss

        if self.reduction == 'mean':
            total_loss = total_loss / len(self.quantiles)

        return total_loss