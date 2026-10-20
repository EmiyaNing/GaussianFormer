"""Class-conditioned empty correction, applied after a frozen global bias."""
import torch
from torch import nn


class LearnableClassEmptyBias(nn.Module):
    def __init__(self, original_bias, num_classes=17, band=2.):
        super().__init__()
        self.register_buffer('original_bias', torch.tensor(float(original_bias), dtype=torch.float64))
        self.class_bias = nn.Parameter(torch.full((num_classes,), float(original_bias), dtype=torch.float64))
        self.num_classes = num_classes
        self.band = float(band)

    def forward(self, scores):
        """Scores are B x (C+1) x ..., already corrected by original_bias.

        No labels or camera mask are accepted. Routing and candidate membership
        use the unmodified input; foreground channels retain their ordering.
        """
        foreground, winner = scores[:, :self.num_classes].max(dim=1)
        empty = scores[:, self.num_classes]
        eligible = (foreground - empty).abs() <= self.band
        adjustment = self.class_bias[winner] - self.original_bias
        empty = empty - torch.where(eligible, adjustment, torch.zeros_like(adjustment))
        return torch.cat((scores[:, :self.num_classes], empty.unsqueeze(1)), dim=1)

    def shrinkage(self):
        return (self.class_bias - self.class_bias.mean()).square().mean()

    def predict(self, scores):
        """Compare the native margin in float64, avoiding bias-rounding ties."""
        foreground, winner = scores[:, :self.num_classes].max(dim=1)
        margin = foreground - scores[:, self.num_classes]
        adjustment = self.class_bias[winner] - self.original_bias
        adjusted = margin.double() + adjustment
        occupied = torch.where(margin.abs() <= self.band, adjusted >= 0, margin >= 0)
        return torch.where(occupied, winner, torch.full_like(winner, self.num_classes))
