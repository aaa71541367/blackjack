"""相容舊版 PyTorch（例如搭配 Python 3.7 的版本）。

- torch.load 的 weights_only 參數在 PyTorch 1.13 才加入；2.6 起預設為 True，
  而本專案的存檔含有設定字典，需要 weights_only=False
- F.smooth_l1_loss 的 beta 參數在 PyTorch 1.7 才加入
"""
import torch


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                      # 舊版沒有 weights_only 參數
        return torch.load(path, map_location=map_location)


def huber(x, y, beta: float = 1.0):
    """Huber loss（等同新版 F.smooth_l1_loss(x, y, beta=beta)）。"""
    d = (x - y).abs()
    return torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean()
