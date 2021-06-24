import torch

def batch_dot(x, y):
    """
    x: tensor, one row is a vec
    y: tensor, one row is a vec
    """
    return (x*y).sum(-1)
    # return torch.div((x*y).sum(-1), torch.norm(x, p=2, dim=-1) * torch.norm(y, p=2, dim=-1))




def get_last_masked_tensor(tensor, mask):
    """
    tensor: batch, length, dim
    """
    device = tensor.device
    length = torch.sum(mask, dim=1)
    length_mask = torch.zeros(mask.shape, dtype=int).to(device)
    for i in range(length_mask.shape[0]):
        length_mask[i][int(length[i]-1)] = 1
    length_mask = length_mask.unsqueeze(-1)
    return torch.sum(length_mask * tensor, dim=1)