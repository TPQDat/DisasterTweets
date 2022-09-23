import torch
class CFG:
    max_length = 256
    batch_size = 8
    hidden_prob_drop = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = 0.5
    lr = 3e-5
    num_epochs = 15
    seed = 42
    pass