import torch

def check_gpu_avaible():
    """Checking the gpu status and returns the torch device if a gpu on
    mac or cuda is avaible.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on MPS. M1 GPU is avaible")
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on Nvidia. Cuda is avaible")
        
    else:
        device = torch.device("cpu")
        print("Running on CPU")
        
    return device
