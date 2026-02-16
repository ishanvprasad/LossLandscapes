import torch
from pyhessian import hessian

def compute_hessian_metrics(model, dataloader, device="cuda"):
    """
    Computes the top eigenvalue and trace of the model's loss landscape.
    """
    model.eval()
    
    # PyHessian requires a specific closure to compute gradients
    def criterion(outputs, labels):
        # HuggingFace CausalLM outputs the loss directly if labels are provided
        return outputs.loss

    # We only need a small batch (e.g., 200 samples) to estimate the Hessian
    batch = next(iter(dataloader))
    inputs = {k: v.to(device) for k, v in batch.items()}
    
    # Initialize PyHessian
    # It requires the model, the loss function, and the data tuple
    hessian_comp = hessian(
        model, 
        criterion, 
        data=(inputs, inputs["labels"]), 
        cuda=(device != "cpu")
    )
    
    # 1. Compute the top eigenvalue (sharpness)
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
    
    # 2. Compute the trace (overall curvature) using Hutchinson's method
    trace = hessian_comp.trace()
    
    return top_eigenvalues[0], np.mean(trace)