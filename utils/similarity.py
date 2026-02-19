import torch

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def linear_HSIC(X, Y):
    """Hilbert-Schmidt Independence Criterion"""
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    """
    Computes Linear CKA between two activation matrices.
    X and Y should be shape: (batch_size, hidden_dim)
    """
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return (hsic / (var1 * var2)).item()

def extract_and_compare(model_1, model_2, inputs, device="cuda"):
    model_1.eval()
    model_2.eval()
    
    with torch.no_grad():
        out_1 = model_1(**inputs, output_hidden_states=True)
        out_2 = model_2(**inputs, output_hidden_states=True)
        
        # FIX 3: MEAN POOL across the sequence length (dim=1)
        # Shape changes from (batch, seq_len, hidden) -> (batch, hidden)
        act_1 = out_1.hidden_states[-1].mean(dim=1)
        act_2 = out_2.hidden_states[-1].mean(dim=1)
        
        cka_score = linear_CKA(act_1, act_2)
        
    return cka_score