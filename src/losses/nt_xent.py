import torch
import torch.nn.functional as F

def nt_xent_loss(z: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes the Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.

    Args:
        z: Tensor of shape (2N, D) where:
           - z[:N] are view 1 embeddings
           - z[N:] are view 2 embeddings
           - row i pairs with row i + N
        temperature: Scaling factor for cosine similarities. Default: 0.5.

    Returns:
        Scalar loss tensor.
    """
    z = F.normalize(z, dim=1)
    batch_size = z.shape[0]
    sim_matrix = torch.matmul(z, z.T) / temperature

    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    sim_matrix[mask] = -float("inf")

    N = batch_size // 2
    labels = torch.cat([torch.arange(N, 2 * N, device=z.device), torch.arange(0, N, device=z.device)])

    loss = F.cross_entropy(sim_matrix, labels)
    return loss
