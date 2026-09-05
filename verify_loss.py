import torch
from src.losses.nt_xent import nt_xent_loss

torch.manual_seed(42)

print("--- Step 7: Basic Correctness Tests ---")
N, D = 4, 16
v = torch.eye(N, D)

# Case 1: Identical positive pairs
z_case1 = torch.cat([v, v], dim=0)
loss1 = nt_xent_loss(z_case1, temperature=0.1)
print(f"Case 1 Loss (Identical): {loss1.item():.4f}")
print("Explanation: Identical pairs yield maximum similarity, driving loss close to 0.\n")

# Case 2: Opposite positive pairs
z_case2 = torch.cat([v, -v], dim=0)
loss2 = nt_xent_loss(z_case2, temperature=0.1)
print(f"Case 2 Loss (Opposite): {loss2.item():.4f}")
print("Explanation: Inverted pairs produce minimum similarity, pushing loss very high.\n")

print("--- Step 8: Sanity-Check Edge Cases ---")

# Edge Case 1: N = 1 (2 rows total)
N_edge = 1
z_n1 = torch.randn(2 * N_edge, D)
loss_n1 = nt_xent_loss(z_n1, temperature=0.5)
print(f"N=1 Batch Loss: {loss_n1.item():.4f}")
print("Observation: With N=1, the only candidate other than the anchor itself is its true positive pair. After masking the diagonal, the softmax probability for the positive match is 1.0, so the loss is exactly 0.0 without index/shape errors.\n")

# Edge Case 2: Temperature Scaling (tau = 0.1 vs tau = 1.0)
# Create a realistic scenario with imperfect alignment
z_noisy = torch.randn(8, D)

loss_tau_low = nt_xent_loss(z_noisy, temperature=0.1)
loss_tau_high = nt_xent_loss(z_noisy, temperature=1.0)

print(f"Loss at temperature=0.1: {loss_tau_low.item():.4f}")
print(f"Loss at temperature=1.0: {loss_tau_high.item():.4f}")
print("Observation: Lower temperature (0.1) magnifies similarity discrepancies in the denominator, heavily penalizing hard negatives and driving up loss magnitude when pairs are not perfectly aligned.")