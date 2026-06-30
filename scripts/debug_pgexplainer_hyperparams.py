"""Test PGExplainer with different edge_size regularization coefficients.

Hypothesis: default edge_size=0.05 creates a local minimum at mask=0 (mode collapse).
Reducing edge_size should allow the mask to learn non-trivial values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, PGExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

dataset = Planetoid(root="./data/cora", name="Cora")
data = dataset[0].to(DEVICE)


class GCN(torch.nn.Module):
    def __init__(self, in_c, hidden_c, out_c):
        super().__init__()
        self.c1 = GCNConv(in_c, hidden_c)
        self.c2 = GCNConv(hidden_c, out_c)
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, edge_index)


model = GCN(data.num_node_features, 64, int(data.y.max().item())+1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for _ in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
model.eval()
print(f"Model trained. Test accuracy: {(model(data.x, data.edge_index).argmax(-1)[data.test_mask] == data.y[data.test_mask]).float().mean():.4f}\n")


# Test different regularization strengths
configs = [
    {"edge_size": 0.05, "edge_ent": 1.0, "name": "DEFAULT_PyG"},
    {"edge_size": 0.01, "edge_ent": 1.0, "name": "lower_edge_size_0.01"},
    {"edge_size": 0.005, "edge_ent": 1.0, "name": "much_lower_0.005"},
    {"edge_size": 0.001, "edge_ent": 1.0, "name": "very_low_0.001"},
    {"edge_size": 0.0001, "edge_ent": 1.0, "name": "near_zero_0.0001"},
    {"edge_size": 0.05, "edge_ent": 0.1, "name": "default_size_low_ent"},
    {"edge_size": 0.005, "edge_ent": 0.1, "name": "low_both"},
]

for cfg in configs:
    print(f"{'='*70}")
    print(f"CONFIG: {cfg['name']}")
    print(f"  edge_size={cfg['edge_size']}, edge_ent={cfg['edge_ent']}")
    print('='*70)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003,
                              edge_size=cfg["edge_size"], edge_ent=cfg["edge_ent"]),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    explainer.algorithm.to(DEVICE)

    target = data.y.to(DEVICE)
    train_nodes = torch.where(data.train_mask)[0]
    losses = []
    for epoch in range(30):
        for idx in train_nodes[:50]:
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, data.x, data.edge_index,
                target=target, index=idx.item(),
            )
            if torch.is_tensor(step_loss): step_loss = step_loss.item()
            if step_loss == step_loss:  # not NaN
                losses.append(step_loss)
    print(f"  Final avg loss: {sum(losses[-50:])/50:.4f}")

    # Explain 3 test nodes to check mask diversity
    test_nodes = torch.where(data.test_mask)[0][:3]
    for test_idx in test_nodes:
        try:
            explanation = explainer(data.x, data.edge_index, index=test_idx.item(), target=target)
            em = explanation.edge_mask.cpu().detach().numpy()
            pct_nonzero = (em > 1e-4).mean() * 100
            print(f"  Node {test_idx.item()}: mask max={em.max():.4f}, mean={em.mean():.4f}, "
                  f"std={em.std():.4f}, %nonzero={pct_nonzero:.1f}%")
        except Exception as exc:
            print(f"  Error: {exc}")
    print()

print("\nVERDICT: look for config where mask max > 0.1 AND std > 0.01")
