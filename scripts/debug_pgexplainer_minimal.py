"""
Minimal PGExplainer reproducer following PyG official docs.

Tests multiple configurations to isolate whether our wrapper is broken
or PyG 2.7 PGExplainer itself fails on Cora.
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

# Load Cora
dataset = Planetoid(root="./data/cora", name="Cora")
data = dataset[0].to(DEVICE)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


# Train model
model = GCN(data.num_node_features, 64, int(data.y.max().item())+1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
model.eval()

# ═══ Test multiple PGExplainer configurations ═══
configs_to_try = [
    {"epochs": 30, "lr": 0.003, "train_nodes": 50, "name": "our_default"},
    {"epochs": 30, "lr": 0.01,  "train_nodes": 50, "name": "higher_lr"},
    {"epochs": 100, "lr": 0.003, "train_nodes": 50, "name": "more_epochs"},
    {"epochs": 30, "lr": 0.003, "train_nodes": 140, "name": "all_train_nodes"},
    {"epochs": 30, "lr": 0.003, "train_nodes": 50, "name": "long_target", "long_target": True},
]

for cfg in configs_to_try:
    print(f"\n{'='*70}\nCONFIG: {cfg['name']} (epochs={cfg['epochs']}, lr={cfg['lr']}, train_nodes={cfg['train_nodes']})")
    print('='*70)

    # Build explainer
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=cfg["epochs"], lr=cfg["lr"]),
        explanation_type="phenomenon",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    explainer.algorithm.to(DEVICE)

    # Prepare target
    if cfg.get("long_target"):
        target = data.y.long().to(DEVICE)
    else:
        target = data.y.to(DEVICE)

    # Train PGExplainer
    train_nodes = torch.where(data.train_mask)[0]
    n_train = cfg["train_nodes"]
    total_loss = 0.0
    nan_count = 0
    for epoch in range(cfg["epochs"]):
        for idx in train_nodes[:n_train]:
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, data.x, data.edge_index,
                target=target, index=idx.item(),
            )
            if torch.is_tensor(step_loss):
                step_loss = step_loss.item()
            if not (step_loss == step_loss):
                nan_count += 1
            else:
                total_loss += step_loss
    avg_loss = total_loss / max(1, cfg["epochs"] * n_train - nan_count)
    print(f"  Train complete. avg_loss={avg_loss:.4f}, nan_steps={nan_count}")

    # Inspect weights of PGExplainer's parametric network
    for name, p in explainer.algorithm.named_parameters():
        print(f"  weight {name}: shape={list(p.shape)}, mean={p.mean().item():.4f}, std={p.std().item():.4f}")

    # Generate explanation for 1 test node
    test_idx = torch.where(data.test_mask)[0][0].item()
    try:
        explanation = explainer(data.x, data.edge_index, index=test_idx, target=target)
        em = explanation.edge_mask.cpu().detach().numpy()
        print(f"  Node {test_idx} edge_mask: min={em.min():.4f}, max={em.max():.4f}, "
              f"mean={em.mean():.4f}, std={em.std():.4f}, %nonzero={(em>1e-6).mean()*100:.1f}%")

        # Top-K edges
        import numpy as np
        top_k_indices = np.argsort(em)[::-1][:10]
        print(f"  Top-10 edge_mask values: {[f'{em[i]:.4f}' for i in top_k_indices]}")

    except Exception as exc:
        print(f"  ERROR: {exc}")

print("\nDONE — compare masks across configs to isolate the issue.")
