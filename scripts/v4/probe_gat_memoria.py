"""Barrido de memoria de GAT sobre el grafo completo de Elliptic.
Mide el pico de VRAM de un forward+backward (3 pasos, 2 capas) para varias
combinaciones de hidden x heads. Es la evidencia del recorte de GAT registrado en
runs_v4/DECISIONES.md (29-sep). Requiere GPU. Uso:
    uv run --frozen python scripts/v4/probe_gat_memoria.py
"""
import sys, yaml, torch
sys.path.insert(0, ".")
from src.data.loader import load_elliptic, apply_label_mode, print_dataset_stats
from src.data.preprocessing import preprocess
from src.training.trainer import build_model

cfg = yaml.safe_load(open("configs/experiment_v4.yaml"))
dev = "cuda"
data = load_elliptic(root=cfg["data"]["root"])
apply_label_mode(data, "licit_unknown")
preprocess(data, train_range=(1, 34), val_range=(35, 42), test_range=(43, 49))
x = data.x.to(dev); ei = data.edge_index.to(dev); y = data.y.to(dev)
tm = data.train_mask.to(dev)
print(f"grafo: x={tuple(x.shape)} edges={ei.shape[1]}")

def run(hidden, heads):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    try:
        model = build_model("GAT", in_channels=x.shape[1], hidden_channels=hidden,
                            num_layers=2, dropout=0.25, heads=heads).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(x, ei)
            loss = lossf(out[tm], y[tm].clamp(min=0))
            loss.backward()
            opt.step()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        del model, opt
        torch.cuda.empty_cache()
        return f"OK  pico={peak:.2f} GiB"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"

# de mayor a menor tamaño efectivo (hidden*heads)
for hidden, heads in [(148,8),(148,4),(128,4),(64,8),(128,2),(64,4),(64,2)]:
    print(f"hidden={hidden:>3} heads={heads}  (ancho efectivo {hidden*heads:>4}):", run(hidden, heads))
