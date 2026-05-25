"""Entrena la U-Net (encoder ResNet-34) sobre recortes, con perdida focal.

Uso (desde la raiz del repositorio, en una maquina con GPU):
    python scripts/train_unet.py

Construye recortes de 256x256 sobre la grilla, respetando la particion por bloques
mediante un peso por pixel (la perdida solo cuenta pixeles del split). Entrena con
perdida focal, valida cada epoch sobre los bloques de validacion (AUC-PR enmascarado)
y guarda el mejor modelo en models/unet.pt. Registra en MLflow si esta instalado.

Dependencias: torch, segmentation-models-pytorch, scikit-learn, rasterio, numpy, pyyaml.
Instalar PyTorch con soporte CUDA segun pytorch.org.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.models.patches import (  # noqa: E402
    PatchDataset, build_patch_index, channel_spec, select_patches)
from amazonia_deforestation.models.unet import build_unet, focal_loss, masked_scores  # noqa: E402


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score

    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    u = config["modeling"]["unet"]
    fl = config["modeling"]["focal_loss"]
    comp = ROOT / "data" / "processed" / "composites"
    idx = ROOT / "data" / "processed" / "indices"
    label = ROOT / "data" / "interim" / "label_2024_20m.tif"
    split = ROOT / "data" / "interim" / "split_blocks.tif"
    size = u["window_size"]

    spec = channel_spec(config, comp, idx)
    in_ch = len(spec)
    records, _ = build_patch_index(split, size, u["patch_stride"])
    tr = select_patches(records, "train", u["min_patch_pixels"])
    va = select_patches(records, "val", u["min_patch_pixels"])
    print("Canales: " + str(in_ch) + " | recortes train: " + str(len(tr)) + " | val: " + str(len(va)))

    ds_tr = PatchDataset(config, comp, idx, label, split, tr, "train", size)
    ds_va = PatchDataset(config, comp, idx, label, split, va, "val", size)
    dl_tr = DataLoader(ds_tr, batch_size=u["batch_size"], shuffle=True,
                       num_workers=u["num_workers"], drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=u["batch_size"], shuffle=False,
                       num_workers=u["num_workers"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo: " + device)
    model = build_unet(in_ch, u["encoder"], u["encoder_weights"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=u["learning_rate"])

    try:
        import mlflow
        mlflow.set_tracking_uri(config["modeling"]["mlflow_tracking_uri"])
        mlflow.set_experiment("unet")
        run = mlflow.start_run(run_name="unet_resnet34")
        mlflow.log_params({"encoder": u["encoder"], "in_channels": in_ch,
                           "batch_size": u["batch_size"], "lr": u["learning_rate"],
                           "epochs": u["epochs"], "alpha": fl["alpha"], "gamma": fl["gamma"]})
    except Exception:
        print("MLflow no disponible; se omite el registro.")
        mlflow = None

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best_ap = -1.0
    for epoch in range(1, u["epochs"] + 1):
        model.train()
        running = 0.0
        for image, lab, w in dl_tr:
            image = image.to(device)
            target = lab.unsqueeze(1).to(device)
            weight = w.unsqueeze(1).to(device)
            opt.zero_grad()
            logits = model(image)
            loss = focal_loss(logits, target, weight, fl["alpha"], fl["gamma"])
            loss.backward()
            opt.step()
            running += float(loss.item())
        train_loss = running / max(1, len(dl_tr))

        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for image, lab, w in dl_va:
                logits = model(image.to(device))
                p, t = masked_scores(logits.cpu(), lab.unsqueeze(1), w.unsqueeze(1))
                ps.append(p); ts.append(t)
        p = np.concatenate(ps); t = np.concatenate(ts)
        ap = float(average_precision_score(t, p)) if t.sum() > 0 else 0.0
        print(f"epoch {epoch:3d}  focal_loss {train_loss:.4f}  val_AUC-PR {ap:.4f}", flush=True)
        if mlflow:
            mlflow.log_metric("train_focal_loss", train_loss, step=epoch)
            mlflow.log_metric("val_auc_pr", ap, step=epoch)
        if ap > best_ap:
            best_ap = ap
            torch.save({"state_dict": model.state_dict(), "in_channels": in_ch,
                        "encoder": u["encoder"]}, models_dir / "unet.pt")

    print("Mejor val AUC-PR: " + format(best_ap, ".4f") + " | modelo en " + str(models_dir / "unet.pt"))
    if mlflow:
        mlflow.log_metric("best_val_auc_pr", best_ap)
        mlflow.end_run()


if __name__ == "__main__":
    main()
