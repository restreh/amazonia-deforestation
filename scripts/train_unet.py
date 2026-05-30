"""Entrena la U-Net (encoder ResNet-34) sobre recortes, con perdida focal.

Uso (desde la raiz del repositorio, en una maquina con GPU):
    python scripts/train_unet.py

Construye recortes de 256x256 sobre la grilla, respetando la particion por bloques
mediante un peso por pixel (la perdida solo cuenta pixeles del split). Entrena con
perdida focal, valida cada epoch sobre los bloques de validacion (AUC-PR enmascarado)
y guarda el mejor modelo en models/unet.pt. Registra en MLflow si esta instalado.

Optimizador AdamW con weight_decay, scheduler ReduceLROnPlateau sobre val_AUC-PR y
early stopping configurables desde config.yaml. Aumentaciones (flips H/V y rot90)
sobre los recortes de entrenamiento. Mascaras de validez por trimestre (4 canales
extra) cuando validity_masks=true. Perdida combinada focal + Dice ponderada por
dice_weight (0 desactiva Dice y deja focal puro).

Dependencias: torch, segmentation-models-pytorch, scikit-learn, rasterio, numpy, pyyaml.
Instalar PyTorch con soporte CUDA segun pytorch.org.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from amazonia_deforestation.models.patches import (  # noqa: E402
    PatchDataset, build_patch_index, channel_spec, select_patches)
from amazonia_deforestation.models.unet import (  # noqa: E402
    build_unet, dice_loss, focal_loss, masked_scores)


def load_stats(stats_path, expected_n, expected_validity):
    """Carga channel_stats.json y devuelve (means, stds) como numpy float32.

    Verifica que el numero de canales y el flag de validez coincidan con la ejecucion
    actual. Falla con mensaje claro si no.
    """
    if not stats_path.exists():
        raise FileNotFoundError(
            "Falta " + str(stats_path) + ". Corre primero "
            "scripts/compute_channel_stats.py o pon standardize=false en config.yaml.")
    obj = json.loads(stats_path.read_text(encoding="utf-8"))
    if int(obj.get("n_channels", -1)) != expected_n:
        raise RuntimeError(
            "Inconsistencia en " + str(stats_path)
            + ": n_channels=" + str(obj.get("n_channels"))
            + " vs spec=" + str(expected_n)
            + ". Vuelve a correr compute_channel_stats.py.")
    if bool(obj.get("validity_masks", False)) != bool(expected_validity):
        raise RuntimeError(
            "Inconsistencia de validity_masks entre stats y config. "
            "Vuelve a correr compute_channel_stats.py.")
    means = np.asarray(obj["means"], dtype="float32")
    stds = np.asarray(obj["stds"], dtype="float32")
    return means, stds


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
    use_validity = bool(u.get("validity_masks", False))
    use_augment = bool(u.get("augment", False))
    use_standardize = bool(u.get("standardize", False))
    weight_decay = float(u.get("weight_decay", 0.0))
    lr_patience = int(u.get("lr_patience", 3))
    es_patience = int(u.get("early_stopping_patience", 8))
    dice_weight = float(u.get("dice_weight", 0.0))
    focal_weight = max(0.0, 1.0 - dice_weight)

    spec = channel_spec(config, comp, idx, validity_masks=use_validity)
    in_ch = len(spec)
    records, _ = build_patch_index(split, size, u["patch_stride"])
    tr = select_patches(records, "train", u["min_patch_pixels"])
    va = select_patches(records, "val", u["min_patch_pixels"])
    print("Canales: " + str(in_ch) + " (validez=" + str(use_validity)
          + ", estandarizado=" + str(use_standardize) + ")"
          + " | recortes train: " + str(len(tr)) + " | val: " + str(len(va)))

    stats = None
    if use_standardize:
        stats_path = ROOT / "data" / "interim" / "channel_stats.json"
        stats = load_stats(stats_path, in_ch, use_validity)
        print("Stats cargadas: " + str(stats_path))

    ds_tr = PatchDataset(config, comp, idx, label, split, tr, "train", size,
                         validity_masks=use_validity, augment=use_augment, stats=stats)
    ds_va = PatchDataset(config, comp, idx, label, split, va, "val", size,
                         validity_masks=use_validity, augment=False, stats=stats)
    dl_tr = DataLoader(ds_tr, batch_size=u["batch_size"], shuffle=True,
                       num_workers=u["num_workers"], drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=u["batch_size"], shuffle=False,
                       num_workers=u["num_workers"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo: " + device)
    model = build_unet(in_ch, u["encoder"], u["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=u["learning_rate"],
                            weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=lr_patience)

    try:
        import mlflow
        mlflow.set_tracking_uri(config["modeling"]["mlflow_tracking_uri"])
        mlflow.set_experiment("unet")
        run = mlflow.start_run(run_name="unet_resnet34")
        mlflow.log_params({"encoder": u["encoder"], "in_channels": in_ch,
                           "encoder_weights": str(u.get("encoder_weights")),
                           "batch_size": u["batch_size"], "lr": u["learning_rate"],
                           "weight_decay": weight_decay, "epochs": u["epochs"],
                           "alpha": fl["alpha"], "gamma": fl["gamma"],
                           "dice_weight": dice_weight,
                           "focal_weight": focal_weight,
                           "validity_masks": use_validity, "augment": use_augment,
                           "standardize": use_standardize,
                           "lr_patience": lr_patience, "es_patience": es_patience})
    except Exception:
        print("MLflow no disponible; se omite el registro.")
        mlflow = None

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best_ap = -1.0
    patience_left = es_patience
    for epoch in range(1, u["epochs"] + 1):
        model.train()
        running = 0.0
        running_focal = 0.0
        running_dice = 0.0
        for image, lab, w in dl_tr:
            image = image.to(device)
            target = lab.unsqueeze(1).to(device)
            weight = w.unsqueeze(1).to(device)
            opt.zero_grad()
            logits = model(image)
            l_focal = focal_loss(logits, target, weight, fl["alpha"], fl["gamma"])
            if dice_weight > 0.0:
                l_dice = dice_loss(logits, target, weight)
                loss = focal_weight * l_focal + dice_weight * l_dice
                running_dice += float(l_dice.item())
            else:
                loss = l_focal
            loss.backward()
            opt.step()
            running += float(loss.item())
            running_focal += float(l_focal.item())
        n_batches = max(1, len(dl_tr))
        train_loss = running / n_batches
        train_focal = running_focal / n_batches
        train_dice = running_dice / n_batches

        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for image, lab, w in dl_va:
                logits = model(image.to(device))
                p, t = masked_scores(logits.cpu(), lab.unsqueeze(1), w.unsqueeze(1))
                ps.append(p); ts.append(t)
        p = np.concatenate(ps); t = np.concatenate(ts)
        ap = float(average_precision_score(t, p)) if t.sum() > 0 else 0.0
        scheduler.step(ap)
        current_lr = opt.param_groups[0]["lr"]
        if dice_weight > 0.0:
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  focal {train_focal:.4f}"
                  f"  dice {train_dice:.4f}  val_AUC-PR {ap:.4f}  lr {current_lr:.2e}",
                  flush=True)
        else:
            print(f"epoch {epoch:3d}  focal_loss {train_loss:.4f}  val_AUC-PR {ap:.4f}"
                  f"  lr {current_lr:.2e}", flush=True)
        if mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_focal_loss", train_focal, step=epoch)
            if dice_weight > 0.0:
                mlflow.log_metric("train_dice_loss", train_dice, step=epoch)
            mlflow.log_metric("val_auc_pr", ap, step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
        if ap > best_ap:
            best_ap = ap
            patience_left = es_patience
            ckpt = {"state_dict": model.state_dict(), "in_channels": in_ch,
                    "encoder": u["encoder"], "validity_masks": use_validity,
                    "standardize": use_standardize}
            if stats is not None:
                ckpt["means"] = stats[0].tolist()
                ckpt["stds"] = stats[1].tolist()
            torch.save(ckpt, models_dir / "unet.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping en epoch " + str(epoch)
                      + " (sin mejora en " + str(es_patience) + " epocas)")
                break

    print("Mejor val AUC-PR: " + format(best_ap, ".4f") + " | modelo en " + str(models_dir / "unet.pt"))
    if mlflow:
        mlflow.log_metric("best_val_auc_pr", best_ap)
        mlflow.end_run()


if __name__ == "__main__":
    main()
