"""
Evaluate TruFor checkpoints fine-tuned on FakeFlickr.

For each (fake, real) pair this loads the corresponding fine-tuned checkpoint
produced by train_fakeflickr.sh (weights/trufor_ff_<short>/best.pth.tar) and
runs detection + localization metrics on the FakeFlickr test split — the same
test split that evaluate_trufor_zero_shot.py uses, so numbers are directly
comparable.

The trufor_ff_*.yaml configs only train the localization head, so the network
returns det=None. In that case we derive an image-level score from the
predicted manipulation map (max of the foreground softmax).

Outputs (next to this script):
    trufor_finetuned_results_<experiment>.csv         per-image results
    trufor_finetuned_evaluation_results.txt           aggregated metrics

Usage:
    python evaluate_trufor_finetuned.py
    python evaluate_trufor_finetuned.py --weights-root /path/to/weights
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset
from metrics import computeLocalizationMetrics


FF_ROOT = "/home/marek/FakeFlickr/data/fake-flickr"

DATASET_PATHS = {
    "real_rescaled":            f"{FF_ROOT}/real_rescaled",
    "flux_1_dev":               f"{FF_ROOT}/generated/flux_1_dev/img",
    "sd_3_5_large":             f"{FF_ROOT}/generated/sd_3_5_large/img",
    "flux_fill_real_rescaled":  f"{FF_ROOT}/generated/flux_fill_real_rescaled/img",
    "flux_fill_flux_1_dev":     f"{FF_ROOT}/generated/flux_fill_flux_1_dev/img",
    "flux_fill_sd_3_5_large":   f"{FF_ROOT}/generated/flux_fill_sd_3_5_large/img",
}

MASKS_PATHS = {
    "flux_fill_real_rescaled": f"{FF_ROOT}/masks/real_rescaled_dilated/masks",
    "flux_fill_flux_1_dev":    f"{FF_ROOT}/masks/flux_1_dev_dilated/masks",
    "flux_fill_sd_3_5_large":  f"{FF_ROOT}/masks/sd_3_5_large_dilated/masks",
}

TEST_LIST_PATH = "/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"

# (fake type, real type, training experiment that produced the checkpoint)
EXPERIMENTS = [
    {
        "name":       "flux_fill_real_rescaled__vs__real_rescaled",
        "fake":       "flux_fill_real_rescaled",
        "real":       "real_rescaled",
        "train_exp":  "trufor_ff_real_rescaled",
    },
    {
        "name":       "flux_fill_flux_1_dev__vs__flux_1_dev",
        "fake":       "flux_fill_flux_1_dev",
        "real":       "flux_1_dev",
        "train_exp":  "trufor_ff_flux_1_dev",
    },
    {
        "name":       "flux_fill_sd_3_5_large__vs__sd_3_5_large",
        "fake":       "flux_fill_sd_3_5_large",
        "real":       "sd_3_5_large",
        "train_exp":  "trufor_ff_sd_3_5_large",
    },
]


def read_test_ids():
    with open(TEST_LIST_PATH, "r") as f:
        return {line.strip() for line in f if line.strip()}


def get_dataset_files(dir_path, test_ids):
    files = sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    )
    return [os.path.join(dir_path, f) for f in files if os.path.splitext(f)[0] in test_ids]


def load_model(experiment_yaml, checkpoint_path, device):
    """Reload config + model for a given fine-tuned checkpoint."""
    args = argparse.Namespace(
        experiment=experiment_yaml,
        opts=['TEST.MODEL_FILE', checkpoint_path],
    )
    update_config(config, args)

    print(f'=> loading checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(config)
    state_dict = checkpoint['state_dict']
    # Tolerate missing det/conf heads in the FF configs.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  (missing keys: {len(missing)} — ok if loc-only config)')
    if unexpected:
        print(f'  (unexpected keys: {len(unexpected)})')
    model = model.to(device)
    model.eval()
    return model


def extract(model, list_img, device):
    dataset = TestDataset(list_img=list_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    results = []
    with torch.no_grad():
        for rgb, path in tqdm(loader, desc="Inference", leave=False):
            img_path = path[0]
            try:
                rgb = rgb.to(device)
                pred, conf, det, npp = model(rgb, save_np=False)

                pred_map = torch.squeeze(pred, 0)
                pred_map = F.softmax(pred_map, dim=0)[1].cpu().numpy()
                # Strip internal multiple-of-8 padding back to original size.
                h, w = rgb.shape[2:]
                pred_map = pred_map[:h, :w]

                if det is not None:
                    score = torch.sigmoid(det).item()
                else:
                    # Loc-only fine-tune — derive image-level score from the map.
                    score = float(pred_map.max())

                results.append({'path': img_path, 'score': score, 'map': pred_map})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return results


def evaluate_experiment(exp, weights_root, device, out_handle):
    ckpt = os.path.join(weights_root, exp['train_exp'], 'best.pth.tar')
    if not os.path.isfile(ckpt):
        alt = os.path.join(weights_root, exp['train_exp'], 'checkpoint.pth.tar')
        if os.path.isfile(alt):
            print(f"[{exp['name']}] best.pth.tar missing, falling back to checkpoint.pth.tar")
            ckpt = alt
        else:
            print(f"[{exp['name']}] SKIP — no checkpoint at {ckpt}")
            return

    model = load_model(exp['train_exp'], ckpt, device)

    test_ids = read_test_ids()
    real_files = get_dataset_files(DATASET_PATHS[exp['real']], test_ids)
    fake_files = get_dataset_files(DATASET_PATHS[exp['fake']], test_ids)
    print(f"[{exp['name']}] real={len(real_files)} fake={len(fake_files)}")
    if not real_files or not fake_files:
        print("  Skipping due to zero files.")
        return

    print("  Running real images...")
    real_results = extract(model, real_files, device)
    print("  Running fake images...")
    fake_results = extract(model, fake_files, device)

    # Detection metrics
    y_true   = [0] * len(real_results) + [1] * len(fake_results)
    y_scores = [r['score'] for r in real_results] + [r['score'] for r in fake_results]
    y_pred   = [1 if s > 0.5 else 0 for s in y_scores]

    test_auc          = roc_auc_score(y_true, y_scores)
    test_acc          = accuracy_score(y_true, y_pred)
    precision         = precision_score(y_true, y_pred, zero_division=0)
    recall            = recall_score(y_true, y_pred, zero_division=0)
    average_precision = average_precision_score(y_true, y_scores)
    mcc               = matthews_corrcoef(y_true, y_pred)

    # Localization metrics on fakes
    mask_dir = MASKS_PATHS[exp['fake']]
    f1_best_l, f1_th_l, iou_best_l, iou_th_l = [], [], [], []
    image_records = []

    print("  Computing localization metrics...")
    for res in tqdm(fake_results, desc="Localization", leave=False):
        base_id = os.path.splitext(os.path.basename(res['path']))[0]
        mask_path = os.path.join(mask_dir, f"mask_{base_id}.png")
        f1_best = f1_th = iou_best = iou_th = float('nan')

        if os.path.exists(mask_path):
            gt = np.array(Image.open(mask_path).convert('L'))
            gt = (gt > 25).astype(bool)
            pmap = res['map']
            if pmap.shape != gt.shape:
                import cv2
                pmap = cv2.resize(pmap, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            f1_best, f1_th, iou_best, iou_th = computeLocalizationMetrics(pmap, gt)
            if not np.isnan(f1_best):  f1_best_l.append(f1_best)
            if not np.isnan(f1_th):    f1_th_l.append(f1_th)
            if not np.isnan(iou_best): iou_best_l.append(iou_best)
            if not np.isnan(iou_th):   iou_th_l.append(iou_th)

        image_records.append({
            'path': res['path'], 'label': 1, 'det_score': res['score'],
            'f1_best': f1_best, 'f1_th_0.5': f1_th,
            'iou_best': iou_best, 'iou_th_0.5': iou_th,
        })

    for res in real_results:
        image_records.append({
            'path': res['path'], 'label': 0, 'det_score': res['score'],
            'f1_best': float('nan'), 'f1_th_0.5': float('nan'),
            'iou_best': float('nan'), 'iou_th_0.5': float('nan'),
        })

    csv_path = os.path.join(_HERE, f"trufor_finetuned_results_{exp['name']}.csv")
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=['path', 'label', 'det_score', 'f1_best', 'f1_th_0.5', 'iou_best', 'iou_th_0.5'],
        )
        writer.writeheader()
        writer.writerows(image_records)

    loc_f1_best = np.mean(f1_best_l)  if f1_best_l  else float('nan')
    loc_f1_th   = np.mean(f1_th_l)    if f1_th_l    else float('nan')
    loc_iou_best = np.mean(iou_best_l) if iou_best_l else float('nan')
    loc_iou_th   = np.mean(iou_th_l)   if iou_th_l   else float('nan')

    report = (
        f"Dataset: {exp['name']}\n"
        f"  checkpoint: {ckpt}\n"
        f"  test_acc:               {test_acc:.4f}\n"
        f"  test_auc:               {test_auc:.4f}\n"
        f"  precision:              {precision:.4f}\n"
        f"  recall:                 {recall:.4f}\n"
        f"  average_precision:      {average_precision:.4f}\n"
        f"  mcc:                    {mcc:.4f}\n"
        f"  localization_f1_best:   {loc_f1_best:.4f}\n"
        f"  localization_f1_th_0.5: {loc_f1_th:.4f}\n"
        f"  localization_iou_best:  {loc_iou_best:.4f}\n"
        f"  localization_iou_th_0.5:{loc_iou_th:.4f}\n"
    )
    print(report)
    out_handle.write(report + "\n")
    out_handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-root', default=os.path.join(_HERE, 'weights'),
                        help='Root containing trufor_ff_<variant>/best.pth.tar')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', default=os.path.join(_HERE, 'trufor_finetuned_evaluation_results.txt'))
    cli = parser.parse_args()

    device = f'cuda:{cli.gpu}' if cli.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    with open(cli.out, 'w') as out_f:
        for exp in EXPERIMENTS:
            print(f"\n=== Evaluating {exp['name']} ===")
            evaluate_experiment(exp, cli.weights_root, device, out_f)

    print(f'Done. Results saved to {cli.out}')


if __name__ == '__main__':
    main()
