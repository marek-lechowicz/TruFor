import sys, os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, matthews_corrcoef

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TruFor_train_test')
if path not in sys.path:
    sys.path.insert(0, path)

from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset
import argparse
from metrics import computeLocalizationMetrics

gpu = 0
device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'

DATASET_PATHS = {
    "real_rescaled": "/home/marek/FakeFlickr/data/fake-flickr/real_rescaled",
    "flux_1_dev": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_1_dev/img",
    "sd_3_5_large": "/home/marek/FakeFlickr/data/fake-flickr/generated/sd_3_5_large/img",
    "flux_fill_real_rescaled": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_real_rescaled/img",
    "flux_fill_flux_1_dev": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_flux_1_dev/img",
    "flux_fill_sd_3_5_large": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_sd_3_5_large/img"
}

MASKS_PATHS = {
    "flux_fill_real_rescaled": "/home/marek/FakeFlickr/data/masks_entities_ensemble/real_rescaled_dilated/masks",
    "flux_fill_flux_1_dev": "/home/marek/FakeFlickr/data/masks_entities_ensemble/flux_1_dev_dilated/masks",
    "flux_fill_sd_3_5_large": "/home/marek/FakeFlickr/data/masks_entities_ensemble/sd_3_5_large_dilated/masks"
}

TEST_LIST_PATH = "/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"

def read_test_ids():
    with open(TEST_LIST_PATH, "r") as f:
        return {line.strip() for line in f if line.strip()}

test_ids = read_test_ids()

def get_dataset_files(dir_path):
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    test_files = []
    for f in files:
        base_id = os.path.splitext(f)[0]
        if base_id in test_ids:
            test_files.append(os.path.join(dir_path, f))
    return test_files

# Setup configuration
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args(['TEST.MODEL_FILE', '/home/marek/projects/fake_flickr_sota/TruFor/test_docker/weights/trufor.pth.tar'])
update_config(config, args)

print('=> loading TruFor model...')
checkpoint = torch.load(config.TEST.MODEL_FILE, map_location=device)
model = get_model(config)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()

experiments = [
    {"name": "flux_fill_real_rescaled__vs__real_rescaled", "fake": "flux_fill_real_rescaled", "real": "real_rescaled"},
    {"name": "flux_fill_flux_1_dev__vs__flux_1_dev", "fake": "flux_fill_flux_1_dev", "real": "flux_1_dev"},
    {"name": "flux_fill_sd_3_5_large__vs__sd_3_5_large", "fake": "flux_fill_sd_3_5_large", "real": "sd_3_5_large"}
]

def extract_trufor(list_img):
    dataset = TestDataset(list_img=list_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    results = []
    
    with torch.no_grad():
        for index, (rgb, path) in enumerate(tqdm(loader, desc="Inference", leave=False)):
            img_path = path[0]
            try:
                rgb = rgb.to(device)
                pred, conf, det, npp = model(rgb, save_np=False)
                
                det_sig = torch.sigmoid(det).item()
                pred_map = torch.squeeze(pred, 0)
                pred_map = F.softmax(pred_map, dim=0)[1].cpu().numpy()
                
                # Model pads internally to multiple of 8, extract original size
                h, w = rgb.shape[2:]
                pred_map = pred_map[:h, :w]
                
                results.append({
                    'path': img_path,
                    'score': det_sig,
                    'map': pred_map
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return results

print('Starting evaluation loop...')
with open("trufor_evaluation_results.txt", "w") as out_f:
    for exp in experiments:
        print(f"\\nEvaluating Experiment: {exp['name']}")
        
        real_files = get_dataset_files(DATASET_PATHS[exp['real']])
        fake_files = get_dataset_files(DATASET_PATHS[exp['fake']])
        
        print(f"Loaded {len(real_files)} real and {len(fake_files)} fake images.")
        if len(real_files) == 0 or len(fake_files) == 0:
            print("Skipping due to zero files.")
            continue
            
        # Inference
        print("Processing Real images...")
        real_results = extract_trufor(real_files)
        print("Processing Fake images...")
        fake_results = extract_trufor(fake_files)
        
        # Detection metrics
        y_true = [0] * len(real_results) + [1] * len(fake_results)
        y_scores = [r['score'] for r in real_results] + [r['score'] for r in fake_results]
        
        y_pred = [1 if s > 0.5 else 0 for s in y_scores]
        
        test_auc = roc_auc_score(y_true, y_scores)
        test_acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        average_precision = average_precision_score(y_true, y_scores)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Localization metrics (only on fake images, matched with masks)
        mask_dir = MASKS_PATHS[exp['fake']]
        f1_best_list = []
        f1_th_list = []
        iou_best_list = []
        iou_th_list = []
        
        import csv
        image_records = []
        
        print("Computing Localization Metrics...")
        for res in tqdm(fake_results, desc="Localization", leave=False):
            base_name = os.path.basename(res['path'])
            base_id = os.path.splitext(base_name)[0]
            mask_path = os.path.join(mask_dir, f"mask_{base_id}.png")
            
            f1_best, f1_th, iou_best, iou_th = float('nan'), float('nan'), float('nan'), float('nan')
            
            if os.path.exists(mask_path):
                gt_img = Image.open(mask_path).convert('L')
                # The original images could be resized, wait!
                # The test images obtained from DATASET_PATHS are original images or resized?
                # Mask should be same size. Let's ensure res['map'] and gt match.
                gt_np = np.array(gt_img)
                gt_np = (gt_np > 25).astype(bool)
                
                p_map = res['map']
                if p_map.shape != gt_np.shape:
                    import cv2
                    p_map = cv2.resize(p_map, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                f1_best, f1_th, iou_best, iou_th = computeLocalizationMetrics(p_map, gt_np)
                if not np.isnan(f1_best): f1_best_list.append(f1_best)
                if not np.isnan(f1_th): f1_th_list.append(f1_th)
                if not np.isnan(iou_best): iou_best_list.append(iou_best)
                if not np.isnan(iou_th): iou_th_list.append(iou_th)
            
            image_records.append({
                'path': res['path'],
                'label': 1,
                'det_score': res['score'],
                'f1_best': f1_best,
                'f1_th_0.5': f1_th,
                'iou_best': iou_best,
                'iou_th_0.5': iou_th
            })
            
        for res in real_results:
            image_records.append({
                'path': res['path'],
                'label': 0,
                'det_score': res['score'],
                'f1_best': float('nan'),
                'f1_th_0.5': float('nan'),
                'iou_best': float('nan'),
                'iou_th_0.5': float('nan')
            })
            
        csv_path = f"trufor_results_{exp['name']}.csv"
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['path', 'label', 'det_score', 'f1_best', 'f1_th_0.5', 'iou_best', 'iou_th_0.5'])
            writer.writeheader()
            for record in image_records:
                writer.writerow(record)
                
        loc_f1_best = np.mean(f1_best_list) if f1_best_list else float('nan')
        loc_f1_th = np.mean(f1_th_list) if f1_th_list else float('nan')
        loc_iou_best = np.mean(iou_best_list) if iou_best_list else float('nan')
        loc_iou_th = np.mean(iou_th_list) if iou_th_list else float('nan')
        
        report = f'''Dataset: {exp['name']}
  test_acc: {test_acc:.4f}
  test_auc: {test_auc:.4f}
  precision: {precision:.4f}
  recall: {recall:.4f}
  average_precision: {average_precision:.4f}
  mcc: {mcc:.4f}
  localization_f1_best: {loc_f1_best:.4f}
  localization_f1_th_0.5: {loc_f1_th:.4f}
  localization_iou_best: {loc_iou_best:.4f}
  localization_iou_th_0.5: {loc_iou_th:.4f}
'''
        print(report)
        out_f.write(report + "\\n")
        out_f.flush()
        
print('Done. Results saved to trufor_evaluation_results.txt')
