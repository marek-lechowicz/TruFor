"""
Generate per-model train/valid/test list files for the FakeFlickr dataset.

Val and test splits are determined by Flickr30k Entities official lists.
Train contains all remaining images.

Produces one set of list files per inpainting model:
    dataset/data/fakeflickr_{model}_{split}_list.txt

Usage:
    python generate_fakeflickr_lists.py
"""

from pathlib import Path

FF_ROOT        = Path('/home/marek/FakeFlickr/data')
VAL_LIST_PATH  = Path('/home/marek/FakeFlickr/data/flickr30k_entities/val.txt')
TEST_LIST_PATH = Path('/home/marek/FakeFlickr/data/flickr30k_entities/test.txt')

# inpaintings subdir -> dilated mask subdir
MODEL_MAP = {
    'flux_fill_flux_1_dev':    'flux_1_dev_dilated',
    'flux_fill_sd_3_5_large':  'sd_3_5_large_dilated',
    'flux_fill_real_rescaled': 'real_rescaled_dilated',
}

# short key used in list file names
MODEL_SHORT = {
    'flux_fill_flux_1_dev':    'flux_1_dev',
    'flux_fill_sd_3_5_large':  'sd_3_5_large',
    'flux_fill_real_rescaled': 'real_rescaled',
}


def load_ids(path: Path) -> set:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main():
    val_ids  = load_ids(VAL_LIST_PATH)
    test_ids = load_ids(TEST_LIST_PATH)

    out_dir = Path(__file__).parent / 'dataset' / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_model, mask_model in MODEL_MAP.items():
        short    = MODEL_SHORT[img_model]
        img_dir  = FF_ROOT / 'fake-flickr-inpaintings' / img_model / 'img'
        mask_dir = FF_ROOT / 'masks_entities_ensemble' / mask_model / 'masks'

        train_lines, valid_lines, test_lines = [], [], []
        missing = 0

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            stem = img_file.stem
            mask_file = mask_dir / f'mask_{stem}.png'
            if not mask_file.exists():
                print(f'WARNING [{short}]: mask not found for {img_file.name}, skipping.')
                missing += 1
                continue

            line = f'{img_file.relative_to(FF_ROOT)},{mask_file.relative_to(FF_ROOT)}'

            if stem in val_ids:
                valid_lines.append(line)
            elif stem in test_ids:
                test_lines.append(line)
            else:
                train_lines.append(line)

        (out_dir / f'fakeflickr_{short}_train_list.txt').write_text('\n'.join(train_lines) + '\n')
        (out_dir / f'fakeflickr_{short}_valid_list.txt').write_text('\n'.join(valid_lines) + '\n')
        (out_dir / f'fakeflickr_{short}_test_list.txt').write_text('\n'.join(test_lines)  + '\n')

        print(f'[{short}]  train={len(train_lines):5d}  valid={len(valid_lines):5d}  test={len(test_lines):5d}'
              + (f'  missing={missing}' if missing else ''))


if __name__ == '__main__':
    main()
