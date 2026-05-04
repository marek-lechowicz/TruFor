import matplotlib
matplotlib.use('Agg') # No GUI
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from glob import glob

img_dir = 'eval_sample/images'
res_dir = 'eval_sample/results'
mask_dir = 'eval_sample/masks'
orig_dir = '/home/marek/projects/fake_flickr_sota/data/real'
out_viz = 'eval_sample/visualizations'

os.makedirs(out_viz, exist_ok=True)

images = sorted(glob(os.path.join(img_dir, '*.png')))

for img_path in images:
    name_with_png = os.path.basename(img_path)
    base_name = os.path.splitext(name_with_png)[0]
    
    res_path = os.path.join(res_dir, name_with_png + '.npz')
    mask_path = os.path.join(mask_dir, 'mask_' + name_with_png)
    orig_path = os.path.join(orig_dir, base_name + '.jpg')
    
    if not os.path.exists(res_path):
        print(f"Result missing for {name_with_png}")
        continue
        
    result = np.load(res_path)
    img_edited = Image.open(img_path)
    img_orig = Image.open(orig_path) if os.path.exists(orig_path) else None
    mask = Image.open(mask_path) if os.path.exists(mask_path) else None
    
    # Increase to 5 columns
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Image ID: {base_name} | Integrity Score: {result['score']:.4f}")
    
    for ax in axs:
        ax.axis('off')
        
    # 1. Original (Before edits)
    if img_orig:
        axs[0].imshow(img_orig)
        axs[0].set_title('Original (Before Edits)')
    else:
        axs[0].text(0.5, 0.5, 'Original Not Found', ha='center')
    
    # 2. Edited Image
    axs[1].imshow(img_edited)
    axs[1].set_title('Edited Image (Input)')
    
    # 3. Ground Truth
    if mask:
        axs[2].imshow(mask, cmap='gray')
    axs[2].set_title('Ground Truth Mask')
    
    # 4. TruFor Prediction
    axs[3].imshow(result['map'], cmap='RdBu_r', vmin=0, vmax=1)
    axs[3].set_title('TruFor Prediction')
    
    # 5. Confidence
    axs[4].imshow(result['conf'], cmap='gray', vmin=0, vmax=1)
    axs[4].set_title('Confidence Map')
    
    save_path = os.path.join(out_viz, f'viz_{base_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization for {base_name}")
