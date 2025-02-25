---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import anndata as ad
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import scgenome 

from sklearn.metrics import adjusted_rand_score
from IPython.display import Image
import matplotlib.image as mimage
import subprocess as sp
```

```python
pipeline_outdir = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
```

```python
cell_info = pd.read_csv(os.path.join(pipeline_outdir, 'preprocessing/summary/filtered_cell_table.csv.gz'))
```

# load anndatas and identify cells in each filtering group

```python
signals_adatas = {}
signals_dir = os.path.join(pipeline_outdir, 'preprocessing/signals')

for f in tqdm.tqdm(os.listdir(signals_dir)):
    p = f.split('_')[1].split('.')[0]
    signals_adatas[p] = ad.read_h5ad(os.path.join(signals_dir, f))
```

## summary plots

```python
rows = []
for p, pdf in tqdm.tqdm(cell_info.groupby('patient_id')):
    row = {}
    row['patient_id'] = p
    row['n_cells'] = len(pdf)

    row['normal_cells'] =  len(pdf[pdf.is_normal == True]) 
    row['prop_normal_cells'] = len(pdf[pdf.is_normal == True]) / len(pdf)
    remaining_cells = pdf[pdf.is_normal != True]

    row['aberrant_normal_cells'] = len(remaining_cells[remaining_cells.is_aberrant_normal_cell])
    row['prop_aberrant_normal_cells'] = len(remaining_cells[remaining_cells.is_aberrant_normal_cell]) / len(pdf)
    remaining_cells = remaining_cells[~remaining_cells.is_aberrant_normal_cell]

    row['s_phase'] = len(remaining_cells[remaining_cells.is_s_phase_thresholds == True])
    row['prop_s_phase'] = len(remaining_cells[remaining_cells.is_s_phase_thresholds == True]) / len(pdf)
    remaining_cells = remaining_cells[~remaining_cells.is_s_phase_thresholds]

    row['doublets'] = len(remaining_cells[remaining_cells.is_doublet != 'No'])
    row['prop_doublet'] = len(remaining_cells[remaining_cells.is_doublet != 'No']) / len(pdf)
    remaining_cells = remaining_cells[remaining_cells.is_doublet == 'No']

    row['filter_pass'] = len(pdf[pdf.include_cell]) / len(pdf)
    rows.append(row)
plotdf = pd.DataFrame(rows)
```

```python
cell_info[cell_info.patient_id == 'SPECTRUM-OV-110'].include_cell.sum()
```

```python
plt.figure(dpi = 300, figsize=(8,6))

xs = np.arange(len(plotdf))
bottom = np.zeros(len(xs))

plt.bar(xs, 1, label = 'filter pass')

plt.bar(xs, plotdf.prop_s_phase, bottom=bottom, label = 's-phase')
bottom += plotdf.prop_s_phase

plt.bar(xs, plotdf.prop_doublet, bottom=bottom, label = 'image_doublet')
bottom += plotdf.prop_doublet

plt.bar(xs, plotdf.prop_aberrant_normal_cells, bottom=bottom, label = 'aberrant_normal')
bottom += plotdf.prop_aberrant_normal_cells

plt.bar(xs, plotdf.prop_normal_cells, bottom=bottom, label = 'normal')
bottom += plotdf.prop_normal_cells

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(xs, plotdf.patient_id, rotation=90)
plt.ylabel("Proportion of HSCN-called cells")
```

```python
plt.figure(dpi = 300, figsize=(8,6))

xs = np.arange(len(plotdf))
bottom = np.zeros(len(xs))

plt.bar(xs, plotdf.n_cells, label = 'filter pass')
plt.bar(xs, plotdf.s_phase, bottom=bottom, label = 's-phase')
bottom += plotdf.s_phase

plt.bar(xs, plotdf.doublets, bottom=bottom, label = 'image_doublet')
bottom += plotdf.doublets

plt.bar(xs, plotdf.aberrant_normal_cells, bottom=bottom, label = 'aberrant_normal')
bottom += plotdf.aberrant_normal_cells

plt.bar(xs, plotdf.normal_cells, bottom=bottom, label = 'normal')
bottom += plotdf.normal_cells

plt.legend()
plt.xticks(xs, plotdf.patient_id, rotation=90)
plt.ylabel("Number of HSCN-called cells")
```

```python

```

# show example cell profiles


## select exemplar cells

```python
sphase_cells = cell_info[cell_info.is_s_phase_thresholds].reset_index(drop=True)

np.random.seed(0)
sample_s_phase = np.random.choice(sphase_cells.index, size=20, replace=False)
for i, c in enumerate(sample_s_phase):
    r = sphase_cells.loc[c]
    
    plt.figure(figsize=(6,2), dpi = 150)
    scgenome.pl.plot_cn_profile(signals_adatas[r.patient_id], r.cell_id,
                            value_layer_name = 'copy', state_layer_name='state',
                           squashy = True)
    plt.title(f'{r.cell_id} ({i})')

```

```python
example_sphase = sphase_cells.loc[sample_s_phase[[4, 6, 9, 13]]].cell_id.values
```

```python
normal_cells = cell_info[cell_info.is_normal].reset_index(drop=True)

np.random.seed(0)
sample_normal = np.random.choice(normal_cells.index, size=20, replace=False)
for i, c in enumerate(sample_normal):
    r = normal_cells.loc[c]
    
    plt.figure(figsize=(6,2), dpi = 150)
    scgenome.pl.plot_cn_profile(signals_adatas[r.patient_id], r.cell_id,
                            value_layer_name = 'copy', state_layer_name='state',
                           squashy = True)
    plt.title(f'{r.cell_id} ({i})')

```

```python
example_normal = normal_cells.loc[sample_normal[[0, 3, 7, 12]]].cell_id.values
```

```python
aberrant_cells = cell_info[~cell_info.is_normal & cell_info.is_aberrant_normal_cell & ~cell_info.is_s_phase_thresholds].reset_index(drop=True)

np.random.seed(0)
sample_aberrant = np.random.choice(aberrant_cells.index, size=20, replace=False)
for i, c in enumerate(sample_aberrant):
    r = aberrant_cells.loc[c]
    
    plt.figure(figsize=(6,2), dpi = 150)
    scgenome.pl.plot_cn_profile(signals_adatas[r.patient_id], r.cell_id,
                            value_layer_name = 'copy', state_layer_name='state',
                           squashy = True)
    plt.title(f'{r.cell_id} ({i})')

```

```python
example_aberrant = aberrant_cells.loc[sample_aberrant[[0, 1, 8, 19]]].cell_id.values
```

```python
doublet_cells = cell_info[~cell_info.is_normal & ~cell_info.is_aberrant_normal_cell & ~cell_info.is_s_phase_thresholds & (cell_info.is_doublet=='Yes')].reset_index(drop=True)

np.random.seed(0)
sample_doublet = np.random.choice(doublet_cells.index, size=20, replace=False)
for i, c in enumerate(sample_doublet):
    r = doublet_cells.loc[c]
    
    plt.figure(figsize=(6,2), dpi = 150)
    scgenome.pl.plot_cn_profile(signals_adatas[r.patient_id], r.cell_id,
                            value_layer_name = 'copy', state_layer_name='state',
                           squashy = True)
    plt.title(f'{r.cell_id} ({i})')

```

```python
example_doublets = doublet_cells.loc[sample_doublet[[0, 11, 13, 16]]].cell_id.values
```

```python
shortsegment_cells = cell_info[~cell_info.is_normal & ~cell_info.is_aberrant_normal_cell & 
~cell_info.is_s_phase_thresholds & (cell_info.is_doublet=='No') & (cell_info.longest_135_segment < 20)].reset_index(drop=True)

np.random.seed(0)
sample_shortsegment = np.random.choice(shortsegment_cells.index, size=20, replace=False)
for i, c in enumerate(sample_shortsegment):
    r = shortsegment_cells.loc[c]
    
    plt.figure(figsize=(6,2), dpi = 150)
    scgenome.pl.plot_cn_profile(signals_adatas[r.patient_id], r.cell_id,
                            value_layer_name = 'copy', state_layer_name='state',
                           squashy = True)
    plt.title(f'{r.cell_id} ({i})')

```

```python
example_shortsegment = shortsegment_cells.loc[sample_shortsegment[[0, 2, 4, 8]]].cell_id.values
```

```python
np.random.seed(1)
example_pass = np.random.choice(cell_info[cell_info.include_cell].cell_id.values, size=4, replace=False)
```

## show example DLP images for doublet-related ones

```python
example_cells = list(example_doublets) + list(example_sphase) + list(example_normal) + list(example_aberrant) + list(example_shortsegment) + list(example_pass)
```

```python
dlp_pdfs_dir = '/data1/shahs3/users/myersm2/dlp/images/pdfs'

```

```python
cell2page = {}
for c in example_cells:
    library = c.split('-')[-3]
    df = pd.read_csv(os.path.join(dlp_pdfs_dir, '..', 'csvs', library + '.csv'))
    cell_id = '-'.join(c.split('-')[-3:])
    cell2page[c] = df[df.cell_id == cell_id].iloc[0].name + 1
```

```python
len(cell2page)
```

### get the PDF page and convert it to a png for each cell
requires qpdf (conda-forge), and imagemagick (conda-forge) for `convert` command

<!-- #raw -->
qpdf A96121B.pdf --pages . 715-715 -- /rtsess01/compute/juno/shah/users/myersm2/repos/spectrum-figures/figures/dlp-images/SPECTRUM-OV-004_S1_LEFT_ADNEXA-A96121B-R52-C17.pdf

convert -density 300 SPECTRUM-OV-004_S1_LEFT_ADNEXA-A96121B-R52-C17.pdf -quality 100 image.png
<!-- #endraw -->

```python
output_dir = '../../figures/filtering/dlp-images'
```

```python
for c, pg in tqdm.tqdm(cell2page.items()):
    library = c.split('-')[-3]
    pdf_file = os.path.join(output_dir, 'pdfs', c + '.pdf')
    png_file = os.path.join(output_dir, 'pngs', c + '.png')

    if not os.path.exists(pdf_file):
        cmd1 = ['qpdf', os.path.join(dlp_pdfs_dir, library + '.pdf'), '--pages', '.', f'{pg}-{pg}', '--', pdf_file]
        sp.run(cmd1)

    if not os.path.exists(png_file):
        cmd2 = ['convert', '-density', '300', '-trim', pdf_file, '-quality', '100', png_file]
        sp.run(cmd2)
    
```

# put together figure

<!-- #raw -->
fig, axes = plt.subplots(4, 3, figsize=(10, 8), width_ratios=[0.4, 0.4, 0.2])

for i, cell_id in enumerate(example_doublets):
    p = cell_id.split('_')[0]
    adata = signals_adatas[p]
    scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'copy', state_layer_name='state', squashy = True, ax = axes[i][0])
    scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'BAF', state_layer_name='Min', ax = axes[i][1])
    axes[i][1].set_title('manually annotated doublet: ' + cell_id)
    axes[i][0].get_legend().remove()
    axes[i][1].get_legend().remove()

    img = mimage.imread(os.path.join(output_dir, 'pngs', f'{cell_id}.png'))
    axes[i][2].imshow(img)
    axes[i][2].axis('off')
plt.tight_layout()
<!-- #endraw -->

```python
plt.figure(figsize = (9,11), dpi = 300)

for i, cell_id in enumerate(example_doublets[:2]):
    p = cell_id.split('_')[0]
    adata = signals_adatas[p]
    
    plt.subplot(4, 1, (2*i) + 1)
    plt.title('manually annotated doublet: ' + cell_id)
    plt.axis('off')
    
    plt.subplot(4, 2, (4*i) + 1)
    scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'copy', state_layer_name='state', squashy = True)
    plt.gca().get_legend().remove()

    plt.subplot(4, 2, (4*i) + 2)
    scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'BAF', state_layer_name='B')
    plt.gca().get_legend().remove()

    plt.subplot(4, 1, (2 * i) + 2)
    img = mimage.imread(os.path.join(output_dir, 'pngs', f'{cell_id}.png'))
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
```

```python
outdirs = ['example_filter_pass', 'example_doublets', 'example_normal', 'example_aberrant', 'example_sphase', 'example_shortsegment']
cell_lists = [example_pass, example_doublets, example_normal, example_aberrant, example_sphase, example_shortsegment]
labels = ['filter_pass', 'manual_doublet', 'normal', 'aberrant_normal', 's-phase', 'no135segment']
```

```python
outdir_stem = '../../figures/final/filtering'
```

```python
for i, (cell_list, outdir, label) in tqdm.tqdm(enumerate(zip(cell_lists, outdirs, labels))):
    for pg in range(2):
        if not os.path.exists(os.path.join(outdir_stem, outdir)):
            os.makedirs(os.path.join(outdir_stem, outdir))
            
        plt.figure(figsize = (9,12), dpi = 300)
        for i, cell_id in enumerate(cell_list[(2*pg):(2*(pg+1))]):
            p = cell_id.split('_')[0]
            adata = signals_adatas[p]
            
            plt.subplot(4, 1, (2*i) + 1)
            plt.title(f'{label}: ' + cell_id)
            plt.axis('off')
            
            plt.subplot(4, 2, (4*i) + 1)
            scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'copy', state_layer_name='state', squashy = True)
            plt.gca().get_legend().remove()
        
            plt.subplot(4, 2, (4*i) + 2)
            scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name = 'BAF', state_layer_name='B')
            plt.gca().get_legend().remove()
        
            plt.subplot(4, 1, (2 * i) + 2)
            img = mimage.imread(os.path.join(output_dir, 'pngs', f'{cell_id}.png'))
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(os.path.join(outdir_stem, outdir, f'{label}_page{pg}.png'))
        plt.close()
```

```python

```

```python

```

```python

```
