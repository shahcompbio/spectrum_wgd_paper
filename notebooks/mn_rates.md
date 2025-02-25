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
    display_name: Python Spectrum
    language: python
    name: python_spectrum
---

```python

import pandas as pd
from scipy.spatial import KDTree
import tqdm
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import anndata as ad
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

import warnings
warnings.filterwarnings("error")

os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'

microns_per_pixel = 0.1625

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

sheet_id = '1UvogBN8nc16OIVKUyA_GtzxiHEGfHsqroldN45-4OKs'
g_id = '161701679'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={g_id}'
batch_info = pd.read_csv(url)
batch_info['slide_id'] = batch_info['spectrum_sample_id']

cols = [
    'slide_id',
    'batch',
    'tumor_type',
    'therapy',
    'procedure',
    'procedure_type',
    'slide_qc',
    'cGAS_staining',
    'genomic_instability_inclusion_status',
]

batch_info.to_csv('../../../annotations/if_qc.csv', index=False)

```

```python

cohort_pn = pd.read_feather(f'{project_dir}/if/cohort_pn.feather')
cohort_mn = pd.read_feather(f'{project_dir}/if/cohort_mn.feather')

```

```python

filtered_mn = cohort_mn[
    (cohort_mn['pn_distance'] < (5 / microns_per_pixel)) &
    (cohort_mn['Nucleus: Circularity'] > 0.65) &
    (cohort_mn['Nucleus: Area µm^2'] > 0.1) &
    (cohort_mn['Nucleus: Area µm^2'] <= 20) &
    (cohort_mn['Detection probability'] > 0.75)
]

```

```python

mn_count = filtered_mn.groupby(['image', 'pn_idx'], observed=True).size()

cohort_pn.set_index(['image', 'idx'], inplace=True)
cohort_pn['mn_count'] = mn_count
cohort_pn['mn_count'] = cohort_pn['mn_count'].fillna(0)
cohort_pn.reset_index(inplace=True)

```

```python

fov_size_pixels = 1000 / microns_per_pixel

cohort_pn['x_fov'] = (cohort_pn['Centroid X'] / fov_size_pixels).astype(int)
cohort_pn['y_fov'] = (cohort_pn['Centroid Y'] / fov_size_pixels).astype(int)

```

```python

fov_summary = cohort_pn.groupby(['image', 'x_fov', 'y_fov'], observed=True).agg(
    pn_count=('idx', 'size'),
    mn_count=('mn_count', 'sum'),
    sting=('STING: Cell: Mean', 'mean'),
    cgas_mean=('cGAS: Cell: Mean', 'mean'),
    cgas_stddev=('cGAS: Cell: Std.Dev.', 'mean'),
    tp53_mean=('p53: Cell: Mean', 'mean'),
    nucleus_area=('Nucleus: Area µm^2', 'mean'),
)

fov_summary['mn_rate'] = (fov_summary['mn_count'] + 1) / fov_summary['pn_count']
fov_summary = fov_summary.reset_index()
fov_summary['patient_id'] = fov_summary['image'].str.split('_', expand=True)[0]
fov_summary['slide_id'] = fov_summary['image'].str.replace('.mrxs', '', regex=False)
fov_summary = fov_summary.merge(batch_info[cols], how='left', on='slide_id')

fov_summary.to_csv(f'{project_dir}/if/mn_rate_fov.csv', index=False)

```

```python

image_summary = cohort_pn.groupby(['image'], observed=True).agg(
    pn_count=('idx', 'size'),
    mn_count=('mn_count', 'sum'),
    sting=('STING: Cell: Mean', 'mean'),
    cgas_mean=('cGAS: Cell: Mean', 'mean'),
    cgas_stddev=('cGAS: Cell: Std.Dev.', 'mean'),
    tp53_mean=('p53: Cell: Mean', 'mean'),
    nucleus_area=('Nucleus: Area µm^2', 'mean'),
)

image_summary['mn_rate'] = (image_summary['mn_count'] + 1) / image_summary['pn_count']
image_summary = image_summary.reset_index()
image_summary['patient_id'] = image_summary['image'].str.split('_', expand=True)[0]
image_summary['slide_id'] = image_summary['image'].str.replace('.mrxs', '', regex=False)
image_summary = image_summary.merge(batch_info[cols], how='left', on='slide_id')

image_summary.to_csv(f'{project_dir}/if/mn_rate.csv', index=False)

```


# QC


```python

sns.boxplot(data=image_summary, x='batch', y='mn_rate', fliersize=0)
sns.stripplot(data=image_summary, x='batch', y='mn_rate')

```

```python

plot_data = fov_summary.query('pn_count > 1000')

sns.boxplot(data=plot_data, x='batch', y='mn_rate', fliersize=0)
sns.stripplot(data=plot_data, x='batch', y='mn_rate')

```

```python

plot_data = fov_summary.query('pn_count > 1000').copy()
plot_data['sample_id'] = plot_data['slide_id'].str.replace('_cGAS_STING_p53_panCK_CD8_DAPI_R1', '')

plt.figure(figsize=(2, 16))
sns.boxplot(data=plot_data, y='sample_id', x='mn_rate', fliersize=0)
sns.stripplot(data=plot_data, y='sample_id', x='mn_rate', s=1)

```

```python

```

```python

```
