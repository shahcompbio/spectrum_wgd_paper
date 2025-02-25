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

pixel_to_micron = 0.1625

```

```python

pn_dtypes = {
    'Centroid X': float,
    'Centroid Y': float,
    'Class': 'category',
    'Parent': 'category',
    'Detection probability': float,
    'Nucleus: Circularity': float,
    'Nucleus: Area µm^2': float,
    'STING: Nucleus: Mean': float,
    'STING: Cytoplasm: Mean': float,
    'STING: Membrane: Mean': float,
    'STING: Cell: Mean': float,
    'cGAS: Cell: Mean': float,
    'cGAS: Cell: Std.Dev.': float,
    'p53: Cell: Mean': float,
    'DAPI: Cell: Mean': float,
}


def read_pn(pn_filename):
    pn_test = pd.read_csv(pn_filename, sep='\t', nrows=1)
    missing_cols = set(pn_dtypes.keys()) - set(pn_test.columns)
    if len(missing_cols) > 0:
        print(f'invalid {pn_filename}, missing {missing_cols}')
        return pd.DataFrame()

    try:
        pn = pd.read_csv(pn_filename, sep='\t', usecols=pn_dtypes.keys(), dtype=pn_dtypes)
    except ValueError as e:
        print(e, pn_filename)
        return pd.DataFrame()

    pn = pn.query('Class == "Primary nucleus"').copy()

    fraction_null = pn[['Centroid X', 'Centroid Y']].isnull().any(axis=1).mean()
    pn = pn[~pn[['Centroid X', 'Centroid Y']].isnull().any(axis=1)]
    assert fraction_null < 0.01

    return pn

```

```python

mn_dtypes = {
    'Centroid X': float,
    'Centroid Y': float,
    'Class': 'category',
    'Parent': 'category',
    'Detection probability': float,
    'Nucleus: Circularity': float,
    'Nucleus: Area µm^2': float,
    'cGAS: Nucleus: Mean': float,
}

def read_mn(mn_filename):
    mn = pd.read_csv(mn_filename, sep='\t', usecols=mn_dtypes.keys(), dtype=mn_dtypes)
    mn = mn.query('Class == "Micronucleus"')

    fraction_null = mn[['Centroid X', 'Centroid Y']].isnull().any(axis=1).mean()
    mn = mn[~mn[['Centroid X', 'Centroid Y']].isnull().any(axis=1)]
    assert fraction_null < 0.01

    return mn

```

```python

def assign_mn(pn, mn):
    pn['idx'] = range(pn.shape[0])
    
    tree = KDTree(pn[['Centroid X', 'Centroid Y']].values)
    
    distances, indices = tree.query(mn[['Centroid X', 'Centroid Y']].values)
    
    mn['pn_idx'] = pn['idx'].values[indices]
    mn['pn_distance'] = distances

    return pn, mn

```

```python

slide_dir = '/data1/shahs3/users/vazquezi/projects/spectrum/results/if/v25/qupath/outputs/tissue_object_detection/slide/SPECTRUM-OV-051_S1_INFRACOLIC_OMENTUM_cGAS_STING_p53_panCK_CD8_DAPI_R1/cgas_sting_p53_panck_cd8_dapi/'
image_name = slide_dir.split('/')[-3]

```

```python

# Read ROI definitions
roi_dir = '/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/qupath/roi-annotation-cgas-sting-p53-panck-cd8-dapi/annotations/roi'
roi_annotations_geojson = os.path.join(roi_dir, f'{image_name}.geojson')
print(roi_annotations_geojson)

if not os.path.exists(roi_annotations_geojson):
    print(f'missing {roi_annotations_geojson}')

roi_annotations = gpd.read_file(roi_annotations_geojson, engine='fiona')

# Filter for only ROI
roi_annotations = roi_annotations[roi_annotations['classification'].notnull()]
roi_annotations = roi_annotations[roi_annotations['classification'].map(lambda a: a['name']) == 'ROI']

roi_annotations.plot()

```

```python

# Read pixel classifier results
region_annotations_geojson = os.path.join(slide_dir, 'region_annotation_results.geojson')
print(region_annotations_geojson)

if not os.path.exists(region_annotations_geojson):
    print(f'missing {region_annotations_geojson}')

region_annotations = gpd.read_file(region_annotations_geojson, engine='fiona')

# Filter for only tumor only
fig = plt.figure(figsize=(10, 10), dpi=1000)
ax = plt.gca()
region_annotations = region_annotations[region_annotations['classification'].map(lambda a: a['name']) == 'Tumor']
region_annotations.plot(aspect='equal', ax=ax)

```

```python

# Read primary nuclei

pn_filename = os.path.join(slide_dir, 'object_detection_results.tsv')

if not os.path.exists(pn_filename):
    print(f'missing {pn_filename}')

pn = read_pn(pn_filename)

```

```python

# Plot primary nuclei

gdf_pn = gpd.GeoDataFrame(
    pn[['Centroid X', 'Centroid Y']],
    geometry=gpd.points_from_xy(pn['Centroid X'], pn['Centroid Y']),
    crs=roi_annotations.crs)

gdf_pn.plot(aspect='equal', markersize=1)

```

```python

# Filter based on ROI annotations, and plot

gdf_pn = gpd.sjoin(gdf_pn, roi_annotations, predicate="within", how="inner").drop(['index_right'], axis=1)
gdf_pn.plot(aspect='equal', markersize=1)

```

```python

# Confirm filtering

pn = pn.loc[gdf_pn.index]
gpd.GeoDataFrame(
    pn[['Centroid X', 'Centroid Y']],
    geometry=gpd.points_from_xy(pn['Centroid X'], pn['Centroid Y']),
    crs=roi_annotations.crs).plot(aspect='equal', markersize=1)

```

```python

# Filter based on pixel classifier and plot

gdf_pn = gpd.sjoin(gdf_pn, region_annotations, predicate="within", how="inner")
gdf_pn.plot(aspect='equal', markersize=1)

```

```python

# Confirm filtering

fig = plt.figure(figsize=(10, 10), dpi=1000)
ax = plt.gca()

pn = pn.loc[gdf_pn.index]
gpd.GeoDataFrame(
    pn[['Centroid X', 'Centroid Y']],
    geometry=gpd.points_from_xy(pn['Centroid X'], pn['Centroid Y']),
    crs=roi_annotations.crs).plot(aspect='equal', markersize=1, ax=ax, alpha=0.1)

```

```python

# Read and assign micronuclei

mn_filename = os.path.join(slide_dir, 'object_detection_results_micronuclei.tsv')

if not os.path.exists(mn_filename):
    print(f'missing {mn_filename}')

mn = read_mn(mn_filename)

# Assign micronuclei to closest primary nuclei
# Note: we do this before filtering any primary nuclei
pn, mn = assign_mn(pn, mn)

# Filter micronuclei assigned to filtered primary nuclei
mn = mn[mn['pn_idx'].isin(pn['idx'].values)].drop(['Class', 'Parent'], axis=1).copy()

```

```python

# Plot filtered micronuclei
filtered_mn = mn[
    (mn['pn_distance'] < (5 / pixel_to_micron)) &
    (mn['Nucleus: Circularity'] > 0.65) &
    (mn['Nucleus: Area µm^2'] > 0.1) &
    (mn['Nucleus: Area µm^2'] <= 20) &
    (mn['Detection probability'] > 0.75)
]

plt.figure(dpi=300)
gpd.GeoDataFrame(
    filtered_mn[['Centroid X', 'Centroid Y']],
    geometry=gpd.points_from_xy(filtered_mn['Centroid X'], filtered_mn['Centroid Y']),
    crs=roi_annotations.crs).plot(aspect='equal', markersize=1)

```

```python

```
