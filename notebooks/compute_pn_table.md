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

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

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

cohort_pn = {}
cohort_mn = {}

```

```python

# Load mn/pn data

slide_dirs = glob.glob('users/vazquezi/projects/spectrum/results/if/v25/qupath/outputs/tissue_object_detection/slide/*_cGAS_STING_p53_panCK_CD8_DAPI_R1/cgas_sting_p53_panck_cd8_dapi/')

for slide_dir in tqdm.tqdm(slide_dirs):
    image_name = slide_dir.split('/')[-3]

    if image_name in cohort_pn:
        continue

    # Read pixel classifier results
    region_annotations_geojson = os.path.join(slide_dir, 'region_annotation_results.geojson')

    if not os.path.exists(region_annotations_geojson):
        print(f'missing {region_annotations_geojson}')
        continue

    region_annotations = gpd.read_file(region_annotations_geojson, engine='fiona')

    # Filter for only tumor only
    region_annotations = region_annotations[region_annotations['classification'].map(lambda a: a['name']) == 'Tumor']

    # Read ROI definitions
    roi_dir = 'users/vazquezi/projects/spectrum/pipelines/if/qupath/roi-annotation-cgas-sting-p53-panck-cd8-dapi/annotations/roi'
    roi_annotations_geojson = os.path.join(roi_dir, f'{image_name}.geojson')

    if not os.path.exists(roi_annotations_geojson):
        print(f'missing {roi_annotations_geojson}')
        continue

    roi_annotations = gpd.read_file(roi_annotations_geojson, engine='fiona')

    # Filter for only ROI
    roi_annotations = roi_annotations[roi_annotations['classification'].notnull()]
    roi_annotations = roi_annotations[roi_annotations['classification'].map(lambda a: a['name']) == 'ROI']

    pn_filename = os.path.join(slide_dir, 'object_detection_results.tsv')

    if not os.path.exists(pn_filename):
        print(f'missing {pn_filename}')
        continue

    mn_filename = os.path.join(slide_dir, 'object_detection_results_micronuclei.tsv')

    if not os.path.exists(mn_filename):
        print(f'missing {mn_filename}')
        continue

    pn = read_pn(pn_filename)

    if pn.empty:
        continue

    gdf_pn = gpd.GeoDataFrame(
        pn[['Centroid X', 'Centroid Y']],
        geometry=gpd.points_from_xy(pn['Centroid X'], pn['Centroid Y']),
        crs=roi_annotations.crs)

    # Filter based on ROI and pixel classifier annotations
    gdf_pn = gpd.sjoin(gdf_pn, roi_annotations, predicate="within", how="inner").drop(['index_right'], axis=1)
    gdf_pn = gpd.sjoin(gdf_pn, region_annotations, predicate="within", how="inner")
    
    pn = pn.loc[gdf_pn.index]

    if pn.empty:
        continue

    mn = read_mn(mn_filename)
    
    # Assign micronuclei to closest primary nuclei
    # Note: we do this before filtering any primary nuclei
    pn, mn = assign_mn(pn, mn)

    # Filter micronuclei assigned to filtered primary nuclei
    mn = mn[mn['pn_idx'].isin(pn['idx'].values)].drop(['Class', 'Parent'], axis=1).copy()

    if not pn.empty:
        cohort_pn[image_name] = pn

    if not mn.empty:
        cohort_mn[image_name] = mn

cohort_pn = pd.concat(cohort_pn, names=['image']).reset_index().drop(['level_1'], axis=1)
cohort_pn['image'] = cohort_pn['image'].astype('category')

cohort_mn = pd.concat(cohort_mn, names=['image']).reset_index().drop(['level_1'], axis=1)
cohort_mn['image'] = cohort_mn['image'].astype('category')

```

```python

cohort_pn.to_feather(f'{project_dir}/if/cohort_pn.feather')
cohort_mn.to_feather(f'{project_dir}/if/cohort_mn.feather')

```

```python

```
