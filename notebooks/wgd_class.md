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

import os
import pandas as pd
import numpy as np

import spectrumanalysis.wgd


project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

fraction_wgd = cell_info.groupby(['patient_id'])['is_wgd'].mean().rename('fraction_wgd').reset_index()

fraction_wgd['wgd_class'] = 'WGD-low'
fraction_wgd.loc[fraction_wgd['fraction_wgd'] >= 0.5, 'wgd_class'] = 'WGD-high'
fraction_wgd

```

```python

fraction_wgd.to_csv('../../../../annotations/fraction_wgd_class.csv', index=False)

```

```python

```
