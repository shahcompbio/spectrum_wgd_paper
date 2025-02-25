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
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scgenome

import spectrumanalysis.dataload

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

```

```python

nullisomies = []

for patient_id in tqdm.tqdm(cell_info['patient_id'].unique()):
    adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
    arm_state = (
        adata
            .to_df(layer='state')
            .loc[cell_info.query(f'patient_id == "{patient_id}"')['cell_id'].values]
            .T.set_index(adata.var['chr'], append=True)
            .set_index(adata.var['arm'], append=True).groupby(level=[1, 2], observed=True)
            .median().stack().rename('count').reset_index())
    nullisomy = arm_state[arm_state['count'] == 0]
    nullisomies.append(nullisomy.assign(patient_id=patient_id))

nullisomies = pd.concat(nullisomies)
nullisomies = nullisomies.merge(cell_info[['cell_id', 'multipolar']], how='left')
assert nullisomies['multipolar'].notnull().all()
nullisomies

```

```python

n_cells_patient = cell_info.groupby(['patient_id']).size().rename('n_cells_patient').reset_index()

nullisomy_rates = (nullisomies
    .groupby(['patient_id'])
    .size()
    .reindex(cell_info['patient_id'].unique(), fill_value=0)
    .rename('arm_count')
    .reset_index())
nullisomy_rates = nullisomy_rates.merge(n_cells_patient)
nullisomy_rates['nullisomy_arm_rate'] = nullisomy_rates['arm_count'] / nullisomy_rates['n_cells_patient']

nullisomy_rates.head()

```

```python

n_cells_group = cell_info.groupby(['patient_id', 'multipolar']).size().rename('n_cells_group').reset_index()

nullisomy_multipolar_rates = (nullisomies
    .groupby(['patient_id', 'multipolar'])
    .size()
    .unstack(fill_value=0).stack()
    .rename('arm_count')
    .reset_index()
    .merge(n_cells_group))

nullisomy_multipolar_rates['nullisomy_arm_rate'] = (nullisomy_multipolar_rates['arm_count'] + 1) / nullisomy_multipolar_rates['n_cells_group']

nullisomy_multipolar_rates.head()

```

```python

patient_id = 'SPECTRUM-OV-004'
adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

cell_id = nullisomies.query(f'patient_id == "{patient_id}"').sort_values('cell_id')['cell_id'].unique()[2]

plt.figure(figsize=(12, 3), dpi=150)
g = scgenome.pl.plot_cn_profile(
    adata,
    cell_id,
    state_layer_name='state',
    value_layer_name='copy',
    squashy=True)
plt.title(f'cell {cell_id}')
plt.ylabel('Total Copy')
sns.despine()

```

```python

nullisomy_rates.to_csv('../../../../annotations/nullisomy_rates.csv', index=False)
nullisomy_multipolar_rates.to_csv('../../../../annotations/nullisomy_multipolar_rates.csv', index=False)

```

```python

```
