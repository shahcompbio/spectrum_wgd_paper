from argparse import ArgumentParser
import pandas as pd
from scipy.spatial import KDTree
import tqdm
import glob
import os
import numpy as np
import scipy.stats

def get_args():
    p = ArgumentParser()
    p.add_argument('--detections_primary_nuclei', help='Object detections of primary nuclei')
    p.add_argument('--detections_micronuclei', help='Object detections of micronuclei')
    p.add_argument('--detections_primary_nuclei_assigned', help='Primary nuclei objects with assigned micronuclei')
    p.add_argument('--detections_micronuclei_assigned', help='Micronuclei objects with assigned primary nuclei')
    p.add_argument('--detections_assigned', help='Micronuclei objects with assigned primary nuclei')
    p.add_argument('--detections_micronuclei_filtered_assigned', help='Micronuclei objects with assigned primary nuclei')
    p.add_argument('--detections_filtered_assigned', help='Micronuclei objects with assigned primary nuclei')

    return p.parse_args()

def find_nearest_detection(pn, mn, microns_per_pixel=0.1625):
    pn['index'] = range(pn.shape[0])

    # Check for non-finite values
    print("Non-finite values in PN data:", ~np.isfinite(pn[['Centroid X', 'Centroid Y']].values))
    print("Non-finite values in MN data:", ~np.isfinite(mn[['Centroid X', 'Centroid Y']].values))

    tree = KDTree(pn[['Centroid X', 'Centroid Y']].values)

    distances, indices = tree.query(mn[['Centroid X', 'Centroid Y']].values)

    pd.set_option('display.max_columns', None)
    test = pn.rename(columns={'Object ID':'Nearest Primary Nucleus ID'})
    print(test.head())

    mn['pn_index'] = indices
    mn = mn.merge(
        pn.rename(columns={'Object ID':'Nearest Primary Nucleus ID'})[['index','Nearest Primary Nucleus ID']], 
        how='left', left_on = 'pn_index', right_on = 'index')
    
    mn['Distance to nearest Primary Nucleus'] = distances

    cutoff_distance = 5 / microns_per_pixel
    mn['Proximity to nearest Primary Nucleus'] = np.where(mn['Distance to nearest Primary Nucleus'] <= cutoff_distance, 'Proximal', 'Distal')

    mn_count = mn.groupby(['pn_index']).size()
    min_distance = mn.groupby(['pn_index'])['Distance to nearest Primary Nucleus'].min()
    max_distance = mn.groupby(['pn_index'])['Distance to nearest Primary Nucleus'].max()

    pn['Nearest Micronuclei Count'] = 0
    pn.loc[pn.index[mn_count.index], 'Nearest Micronuclei Count'] = mn_count.values

    pn['Min distance to nearest Micronuclei'] = 0
    pn.loc[pn.index[min_distance.index], 'Min distance to nearest Micronuclei'] = min_distance.values

    pn['Max distance to nearest Micronuclei'] = 0
    pn.loc[pn.index[max_distance.index], 'Max distance to nearest Micronuclei'] = max_distance.values

    return pn, mn

def main():
    argv = get_args()

    pn_dtypes = {
        'Image': 'category',
        'Object ID': 'category',
        'Centroid X': float,
        'Centroid Y': float,
        'Name': 'category',
        'Class': 'category',
        'Parent': 'category',
        'Parent ID': 'category',
        'ROI': 'category',
        'ROI ID': 'category',
        'Detection probability': float,
        'Nucleus: Circularity': float,
        'Nucleus: Area µm^2': float,
    }

    pn = pd.read_csv(argv.detections_primary_nuclei, sep='\t', dtype=pn_dtypes)

    mn_dtypes = {
        'Image': 'category',
        'Object ID': 'category',
        'Centroid X': float,
        'Centroid Y': float,
        'Name': 'category',
        'Class': 'category',
        'Parent': 'category',
        'Parent ID': 'category',
        'ROI': 'category',
        'ROI ID': 'category',
        'Detection probability': float,
        'Nucleus: Circularity': float,
        'Nucleus: Area µm^2': float,
    }

    mn = pd.read_csv(argv.detections_micronuclei, sep='\t', dtype=mn_dtypes)

    # Filter out data with nan values for centroids
    pn = pn.dropna(subset=['Centroid X', 'Centroid Y'])
    mn = mn.dropna(subset=['Centroid X', 'Centroid Y'])

    # Find nearest detection for each micronuclei
    if len(mn.index) > 0:
        pn_assigned, mn_assigned = find_nearest_detection(pn, mn)
    else:
        pn_assigned = pn
        mn_assigned = mn

    pn_assigned.to_csv(argv.detections_primary_nuclei_assigned, sep='\t', index=False)

    mn_assigned.to_csv(argv.detections_micronuclei_assigned, sep='\t', index=False)

    pn_mn_assigned = pd.concat([pn_assigned, mn_assigned])
    
    pn_mn_assigned.to_csv(argv.detections_assigned, sep='\t', index=False)

    # Filter micronuclei based on features
    mn_filtered = mn[
        (mn['Class'].str.contains('Micronucleus', na=False)) & 
        (mn['Nucleus: Circularity'] > 0.65) & 
        (mn['Nucleus: Area µm^2'] > 0.1) & 
        (mn['Nucleus: Area µm^2'] <= 20) & 
        (mn['Detection probability'] > 0.75)
    ]
    
    if len(mn_filtered.index) > 0:
        # Find nearest detection for each micronuclei post-filtering
        pn_assigned, mn_filtered_assigned = find_nearest_detection(pn, mn_filtered)
        # Filter out distal micronuclei
        mn_filtered_assigned = mn_filtered_assigned[mn_filtered_assigned['Proximity to nearest Primary Nucleus'] == 'Proximal']
    else:
        # If no micronuclei pass filtering, return empty DataFrame with expected columns
        additional_columns = [
            'pn_index', 
            'Nearest Primary Nucleus ID',
            'Distance to nearest Primary Nucleus', 
            'Proximity to nearest Primary Nucleus',
            'Nearest Micronuclei Count',
            'Min distance to nearest Micronuclei',
            'Max distance to nearest Micronuclei'
        ]
        mn_filtered_assigned = pd.DataFrame(columns=list(mn.columns) + additional_columns)
        
    mn_filtered_assigned.to_csv(argv.detections_micronuclei_filtered_assigned, sep='\t', index=False)

    # Concatenate dataframes, and skip concatenation if empty
    if not mn_filtered_assigned.empty:
        pn_mn_filtered_assigned = pd.concat([pn_assigned, mn_filtered_assigned])
    else:
        pn_mn_filtered_assigned = pn_assigned.copy()

    print(pn_assigned.loc[pn_assigned.index.duplicated(keep='first')])
    print(mn_filtered_assigned.loc[mn_filtered_assigned.index.duplicated(keep='first')])

    pn_mn_filtered_assigned = pd.concat([pn_assigned, mn_filtered_assigned], ignore_index = True)
    
    pn_mn_filtered_assigned.to_csv(argv.detections_filtered_assigned, sep='\t', index=False)

if __name__ == '__main__':
    main()