

def classify_subclonal_wgd(data):
    data['majority_n_wgd'] = data.groupby(['patient_id'])['n_wgd'].transform('mean').round().astype(int)
    data.loc[data['patient_id'] == 'SPECTRUM-OV-081', 'majority_n_wgd'] = 0
    data.loc[data['patient_id'] == 'SPECTRUM-OV-125', 'majority_n_wgd'] = 0
    data['subclonal_wgd'] = data['n_wgd'] > data['majority_n_wgd']
    return data

