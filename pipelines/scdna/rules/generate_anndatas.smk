
def get_hmmcopy_anndata_inputs(wildcards):
    aliquot_hmmcopy = hmmcopy_info.set_index('isabl_aliquot_id').loc[wildcards.aliquot_id]
    return {
        'metrics': aliquot_hmmcopy['MONDRIAN_HMMCOPY_metrics'],
        'reads': aliquot_hmmcopy['MONDRIAN_HMMCOPY_reads'],
    }


rule create_hmmcopy_anndata:
    input:
        unpack(get_hmmcopy_anndata_inputs)
    params:
        patient_id=lambda wildcards, input: hmmcopy_info.set_index('isabl_aliquot_id').loc[wildcards.aliquot_id, 'isabl_patient_id'],
        sample_id=lambda wildcards, input: hmmcopy_info.set_index('isabl_aliquot_id').loc[wildcards.aliquot_id, 'isabl_sample_id'],
    output:
        temp(os.path.join(output_directory, "preprocessing/hmmcopy/aliquot/{patient_id}/hmmcopy_{aliquot_id}.h5ad"))
    shell:
        """
        {python_bin} {scripts_dir}/create_hmmcopy_anndata.py {input.metrics} {input.reads} \
            {params.patient_id} {params.sample_id} {wildcards.aliquot_id} \
            {output}
        """


def get_concat_hmmcopy_anndata_inputs(wildcards):
    aliquot_ids = hmmcopy_info.set_index('isabl_patient_id').loc[wildcards.patient_id:wildcards.patient_id, 'isabl_aliquot_id'].values
    return {
        'hmmcopy': [os.path.join(output_directory, f"preprocessing/hmmcopy/aliquot/{wildcards.patient_id}", f"hmmcopy_{aliquot_id}.h5ad") for aliquot_id in aliquot_ids],
    }


rule concat_hmmcopy_adata:
    input:
        unpack(get_concat_hmmcopy_anndata_inputs)
    params:
        hmmcopy_args=lambda wildcards, input: ' '.join(f'--input_filename {a}' for a in input.hmmcopy)
    output:
        temp(os.path.join(output_directory, "preprocessing/hmmcopy/patient/hmmcopy_{patient_id}.h5ad"))
    shell:
        """
        {python_bin} {scripts_dir}/concat_anndata.py {params.hmmcopy_args} {output}
        """


def get_create_signals_anndata_inputs(wildcards):
    patient_signals = signals_info.set_index('isabl_patient_id').loc[wildcards.patient_id]
    return {
        'hmmcopy': os.path.join(output_directory, f"preprocessing/hmmcopy/patient/hmmcopy_{wildcards.patient_id}.h5ad"),
        'signals': patient_signals['SIGNALS_hscn'],
        'manual_normal_cells':manual_normal_cells_csv,
        'classifier_normal_cells':os.path.join(output_directory, f"preprocessing/hscn_normal_classifier/{wildcards.patient_id}_normalcells.csv"),
    }


rule create_signals_anndata:
    input:
        unpack(get_create_signals_anndata_inputs)
    output:
        os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5")
    resources:
        mem_mb=64000
    shell:
        """
        {python_bin} {scripts_dir}/create_signals_anndata.py \
            --manual_normal_cells_csv {input.manual_normal_cells} \
            --classifier_normal_cells_csv {input.classifier_normal_cells} \
            --image_features_dir {image_features_dir} \
            --doublets_csv {doublets_csv} \
            --rt_bigwig {rt_bigwig_filename} \
            --s_phase_thresholds_csv {s_phase_thresholds_csv} \
            {wildcards.patient_id} {input.signals} {input.hmmcopy} {output}
        """