rule baf_filtering_notebook:
    input:
        signals=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/baf_filtering.md'), 
    params: 
        patient_id="{patient_id}",
        job_dir=os.path.join(work_dir, "signals_{patient_id}")
    output:
        os.path.join(output_directory, "notebooks/baf_filtering/baf_filtering_{patient_id}.ipynb")
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        jupytext temp_{params.patient_id}.md --set-kernel - --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        papermill -p filename {input.signals} -p patient_id {params.patient_id} temp_{params.patient_id}_template.ipynb {output}
        """

rule s_phase_classification_notebook:
    input:
        signals=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        s_phase_thresholds_csv=s_phase_thresholds_csv,
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/s_phase_classification.md')
    params: 
        job_dir=os.path.join(work_dir, "s_phase_classification_{patient_id}"),
        patient_id="{patient_id}",
    output:
        os.path.join(output_directory, "notebooks/s_phase_classification/s_phase_classification_{patient_id}.ipynb")
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        jupytext temp_{params.patient_id}.md --set-kernel - --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        papermill -p filename {input.signals} -p patient_id {params.patient_id} -p s_phase_thresholds_csv {input.s_phase_thresholds_csv} temp_{params.patient_id}_template.ipynb {output}
        """

rule s_phase_qc:
    input:
        os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5")
    output:
        os.path.join(output_directory, "preprocessing/s_phase_qc/s_phase_qc_{patient_id}.pdf")
    shell:
        """
        {python_bin} {scripts_dir}/generate_sphase_qc.py {input} {output}
        """

rule write_anndata_obs:
    input:
        os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5")
    output:
        os.path.join(output_directory, "preprocessing/summary/cell_table_{patient_id}.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/write_anndata_obs.py {input} {output}
        """

rule concat_anndata_obs:
    input:
        expand(os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"), patient_id=patient_ids),
    output:
        os.path.join(output_directory, "preprocessing/summary/cell_table.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concatenate_anndata_obs.py {output} {input}
        """

rule segment_stats:
    input:
        signals_adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        whitelist_bins=whitelist_bins_file,
    output:
        os.path.join(output_directory, "preprocessing/segment_stats/segment_data_{patient_id}.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/compute_segment_stats.py {input.signals_adata} {input.whitelist_bins} {output}
        """

rule concat_segment_stats:
    input:
        expand(os.path.join(output_directory, "preprocessing/segment_stats/segment_data_{patient_id}.csv.gz"), patient_id=patient_ids),
    output:
        os.path.join(output_directory, "preprocessing/segment_stats/segment_data.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py --compressed {output} {input}
        """

# Depends on inputs from other sections of the pipeline
rule filter_cells:
    input:
        cell_table=os.path.join(output_directory, "preprocessing/summary/cell_table_{patient_id}.csv.gz"),
        cell_nnd=os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd_{patient_id}.csv.gz"),
        segment_info=os.path.join(output_directory, "preprocessing/segment_stats/segment_data_{patient_id}.csv.gz"),
    params:
        multipolar_nnd_threshold=multipolar_nnd_threshold,
    output:
        os.path.join(output_directory, "preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz"),
    shell:
        """
        {python_bin} {scripts_dir}/filter_cells.py \
            {input.cell_table} {input.cell_nnd} {input.segment_info} {params.multipolar_nnd_threshold} {output}
        """

rule concat_cell_table:
    input:
        expand(os.path.join(output_directory, "preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz"), patient_id=patient_ids),
    output:
        os.path.join(output_directory, "preprocessing/summary/filtered_cell_table.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py --compressed {output} {input}
        """

rule qc_signals_anndata:
    input:
        signals=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/clustering_qc.md'), 
    params: 
        patient_id="{patient_id}",
        job_dir=os.path.join(work_dir, "signals_{patient_id}")
    output:
        os.path.join(output_directory, "notebooks/signals/signals_{patient_id}.ipynb")
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        jupytext temp_{params.patient_id}.md --set-kernel - --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        papermill -p filename {input.signals} -p patient_id {params.patient_id} temp_{params.patient_id}_template.ipynb {output}
        """


def get_hscn_normal_classifier_inputs(wildcards):
    patient_signals = signals_info.set_index('isabl_patient_id').loc[wildcards.patient_id]
    return {
        'qc': patient_signals['SIGNALS_hscn'].replace('hscn.csv.gz', 'qc.csv'),
        'hscn': patient_signals['SIGNALS_hscn'],   
    }

rule hscn_normal_classifier:
    input:
        unpack(get_hscn_normal_classifier_inputs)
    output:
        assignment = os.path.join(output_directory, "preprocessing/hscn_normal_classifier/{patient_id}_normalcells.csv"),
        heatmap = os.path.join(output_directory, "preprocessing/hscn_normal_classifier/{patient_id}_heatmap.png")
    resources: mem_mb=1024 * 12
    singularity: signals_singularity_image
    script: os.path.join(spectrumanalysis_repo, 'pipelines', 'scdna', 'bin', 'assign_normal_cells.R')


rule merge_hscn_normal_classifier:
    input: 
        expand(os.path.join(output_directory, "preprocessing/hscn_normal_classifier/{patient_id}_normalcells.csv"), patient_id = patient_ids)
    output:
        merged = os.path.join(output_directory, "preprocessing/hscn_normal_classifier/normalcells_spectrum.csv")
    resources:
        mem_mb=1025*10
    run:
        dist_list = []
        for f in input:
            dist_temp = pd.read_csv(f)
            dist_list.append(dist_temp)
        dist = pd.concat(dist_list, ignore_index=True)
        dist.to_csv(output.merged, sep=',')
