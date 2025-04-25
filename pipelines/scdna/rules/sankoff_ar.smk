def get_compute_sankoff_ancestral_reconstruction_inputs(wildcards):
    patient_medicc = medicc_info.set_index('isabl_patient_id').loc[wildcards.patient_id]
    return {
        'medicc_cn': patient_medicc['medicc_cn_filename'],
        'tree': patient_medicc['tree_filename'],
        'events': patient_medicc['events_filename'],
    }

def get_medicc_tree(wildcards):
    patient_medicc = medicc_info.set_index('isabl_patient_id').loc[wildcards.patient_id]
    return patient_medicc['tree_filename']

# Compute an alternative ancestral reconstruction using the Sankoff algorithm
# using input medicc2 copy number and tree files.  Also use a greedy algorith
# to call events from the Sankoff reconstruction, including pre/post WGD events.
rule compute_sankoff_ancestral_reconstruction:
    input:
        unpack(get_compute_sankoff_ancestral_reconstruction_inputs),
        cell_info_filename=os.path.join(output_directory, "preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz"),
    params:
        sankoff_ar_directory=os.path.join(output_directory, "postprocessing/sankoff_ar/"),
        output_dir=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}"),
        patient_id="{patient_id}",
    output:
        adata=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}/sankoff_ar_{patient_id}.h5"),
        tree_pickle=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}/sankoff_ar_tree_{patient_id}.pickle"),
        events=os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_copynumber_events_df.tsv"),
        overlaps=os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_events_overlap.tsv"),
    priority: 1
    shell:
        """
        mkdir -p {params.sankoff_ar_directory}
        mkdir -p {params.output_dir}
        {python_bin} {scripts_dir}/compute_sankoff_ancestral_reconstruction.py \
            {input.medicc_cn} {input.tree} {input.events} {input.cell_info_filename} {params.patient_id} \
            {output.adata} {output.tree_pickle} {output.events} {output.overlaps}
        """


rule merge_sankoff_ar_events:
    input:
        events=expand(os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_copynumber_events_df.tsv"), patient_id=patient_ids),
    output:
        merged_events=os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/sankoff_ar_events.tsv"),
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py --tsv {output.merged_events} {input.events}
        """


rule compute_sankoff_ar_rates:
    input:
        events=os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_copynumber_events_df.tsv"),
        adata=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}/sankoff_ar_{patient_id}.h5"),
        cell_info=os.path.join(output_directory, "preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz"),
    output:
        rates=os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_rates.tsv"),
    shell:
        """
        {python_bin} {scripts_dir}/compute_misseg_rates.py {input.events} {input.adata} {input.cell_info} {output.rates}
        """


rule merge_sankoff_ar_rates:
    input:
        rates=expand(os.path.join(output_directory, "postprocessing/sankoff_ar/greedy_events/{patient_id}/sankoff_ar_{patient_id}_rates.tsv"), patient_id=patient_ids),
    output:
        merged_rates=os.path.join(output_directory, "postprocessing/sankoff_ar/sankoff_ar_rates.tsv"),
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py --tsv {output.merged_rates} {input.rates}
        """


# QC notebook of Sankoff ancestral reconstruction
rule run_sankoff_ar_notebook:
    input:
        adata=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}/sankoff_ar_{patient_id}.h5"),
        tree=os.path.join(output_directory, "postprocessing/sankoff_ar/{patient_id}/sankoff_ar_tree_{patient_id}.pickle"),
        events=os.path.join(output_directory, "postprocessing/sankoff_ar/{events_key}_events/{patient_id}/sankoff_ar_{patient_id}_copynumber_events_df.tsv"),
        overlaps=os.path.join(output_directory, "postprocessing/sankoff_ar/{events_key}_events/{patient_id}/sankoff_ar_{patient_id}_events_overlap.tsv"),
        chr2centro_filename=chr2centro_filename,
    params: 
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/sankoff_reconstruction.md'),
        job_dir=os.path.join(work_dir, "{events_key}/sankoff_ar_{patient_id}"),
        patient_id="{patient_id}",
    resources:
        mem_mb=64000
    output:
        os.path.join(output_directory, "notebooks/sankoff_ar/{events_key}_events/sankoff_ar_{patient_id}.ipynb")
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {params.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        jupytext temp_{params.patient_id}.md --set-kernel - --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        papermill -p filename {input.adata} -p tree_filename {input.tree} -p patient_id {params.patient_id} -p events_filename {input.events}  -p overlaps_filename {input.overlaps} -p chr2centro_filename {input.chr2centro_filename} temp_{params.patient_id}_template.ipynb {output}
        """
