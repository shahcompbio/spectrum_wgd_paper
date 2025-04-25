

rule compute_amplifications:
    input:
        adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        cell_info=os.path.join(output_directory, "preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz"),
    output:
        amp_events=output_directory+"/amplifications/{patient_id}/amp_events_{patient_id}.csv",
    shell:
        """
        {python_bin} {scripts_dir}/compute_amplifications.py {input.adata} {input.cell_info} {output.amp_events}
        """


rule merge_amplifications:
    input:
        amp_events=expand(output_directory+"/amplifications/{patient_id}/amp_events_{patient_id}.csv", patient_id=patient_ids),
    output:
        merged_amp_events=output_directory+"/amplifications/amp_events.csv",
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py {output.merged_amp_events} {input.amp_events}
        """


rule annotate_amplifications:
    input:
        amp_events=output_directory+"/amplifications/amp_events.csv",
        gtf=gtf_filename,
    output:
        amp_events_genes=output_directory+"/amplifications/amp_events_genes.csv",
    shell:
        """
        {python_bin} {scripts_dir}/annotate_amplifications.py {input.amp_events} {input.gtf} {output.amp_events_genes}
        """


