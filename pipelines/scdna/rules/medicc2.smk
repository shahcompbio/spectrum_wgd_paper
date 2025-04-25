    
medicc_args = """-j 24 --input-type t --verbose --plot none --no-plot-tree --events --wgd-x2 \
--chromosomes-bed /data1/shahs3/isabl_data_lake/software/dependencies/medicc2/medicc/objects/hg19_chromosome_arms.bed \
--regions-bed /data1/shahs3/users/myersm2/repos/medicc2_nextflow/Davoli_2013_TSG_OG_genes_hg37.bed"""
    
rule create_medicc2_cell_list:
    input:
        cell_table=os.path.join(output_directory, "preprocessing/summary/cell_table_{patient_id}.csv.gz"),
        cell_nnd=os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd_{patient_id}.csv.gz"),
        segment_info=os.path.join(output_directory, "preprocessing/segment_stats/segment_data_{patient_id}.csv.gz"),
    params:
        multipolar_nnd_threshold=multipolar_nnd_threshold,
        patient_id='{patient_id}'
    output:
        cell_list = os.path.join(output_directory, 'medicc', 'cell_lists', "{patient_id}.txt"),
    shell:
        """
        {python_bin} {scripts_dir}/create_medicc2_cell_list.py \
            {input.cell_table} {input.cell_nnd} {input.segment_info} {params.patient_id} {params.multipolar_nnd_threshold} {output}
        """


rule resegment_mspcf:
    input:
        adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        cell_list = os.path.join(output_directory, 'medicc', 'cell_lists', "{patient_id}.txt"),
        chr2centro_filename=chr2centro_filename,
    output:
        resegmented_cn = os.path.join(output_directory, 'medicc', 'resegmented_cn', "{patient_id}.csv.gz"),
    shell:
        """
        {python_bin} {scripts_dir}/compute_segments_mspcf.py {input.adata} {input.cell_list} {input.chr2centro_filename} {output.resegmented_cn}
        """

rule create_medicc2_input:
    input:
        unpack(get_create_signals_anndata_inputs),
        segments=os.path.join(output_directory, 'medicc', 'resegmented_cn', "{patient_id}.csv.gz"),
        cell_list=os.path.join(output_directory, 'medicc', 'cell_lists', "{patient_id}.txt")
    output:
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}.tsv')
    shell:
        """
        {python_bin} {scripts_dir}/create_medicc2_input.py {output} --signals_results {input.signals} --segments_filename {input.segments} --allele_specific --cell_list {input.cell_list}
        """

rule run_medicc2_serial:
    input:
       os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}.tsv')
    params:
        medicc_args=medicc_args,
        work_dir=os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}')
    threads: 24
    resources:
        mem_mb=128000,
        runtime=24*60*3
    output:
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_pairwise_distances.tsv'),
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_copynumber_events_df.tsv'),
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_final_tree.new'),
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_events_overlap.tsv'),
        os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_final_cn_profiles.tsv'),
    shell:
        """
        cd {params.work_dir}
        {medicc2_bin} {params.medicc_args} {input} ./
        """

    
rule create_rconnect_html:
    input:
        os.path.join(output_directory, "preprocessing/summary/filtered_cell_table.csv.gz"),
        expand(os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"), patient_id=patient_ids),
        expand(os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_copynumber_events_df.tsv'), patient_id=patient_ids),
        expand(os.path.join(output_directory, 'medicc/output', '{patient_id}' + f'{medicc2_suffix}', '{patient_id}' + f'{medicc2_suffix}_final_tree.new'), patient_id=patient_ids),
        expand(output_directory+'/sbmclone/sbmclone_{patient_id}_snv.h5', patient_id=patient_ids),        
        expand(output_directory+'/tree_snv/inputs/{patient_id}_general_clone_adata.h5', patient_id=tree_snv_patient_ids),
        expand(output_directory+'/tree_snv/postprocessing/{patient_id}_clones_pruned.pickle', patient_id=tree_snv_patient_ids),
        expand(output_directory+'/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv', patient_id=tree_snv_patient_ids),
    params:
        medicc_output_dir=os.path.join(output_directory, 'medicc/output'),
        medicc2_suffix=medicc2_suffix
    resources:
        mem_mb=64000
    output:
        os.path.join(output_directory, 'notebooks/rconnect.html')
    shell:
        """
        {python_bin} {scripts_dir}/create_rconnect_html.py {output_directory} {spectrumanalysis_repo} {params.medicc_output_dir} {params.medicc2_suffix} \
            {infercnv_plot_dir} {output} --has_doubletime --has_sbmclone 
        """