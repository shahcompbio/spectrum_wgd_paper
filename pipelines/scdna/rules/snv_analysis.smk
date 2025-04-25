rule infer_sbmclone_tree:
    input:
        snv_adata=output_directory+'/sbmclone/sbmclone_{patient_id}_snv.h5',
    params: 
        patient_id="{patient_id}",
        tree_snv_binarization_threshold=tree_snv_binarization_threshold,
    output:
        output_directory+'/tree_snv/inputs/{patient_id}_clones.pickle',
    shell:
        """
        {python_bin} {scripts_dir}/infer_sbmclone_tree.py --snv_adata {input.snv_adata} \
            --patient_id {params.patient_id} --binarization_threshold {params.tree_snv_binarization_threshold} --output {output}
        """

rule compute_clustered_snv_adata:
    input:
        cna_adata=output_directory+'/preprocessing/signals/signals_{patient_id}.h5',
        snv_adata=output_directory+'/sbmclone/sbmclone_{patient_id}_snv.h5',
        tree=output_directory+'/tree_snv/inputs/{patient_id}_clones.pickle',
        cell_info_filename=output_directory+'/preprocessing/summary/filtered_cell_table_{patient_id}.csv.gz',
    params: 
        patient_id="{patient_id}",
    resources:
        mem_mb=64000
    output:
        clustered_snv_adata=output_directory+'/tree_snv/inputs/{patient_id}_general_clone_adata.h5',
        clustered_cna_adata=output_directory+'/tree_snv/inputs/{patient_id}_cna_clustered.h5',
        pruned_tree=output_directory+'/tree_snv/inputs/{patient_id}_clones_pruned.pickle',
    shell:
        """
        {python_bin} {scripts_dir}/compute_clustered_snv_adata.py \
            --adata_cna {input.cna_adata}  --adata_snv {input.snv_adata} --tree_filename {input.tree} --cell_info_filename {input.cell_info_filename} --patient_id {params.patient_id} \
            --min_clone_size {tree_snv_min_clone_size} \
            --within_clone_homogeneity_threshold {within_clone_cna_homogeneity_threshold} \
            --across_clones_homogeneity_threshold {across_clones_cna_homogeneity_threshold} \
            --output_cn {output.clustered_cna_adata} --output_snv {output.clustered_snv_adata} --output_pruned_tree {output.pruned_tree}
        """

rule tree_snv_model:
    input:
        tree=output_directory+'/tree_snv/inputs/{patient_id}_clones_pruned.pickle',
        adata=output_directory+'/tree_snv/inputs/{patient_id}_general_clone_adata.h5',
    params: 
        patient_id="{patient_id}",
    output:
        table=output_directory+'/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv'
    shell:
        """
        {python_pyro_bin} {scripts_dir}/run_tree_snv_model.py --adata {input.adata} --tree {input.tree} --ref_genome {genome_fasta_filename} --output {output.table}
        """

rule tree_snv_postprocessing:
    input:
        tree=output_directory+'/tree_snv/inputs/{patient_id}_clones_pruned.pickle',
        assignments=output_directory+'/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv',
        snv_adata=output_directory+'/sbmclone/sbmclone_{patient_id}_snv.h5',
        adata=output_directory+'/preprocessing/signals/signals_{patient_id}.h5',
        adata_clusters=output_directory+'/tree_snv/inputs/{patient_id}_cna_clustered.h5',
    output:
        tree=output_directory+'/tree_snv/postprocessing/{patient_id}_clones_pruned.pickle',
        branch_info=output_directory+'/tree_snv/postprocessing/{patient_id}_branch_info.csv.gz',
        cluster_info=output_directory+'/tree_snv/postprocessing/{patient_id}_cluster_info.csv.gz',
        cell_info=output_directory+'/tree_snv/postprocessing/{patient_id}_cell_info.csv.gz',
    shell:
        """
        {python_bin} {scripts_dir}/tree_snv_postprocessing.py \
            {input.tree} \
            {input.assignments} \
            {input.snv_adata} \
            {input.adata} \
            {input.adata_clusters} \
            {output.tree} \
            {output.branch_info} \
            {output.cluster_info} \
            {output.cell_info}
        """

rule tree_snv_notebook:
    input:
        tree=output_directory+'/tree_snv/inputs/{patient_id}_clones_pruned.pickle',
        adata=output_directory+'/tree_snv/inputs/{patient_id}_general_clone_adata.h5',
        table=output_directory+'/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv',
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/tree_snv_qc.md'),
    params: 
        patient_id="{patient_id}",
        job_dir=os.path.join(work_dir, "tree_snv_qc_{patient_id}")
    output:
        notebook=output_directory+'/notebooks/tree_snv_qc/tree_snv_qc_{patient_id}.ipynb',
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        {jupytext_bin} temp_{params.patient_id}.md --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        {python_bin} -m papermill -p patient_id {params.patient_id} -p adata_filename {input.adata} -p tree_filename {input.tree} -p table_filename {input.table} temp_{params.patient_id}_template.ipynb {output.notebook}
        """
