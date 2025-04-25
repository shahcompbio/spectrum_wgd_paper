n_cell_splits = 10

rule split_cells:
    input:
        os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5")
    output:
        temp(expand(os.path.join(work_dir, 'pairwise_distances_{{patient_id}}', 'cell_ids_{{patient_id}}.part_00{n}.csv.gz'), n=range(n_cell_splits)))
    shell:
        """
        {python_bin} {scripts_dir}/create_cell_splits.py {input} {n_cell_splits} {output}
        """

rule compute_pairwise_distances:
    input:
        adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        cells=os.path.join(work_dir, 'pairwise_distances_{patient_id}', 'cell_ids_{patient_id}.part_00{n}.csv.gz')
    output:
        temp(os.path.join(work_dir, 'pairwise_distances_{patient_id}', 'pairwise_distances_{patient_id}.part_00{n}.csv.gz'))
    shell:
        """
        {python_bin} {scripts_dir}/compute_pairwise_distances.py {input.adata} {input.cells} {output}
        """

rule merge_pairwise_distances:
    input:
        expand(os.path.join(work_dir, 'pairwise_distances_{{patient_id}}', 'pairwise_distances_{{patient_id}}.part_00{n}.csv.gz'), n=range(n_cell_splits))
    output:
        os.path.join(output_directory, "preprocessing/pairwise_distance/pairwise_distance_{patient_id}.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py  --compressed {output} {input}
        """

rule compute_cell_nnd:
    input:
        distances=os.path.join(output_directory, "preprocessing/pairwise_distance/pairwise_distance_{patient_id}.csv.gz"),
        adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
    output:
        os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd_{patient_id}.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/compute_cell_nnd.py {input.adata} {input.distances} {output}
        """

rule merge_cell_nnd:
    input:
        expand(os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd_{patient_id}.csv.gz"), patient_id=patient_ids)
    output:
        os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concat_csvs.py --compressed {output} {input}
        """
