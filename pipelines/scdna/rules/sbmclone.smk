import pandas as pd
import numpy as np

seeds = list(range(sbmclone_restarts))

def get_articull_inputs(wildcards):
    patient_signals = signals_info.set_index('isabl_patient_id').loc[wildcards.patient_id]
    return {
        'mutect': patient_signals['MUTECT_maf'],
        'bamfiles': list(hmmcopy_info.loc[hmmcopy_info.isabl_patient_id == wildcards.patient_id].bamfile)
    }

rule run_articull:
    input:
       unpack(get_articull_inputs)
    params:
        outdir=os.path.join(output_directory, "preprocessing/articull/{patient_id}"),
        bamfiles_arg=lambda wildcards, input:'"' + ' '.join(input.bamfiles) + '"'
    output:
        result=os.path.join(output_directory, "preprocessing/articull/{patient_id}/result.tsv"),
        log=os.path.join(output_directory, "preprocessing/articull/{patient_id}/classification.log"),
    shell:
        """
        export PATH=/home/mcphera1/micromamba/envs/snv_filter/bin/:$PATH
        export PATH=/data1/shahs3/isabl_data_lake/software/dependencies/articull/bedtools2/bin:$PATH

        /data1/shahs3/isabl_data_lake/software/dependencies/articull/scripts/run_classify.sh \
            {input.mutect} \
            {params.outdir} \
            {params.bamfiles_arg} \
            {output.log} \
            {mappability_bedgraph_filename} \
            2
        """


rule create_sbmclone_cell_list:
    input:
        cell_table=os.path.join(output_directory, "preprocessing/summary/cell_table_{patient_id}.csv.gz"),
        cell_nnd=os.path.join(output_directory, "preprocessing/pairwise_distance/cell_nnd_{patient_id}.csv.gz"),
        segment_info=os.path.join(output_directory, "preprocessing/segment_stats/segment_data_{patient_id}.csv.gz"),
    params:
        multipolar_nnd_threshold=multipolar_nnd_threshold,
        patient_id='{patient_id}'
    output:
        cell_list = os.path.join(output_directory, "sbmclone/cell_lists/{patient_id}.txt"),
    shell:
        """
        {python_bin} {scripts_dir}/create_sbmclone_cell_list.py \
            {input.cell_table} {input.cell_nnd} {input.segment_info} {params.patient_id} {params.multipolar_nnd_threshold} {output}
        """

def get_sbmclone_inputs(wildcards):
    patient_hmmcopy_info = hmmcopy_info.set_index('isabl_patient_id').loc[wildcards.patient_id:wildcards.patient_id]
    return {
        'snvgenotyping': patient_hmmcopy_info['MONDRIAN_SNVGENOTYPING_vartrix'].values.tolist(),
    }

rule create_sbmclone_input:
    input:
        unpack(get_sbmclone_inputs),
        articull=os.path.join(output_directory, "preprocessing/articull/{patient_id}/result.tsv"),
        cell_list = os.path.join(output_directory, "sbmclone/cell_lists/{patient_id}.txt"),
    output:
        matrix = os.path.join(output_directory, "sbmclone/{patient_id}/matrix.npz"),
        metadata = os.path.join(output_directory, "sbmclone/{patient_id}/metadata.pickle"),
        alt_matrix = os.path.join(output_directory, "sbmclone/{patient_id}/alt_counts.npz"),
        ref_matrix = os.path.join(output_directory, "sbmclone/{patient_id}/ref_counts.npz"),
    params:
        output_directory=os.path.join(output_directory, "sbmclone/{patient_id}"),
        patient = "{patient_id}",
        snvgenotyping_args = lambda wildcards, input: ' '.join(f'--counts_files {a}' for a in input.snvgenotyping),
    resources:
        mem_mb=128000,
        runtime=60
    #retries: 3
    shell:
        """
            mkdir -p {params.output_directory}
            {python_bin} {scripts_dir}/create_sbmclone_input.py \
            {params.snvgenotyping_args} \
            --filter_files {input.articull} \
            --cell_list {input.cell_list} --output_matrix {output.matrix} \
            --output_altcounts {output.alt_matrix} --output_refcounts {output.ref_matrix} \
            --output_metadata {output.metadata}
        """

rule run_sbmclone:
    input:
        matrix = os.path.join(output_directory, "sbmclone/{patient_id}/matrix.npz"),
    output:
        os.path.join(output_directory, "sbmclone/{patient_id}/results.{seed}.pickle")
    resources:
        runtime=60*24*7
    params:
        seed = "{seed}"
    conda: "sbmclone"
    shell:
        """
        {python_sbmclone_bin} {scripts_dir}/run_sbmclone.py {input.matrix} {output} {params.seed} --max_blocks {sbmclone_max_blocks}
        """

rule process_sbmclone_output:
    input:
        runs = expand(os.path.join(output_directory, "sbmclone/{patient_id}/results.{seed}.pickle"), seed = seeds, allow_missing = True),
        matrix = os.path.join(output_directory, "sbmclone/{patient_id}/matrix.npz"),
        metadata = os.path.join(output_directory, "sbmclone/{patient_id}/metadata.pickle")
    output:
        matrix_fig = os.path.join(output_directory, "sbmclone/{patient_id}/matrix_fig.png"),
        obj_fig = os.path.join(output_directory, "sbmclone/{patient_id}/objectives.png"),
        cell_ari_fig = os.path.join(output_directory, "sbmclone/{patient_id}/row_block_ari.png"),
        row_ari_fig = os.path.join(output_directory, "sbmclone/{patient_id}/col_block_ari.png"),
        density_fig = os.path.join(output_directory, "sbmclone/{patient_id}/density_fig.png"),
        cell_table = os.path.join(output_directory, "sbmclone/{patient_id}/cells.csv"),
        snv_table = os.path.join(output_directory, "sbmclone/{patient_id}/snvs.csv")
        # this script also produces a bunch of other outputs that are not enumerated here
    conda: "sbmclone"
    shell:
        """
        {python_sbmclone_bin} {scripts_dir}/process_sbmclone_output.py {input.runs} {input.matrix} {input.metadata} {output.matrix_fig}
        """

rule create_sbmclone_anndata:
    input:
        signals_adata=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        sbmclone_cells=os.path.join(output_directory, "sbmclone/{patient_id}/cells.csv"),
        sbmclone_snvss=os.path.join(output_directory, "sbmclone/{patient_id}/snvs.csv"),
        sbmclone_altcounts=os.path.join(output_directory, "sbmclone/{patient_id}/alt_counts.npz"),
        sbmclone_refcounts=os.path.join(output_directory, "sbmclone/{patient_id}/ref_counts.npz"),      
    params: 
        sbmclone_results_dir=os.path.join(output_directory, "sbmclone/{patient_id}"),
    resources:
        mem_mb=64000
    output:
        os.path.join(output_directory, "sbmclone/sbmclone_{patient_id}_snv.h5"),
    shell:
        """
        {python_bin} {scripts_dir}/align_snv_cna.py --signals_adata {input.signals_adata} \
            --sbmclone_outdir {params.sbmclone_results_dir} \
            --output {output}
        """

rule concat_sbmclone_anndata_obs:
    input:
        expand(os.path.join(output_directory, "sbmclone/sbmclone_{patient_id}_snv.h5"), patient_id=patient_ids),
    output:
        os.path.join(output_directory, "sbmclone/sbmclone_cell_table.csv.gz")
    shell:
        """
        {python_bin} {scripts_dir}/concatenate_anndata_obs.py {output} {input}
        """

rule sbmclone_notebook:
    input:
        signals=os.path.join(output_directory, "preprocessing/signals/signals_{patient_id}.h5"),
        snv_table=os.path.join(output_directory, "sbmclone/{patient_id}/snvs.csv"),
        cell_table=os.path.join(output_directory, "sbmclone/{patient_id}/cells.csv"),
        template=os.path.join(spectrumanalysis_repo, 'analysis/notebooks/scdna/templates/sbmclone_signatures.md'),
        ref_genome_fasta = genome_fasta_filename,
    params: 
        patient_id="{patient_id}",
        job_dir=os.path.join(work_dir, "signals_{patient_id}")
    output:
        os.path.join(output_directory, "notebooks/sbmclone/sbmclone_{patient_id}.ipynb")
    shell:
        """
        mkdir -p {params.job_dir}
        cd {params.job_dir}
        {python_bin} {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
        jupytext temp_{params.patient_id}.md --set-kernel - --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
        papermill -p signals_adata_filename {input.signals} -p patient_id {params.patient_id} \
                -p snv_block_filename {input.snv_table} -p cell_block_filename {input.cell_table} -p ref_genome_fasta {input.ref_genome_fasta} \
                temp_{params.patient_id}_template.ipynb {output}
        """
