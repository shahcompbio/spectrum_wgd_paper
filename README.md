
# spectrumanalysis: a repo for organization of analyses and results for spectrum

## Setup

### Python package

Spectrum specific python code is provided in the spectrumanalysis python package at `./pysrc`.

Recommended installation:

```
cd pysrc/
pip install -e ./
```

## Data Access

All locations are on juno.

### WGS

#### Sample Metadata and Paths to Data

Sample metadata for bulk tumor and normal samples can be found in the sample_metadata_tumor and sample_metadata_normal tabs of this Google Sheet.

Normal and tumor BAM file locations, somatic SNV and SV calls, copy number calls and germline SNP calls can be found in the wgs_cluster_paths tab of this Google Sheet.

- normal_bam
- tumor_bam
- variants
- copynumber
- breakpoints
- germlines

https://docs.google.com/spreadsheets/d/10YbRXTv59kgRu1Fr2bpMVh8kYDK2zg_PCtJB889KgCg/edit?usp=sharing

owner: Mcpherson, Andrew W./Sloan Kettering Institute Vazquez Garcia, Ignacio

#### Copy Number Data

##### Remixt

Paths to remixt data files can be found in the following notebook.  Also included is code for selecting the correct solution in situations where model selection does not work automatically.

https://github.mskcc.org/shahcompbio/spectrumanalysis/blob/master/analysis/notebooks/bulk-dna/spectrumcnv.ipynb

owner: Mcpherson, Andrew W./Sloan Kettering Institute

##### Isabl Data from the Papaemmanuil Lab

Many tools have been kindly run for us by the Papaemmanuil lab.  The results have been copied to the location below, where the subdirectories are named with the analysis id.  A table of metadata details the sample and application information for each analysis.

Metadata: /work/shah/mcphera1/elli_analyses_metadata.csv

Location: /work/isabl/public/shah/p344/

owner: Mcpherson, Andrew W./Sloan Kettering Institute

#### QC Figures

QC figures for the spectrum samples are located in OneDrive: https://mskcc-my.sharepoint.com/:f:/g/personal/abramsd_mskcc_org/EnaQm4yeGi9GhxVw4WX_Uy0BnrrIyqIKzc4_yc_qNMvaKA?e=oDgjhO

owner: Mcpherson, Andrew W./Sloan Kettering Institute

### DLP Metadata and Paths to Data

#### BC Cancer Spreadsheet

The following google sheet provides metadata, analysis statuses and results and sequence data paths.

Samples and sample metadata:
- samples
- sample_metadata

Analysis status:
- qc_status
- snv_genotyping_status
- split_wgs_bam_status
- infer_haps_status
- pseudobulk_status
- annotation_status
- cellenone

Paths:
- juno_fastq_paths
- juno_bam_paths
- juno_analysis_paths

https://docs.google.com/spreadsheets/d/1_aDR0eaBHVq2L7pn47Vjf3APBlgA5KaIvEJilyZ5-NU/edit?usp=sharing

owner: Mcpherson, Andrew W./Sloan Kettering Institute Vazquez Garcia, Ignacio

##### Important tabs

samples: bccrc identifiers for each sample / library

qc_status: IDs for qc analyses in tantalus, bccrc qc Jira ticket IDs, elab sample id

pseudobulk_status: IDs and statuses for pseudobulk analyses in tantalus, bccrc qc Jira ticket IDs

juno_fastq_paths: paths for raw fastqs, missing paths indicate not yet copied to juno

juno_bam_paths: paths for cell bams, missing paths indicate not yet copied to juno

juno_analysis_paths: paths for analyses by sample / analysis type, missing paths indicate not yet copied to juno

#### BC Cancer / Elab Mapping

The following table provides a mapping from bc cancer sample ids to elab sample ids and spectrum aliquot ids:

https://github.mskcc.org/shahcompbio/spectrumanalysis/blob/master/metadata/dlp_samples.tsv 

owner: Havasov, Eliyahu/Sloan Kettering Institute

### scRNA

#### Sample metadata

Sample metadata for single-cell RNA sequencing data can be found in the sample_metadata tab of this Google Sheet:

https://docs.google.com/spreadsheets/d/1plhIL1rH2IuQ8b_komjAUHKKrnYPNDyhvNNRsTv74u8/edit?usp=sharing

owner: Vazquez Garcia, Ignacio

#### Cohort object

Annotated cohort object v5 lives on isabl:  https://isabl.shahlab.mskcc.org/?project=1&analysis=1652
Direct path to cohort object on juno: /work/shah/isabl_data_lake/analyses/16/52/1652/cohort_merged.rdata

#### Cell type objects

Curated cell type objects with `$cluster_label` for cell sub types are in progress:

T/NK-cell object: /work/shah/uhlitzf/data/SPECTRUM/freeze/v5/T.cell_processed_filtered.rds
Fibroblast object: /work/shah/uhlitzf/data/SPECTRUM/freeze/v5/Fibroblast_processed_filtered.rds

## Results

### Mutational signatures

curated table of mutational signatures and recurrent drivers:
https://github.com/shahcompbio/spectrumanalysis/blob/master/results/impact/v1/signatures/outputs/signatures/mutational_signatures_summary.tsv

For mutational signatures, there are three types of columns:

wgs_signature is our current best guess by MMCTM signature inference based on SNVs and SVs
impact_signature is derived using IMPACT mutation calls and CNAs alone, by filtering pathogenic variants that are highly associated with previously characterised SNV+SV mutational signatures
consensus_signature uses wgs_signature when available, and impact_signature  when WGS data isn’t available.

Each of these signatures can be assigned as HRD-Dup, HRD-Del, HRD-Other, FBI, TD, or Ambiguous. Note that the HRD-Other category captures HR genes that are much rarer than BRCA1/2, but which could engender comparable WGS signatures. There is also a consolidated column consensus_signature_short which aggregates all HRD cases in consensus_signature together, into HRD, FBI, TD or Ambiguous. I would recommend that you use either consensus_signature or consensus_signature_short excluding Ambiguous cases for correlative analyses.

To derive impact_signature, I am doing the following filtering in IMPACT mutation and CNA calls:

- Keeping missense/frameshift SNVs/indels and CNAs
- Keeping >=2 log-fold high-level amplifications in oncogenes and <= -2 log-fold deletions in TSGs (based on Cancer Gene Census definition)

I am then applying the following XOR logic per IMPACT sample:

- HRD-Dup: Any missense/frameshift BRCA1 SNV/indel or any BRCA1 deep deletion
- HRD-Del: Any missense/frameshift BRCA2 SNV/indel or any BRCA2 deep deletion
- HRD-Other: Any missense/frameshift SNV/indel in ATM,PALB2,RAD50,RAD51B,RAD51C,RAD51D,RAD52,RAD54B,RAD54L
- FBI: Any high-level amplification in CCNE1
- TD: Any missense/frameshift CDK12 SNV/indel
- Ambiguous: Any other sample without these SNVs/indels or CNAs

The table also contains a summary of mutations in a shortlist of genes within the IMPACT panel, particularly genes involve in DNA damage/repair (e.g. BRCA1/2, CDK12), cell cycle (e.g. CCNE1, RB1) or the KRAS pathway (PIK3CA, NF1, KRAS, PTEN). For SNVs/indels, the germline/somatic mutation status and the amino acid change is listed. I also added a column with the URLs to the patient-centric view of each of these patients in cBioPortal.

This version is currently lacking IMPACT results and signatures for two patients (OV-077 and OV-078). I will get the IMPACT results for these two patients early next week.

owner: Vazquez Garcia, Ignacio

### WGS MMCTM based signature assignment table

See the mutational_signatures_tumour tab here: https://docs.google.com/spreadsheets/d/10YbRXTv59kgRu1Fr2bpMVh8kYDK2zg_PCtJB889KgCg/edit#gid=2077145616

owner: Vazquez Garcia, Ignacio

### scRNA

#### Cell type compositions (TSV) (florian)

* Embeddings and bar plots are here: https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/cell_type_compositions/001_major_cell_type_compositions.html
* TSV is here: https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/cell_type_compositions/tab/major_cell_type_compositions.tsv

#### Cell subtype compositions (TSV) (florian)

#### HLA LOH of cancer clusters in scRNA (TSV) (Allen)

* HLA allele-specific expression and LOH results, along with integrative analysis using available data (major cell type proportions, IFNG pathway expression): https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/hla_loss/v9/hla_loh_report.html

#### InferCNV (Hongyu)
* InferCNV heatmaps:
https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/infercnv/SPECTRUM_infercnv_heatmap_0601.html
* InferCNV based phylogenetic Mean Pairwise Distance (MPD) (CSV):
https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/infercnv/infercnv_phylo_diveristy_mpd.csv
* InferCNV based phylogenetic Mean Pairwise Distance (MPD) (barplot):
https://github.com/shahcompbio/spectrumanalysis/blob/master/results/scrna/infercnv/barplot_infercnv_mpd_diversity_0601.pdf

### WGS

#### HLA LOH of WGS samples (TSV) (allen)

* Annotated LOHHLA output: https://github.com/shahcompbio/spectrumanalysis/blob/master/results/wgs/lohhla/v1/hla_loss_filtered.tsv

#### ecDNA patients (marc)

### scWGS

#### Cell-clone assignments (TSV) (Ignacio)

#### Tree (newick) (Ignacio)

#### Rates of mitotic error per sample (TSV) (marc)

#### Homozygous deleted bins per cell (TSV) (ignacio)

#### High-level amplifications per cell (TSV) (ignacio)

#### Clone-by-gene copy number matrix (Ignacio)

#### Morphology summary table with diameter, elongation and circularity (TSV) (ignacio)

#### ASCN/HSCN (rdata object) (marc/ignacio)

### H&E

#### Cell type compositions (TSV) (ignacio)

#### Tumor/immune/stroma (based on object detection)

#### TIL presence in tumour/stroma (TSV) (TBD)

#### Immune/non-immune (based on object detection)

#### Tumor/stroma (based on region detection)

#### Immune object density in epithelial vs stromal regions

### Flow

#### Live/dead and CD45+/- counts and proportions per SCS aliquot (TSV) (ignacio)

### Clinical

#### Treatments + treatment regimens (TSV) (Ignacio)

#### Laboratory test (TSV) (Ignacio)

