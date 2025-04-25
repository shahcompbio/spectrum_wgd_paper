library(argparse)
library(stringr)
library(plyr)
library(dplyr)

# library(devtools)
# load_all("/juno/work/shah/vazquezi/projects/spectrum/packages/ithi.utils")
# load_all("/juno/work/shah/vazquezi/projects/spectrum/packages/ithi.meta")
# load_all("/juno/work/shah/vazquezi/projects/spectrum/packages/ithi.spatial")
source("/juno/work/shah/vazquezi/projects/spectrum/pipelines/if/summary/R/region_file_summary.R")

parser <- ArgumentParser(description = "Summarize region file.")
parser$add_argument('--region', metavar='FILE', type='character', nargs=1,
                    help="Detection file path")
parser$add_argument('--outfname', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
args <- parser$parse_args()

stats <- region_file_summary(args$region)

readr::write_tsv(stats, args$outfname, na = "")
# write.table(stats, file = args$outfname, sep = "\t", quote = FALSE, row.names = FALSE,
#             col.names = TRUE)

message("Finished.")
