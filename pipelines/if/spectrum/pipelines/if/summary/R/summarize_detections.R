library(argparse)
library(stringr)
library(plyr)
library(dplyr)

source("/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/summary/R/detection_file_summary.R")

parser <- ArgumentParser(description = "Summarize detection file.")
parser$add_argument('--detection', metavar='FILE', type='character', nargs=1,
                    help="Detection file path")
parser$add_argument('--phenotype', metavar='FILE', type='character', nargs=1,
                    help="Phenotype file path")
parser$add_argument('--segmented', action = 'store_true',
                    help = "Set flag if detection file is segmented.")
parser$add_argument('--thresholded', action = 'store_true',
                    help = "Set flag if detection file is segmented.")
parser$add_argument('--detection_summary_slide', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--detection_summary_roi', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--detection_summary_region', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--assigned_detection_summary', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--binned_detection_summary', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
args <- parser$parse_args()

detection <- vroom::vroom(args$detection)

if (!args$segmented && !args$thresholded) {
  # Slide
  common_cols <- c("cell_id", "Object ID", "objectType", "Parent ID", "ROI ID", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  grouping_cols <- c("Image", "Name", "Class", "Parent", "ROI")
  detection_summary_slide <- detection_file_summary(detection, common_cols, grouping_cols, type = "tbl")
  # ROI
  common_cols <- c("cell_id", "Object ID", "objectType", "Parent ID", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  grouping_cols <- c("Image", "Name", "Class", "Parent", "ROI ID", "ROI")
  detection_summary_roi <- detection_file_summary(detection, common_cols, grouping_cols, type = "tbl")
  # Region
  common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI")
  detection_summary_region <- detection_file_summary(detection, common_cols, grouping_cols, type = "tbl")

  common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI", "Nearest Micronuclei Count")
  assigned_stats <- assigned_detection_file_summary(detection, common_cols, grouping_cols, type = "tbl")

  common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI")
  binned_stats <- binned_detection_file_summary(detection, common_cols, grouping_cols, type = "tbl")
} 
if (args$segmented) {
  stats <- segmented_detection_file_summary(args$detection)
} 
if (args$thresholded) {
  stats <- thresholded_detection_file_summary(args$detection, args$phenotype)
  # binned_stats <- thresholded_binned_detection_file_summary(args$detection, args$phenotype)
}

readr::write_tsv(detection_summary_slide, file = args$detection_summary_slide, na = "")
readr::write_tsv(detection_summary_roi, file = args$detection_summary_roi, na = "")
readr::write_tsv(detection_summary_region, file = args$detection_summary_region, na = "")
readr::write_tsv(assigned_stats, file = args$assigned_detection_summary, na = "")
readr::write_tsv(binned_stats, file = args$binned_detection_summary, na = "")

message("Finished.")