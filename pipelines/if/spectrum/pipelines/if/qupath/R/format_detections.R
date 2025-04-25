library(argparse)
library(tidyverse)
library(plyr)

source("/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/summary/R/detection_file_summary.R")

parser <- ArgumentParser(description = "Format detection file.")
parser$add_argument('--detections', metavar='FILE', type='character', nargs=1,
                    help="Detection file path")
parser$add_argument('--image_id', metavar='FILE', type='character', nargs=1,
                    help="Detection file path")
parser$add_argument('--detections_formatted', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--pn_detections_formatted', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
parser$add_argument('--mn_detections_formatted', metavar='FILE', type='character', nargs=1,
                    help="Output file.")
args <- parser$parse_args()

detections <- vroom::vroom(args$detections)

detections <- detections %>%
  # Add "Image" column if it doesn't exist
  mutate(Image = if ("Image" %in% colnames(.)) Image else args$image_id) %>%
  relocate(Image, .before = 1) %>% # Move it to the first position
  # Add other missing columns
  mutate(Name = if ("Name" %in% colnames(.)) Name else NA_character_) %>%
  mutate(ROI = if ("ROI" %in% colnames(.)) ROI else NA_character_) %>%
  # Conditionally rename "id" to "Object ID" if "Object ID" does not exist
  rename_with(~ ifelse(. == "id" & !"Object ID" %in% colnames(.), "Object ID", .), everything()) # %>%
  # Remove duplicate columns
  # select_if(!duplicated(sub("\\.\\.\\..*", "", names(.))))

# Remove duplicate columns
detections <- detections[, !duplicated(sub("\\.\\.\\..*", "", names(detections)))]
names(detections) <- sub("\\.\\.\\..*", "", names(detections))

detections <- detections %>%
  # Fix image ID
  dplyr::mutate(Image = str_replace_all(Image, "(.mrxs|.tif|.ome.tiff)", "")) %>%
  dplyr::mutate(Image = str_replace_all(Image, " - Image0", "")) %>%
  # Create cell ID
  dplyr::mutate(cell_id = paste(Image, `Object ID`, sep = "_")) %>%
  relocate(cell_id, .before = 2) %>% # Move it to the second position
  # # Keep x coordinates
  # dplyr::mutate(`Centroid X µm` = +`Centroid X µm`) %>%
  # # Flip y coordinates
  # dplyr::mutate(`Centroid Y µm` = max(`Centroid Y µm`) - `Centroid Y µm`) %>%
  # Map channel names
  map_channels()

pn_detections <- filter(detections, str_detect(Class, "Primary nucleus"))

mn_detections <- filter(detections, str_detect(Class, "Micronucleus"))

readr::write_tsv(detections, file = args$detections_formatted, na = "")
readr::write_tsv(pn_detections, file = args$pn_detections_formatted, na = "")
readr::write_tsv(mn_detections, file = args$mn_detections_formatted, na = "")

message("Finished.")