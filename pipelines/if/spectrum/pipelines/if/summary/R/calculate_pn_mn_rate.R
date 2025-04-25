library(tidyverse)
library(argparse)

parser <- ArgumentParser(description = "Calculate MN rates and summary statistics.")
parser$add_argument('--detection_summaries', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_slide', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_roi', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_region', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_compartment_slide', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_compartment_roi', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
parser$add_argument('--pn_mn_rates_compartment_region', metavar='FILE', type='character', nargs='+',
                    help="Detection file path")
args <- parser$parse_args()

detection_summaries_tbl <- readr::read_tsv(args$detection_summaries)

# Define function to calculate rates
calculate_rates <- function(data) {
  data %>%
    dplyr::mutate(
      total_count = `Primary nucleus_count` + `Micronucleus_count`,
      micronucleus_rate = `Micronucleus_count` / `Primary nucleus_count`,
      micronucleus_rate_rank = dense_rank(dplyr::desc(micronucleus_rate)),
      scaled_micronucleus_rate_rank = micronucleus_rate_rank / max(micronucleus_rate_rank, na.rm = TRUE),
      nucleus_type_ratio = `Micronucleus_count` / total_count,
      nucleus_type_rank = dense_rank(dplyr::desc(nucleus_type_ratio)),
      scaled_nucleus_type_rank = nucleus_type_rank / max(nucleus_type_rank, na.rm = TRUE)
    )
}

# detection_summaries_tbl <- detection_summaries_tbl %>%
#   mutate(
#     Subclass = Class,
#     Class = case_when(
#       Subclass %in% c("Micronucleus cGAS LSBio", "Micronucleus cGAS CST") ~ "Micronucleus",
#       Subclass == "Primary nucleus" ~ "Primary nucleus",
#       TRUE ~ "Other"
#     )
#   )

# x <- detection_summaries_tbl %>%
#   # filter(Class == "Primary nucleus" | Name == "Micronucleus") %>%
#   # filter(Parent != "Glass") %>%
#   group_by(Image, Class) %>%
#   dplyr::summarize(count = sum(count)) %>%
#   ungroup %>%
#   pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count")

# print(x)

# y <- x %>%
#   expand(Image, Class, Subclass)

# print(y)

pn_mn_rates_slide_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>%
  calculate_rates

readr::write_tsv(pn_mn_rates_slide_tbl, args$pn_mn_rates_slide, na = "")

pn_mn_rates_roi_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `ROI ID`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>%
  calculate_rates

readr::write_tsv(pn_mn_rates_roi_tbl, args$pn_mn_rates_roi, na = "")

pn_mn_rates_region_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `ROI ID`, `Parent ID`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>%
  calculate_rates

readr::write_tsv(pn_mn_rates_region_tbl, args$pn_mn_rates_region, na = "")

pn_mn_rates_compartment_slide_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `Parent`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>% 
  calculate_rates

readr::write_tsv(pn_mn_rates_compartment_slide_tbl, args$pn_mn_rates_compartment_slide, na = "")

pn_mn_rates_compartment_roi_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `ROI ID`, `Parent`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>% 
  calculate_rates

readr::write_tsv(pn_mn_rates_compartment_roi_tbl, args$pn_mn_rates_compartment_roi, na = "")

pn_mn_rates_compartment_region_tbl <- detection_summaries_tbl %>%
  group_by(`Image`, `Name`, `ROI`, `ROI ID`, `Parent`, `Parent ID`, `Class`) %>%
  dplyr::summarize(count = sum(count)) %>%
  ungroup %>%
  pivot_wider(names_from = Class, values_from = c("count"), names_glue = "{Class}_count") %>%
  calculate_rates

readr::write_tsv(pn_mn_rates_compartment_region_tbl, args$pn_mn_rates_compartment_region, na = "")
