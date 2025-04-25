# #' Summarize detection file from Finn
# #' 
# #' @param f Filename
# #' @param type file or table
# #' @export
# detection_file_summary <- function(x, type = "file") {
#   if (type == "file") {
#     detections <- data.table::fread(x)
#     image_id <- parse_sample_ids(x)
#   } else {
#     detections <- x
#     image_id <- image_id
#   }
  
#   tb <- table(detections$Class)
#   tb_proportions <- tb/sum(tb)
  
#   if ("Tumor" %in% names(tb)) {
#     tumor_quantity <- tb["Tumor"]
#     tumor_proportions <- tb_proportions["Tumor"]
#   } else {
#     tumor_quantity <- 0
#     tumor_proportions <- 0
#   }
  
#   if ("Immune cells" %in% names(tb)) {
#     immune_quantity <- tb["Immune cells"]
#     immune_proportions <- tb_proportions["Immune cells"]
#   } else {
#     immune_quantity <- 0
#     immune_proportions <- 0
#   }

#   if ("Stroma" %in% names(tb)) {
#     stroma_quantity <- tb["Stroma"]
#     stroma_proportions <- tb_proportions["Stroma"]
#   } else {
#     stroma_quantity <- 0
#     stroma_proportions <- 0
#   }
  
#   df <- data.frame(image_id = image_id, tumor_count=tumor_quantity, immune_count=immune_quantity, stroma_count=stroma_quantity,
#                    tumor_prop=tumor_proportions, immune_prop=immune_proportions, stroma_prop=stroma_proportions)
#   rownames(df) <- NULL
#   return(df)
# }

detection_file_summary <- function(x, common_cols, grouping_cols, type = "file") {
  if (type == "file") {
    detections <- vroom::vroom(x)
    # image_id <- parse_sample_ids(x)
  } else {
    detections <- x
    # image_id <- image_id
  }
  
  # detections <- detections %>%
  #   # janitor::clean_names() %>%
  #   # Fix image ID
  #   dplyr::mutate(Image = str_replace_all(Image, ".mrxs", "")) %>%
  #   # Map channel names
  #   map_channels()
  # common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID")
  # common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  # grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI")

  # Identify available columns
  available_common_cols <- intersect(common_cols, colnames(detections))
  available_grouping_cols <- intersect(grouping_cols, colnames(detections))

  # Inform about missing columns (optional)
  missing_common_cols <- setdiff(common_cols, colnames(detections))
  missing_grouping_cols <- setdiff(grouping_cols, colnames(detections))

  if (length(missing_common_cols) > 0) {
    message("The following common columns are missing and will be ignored: ", paste(missing_common_cols, collapse = ", "))
  }
  if (length(missing_grouping_cols) > 0) {
    message("The following grouping columns are missing and will be ignored: ", paste(missing_grouping_cols, collapse = ", "))
  }

  counts <- detections %>%
    dplyr::select(-c(all_of(available_common_cols), contains("phenotype"), ends_with("state"))) %>%
    group_by(across(all_of(available_grouping_cols))) %>%
    dplyr::summarize(count = n()) %>%
    ungroup() %>%
    group_by(Image) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    ungroup()
    
  features <- detections %>%
    dplyr::select(-c(all_of(available_common_cols), contains("phenotype"), ends_with("state"))) %>%
    group_by(across(all_of(available_grouping_cols))) %>%
    dplyr::summarize(across(everything(), 
                            list(mean = mean, 
                                sd = sd, 
                                min = min, 
                                max = max, 
                                sum = sum), 
                            .names = "{.col}_{.fn}"), 
                    .groups = "drop")
    # dplyr::summarize_all(
    #   list(
    #     mean = mean, 
    #     sd = sd, 
    #     min = min, 
    #     max = max, 
    #     sum = sum
    #   )
    # ) %>%
    # ungroup()
  
  df <- dplyr::left_join(counts, features)
  
  return(df)
}

assigned_detection_file_summary <- function(x, common_cols, grouping_cols, type = "file") {
  if (type == "file") {
    detections <- vroom::vroom(x)
    # image_id <- parse_sample_ids(x)
  } else {
    detections <- x
    # image_id <- image_id
  }
  # common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID")
  # common_cols <- c("cell_id", "Object ID", "index", "pn_index", "Nearest Primary Nucleus ID")
  # common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  # grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI", "Nearest Micronuclei Count")

  # Identify available columns
  available_common_cols <- intersect(common_cols, colnames(detections))
  available_grouping_cols <- intersect(grouping_cols, colnames(detections))

  # Inform about missing columns (optional)
  missing_common_cols <- setdiff(common_cols, colnames(detections))
  missing_grouping_cols <- setdiff(grouping_cols, colnames(detections))

  counts <- detections %>%
    dplyr::select(-c(all_of(available_common_cols), contains("phenotype"), ends_with("state"))) %>%
    group_by(across(all_of(available_grouping_cols))) %>%
    # group_by(across(all_of(available_grouping_cols))) %>%
    # count("Nearest Micronuclei Count"=`Nearest Micronuclei Count`) %>%
    dplyr::summarize(count = n()) %>%
    ungroup() %>%
    group_by(Image) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    ungroup()

  print(head(counts))

  features <- detections %>%
    dplyr::select(-c(all_of(available_common_cols), contains("phenotype"), ends_with("state"))) %>%
    group_by(across(all_of(available_grouping_cols))) %>%
    # group_by(`Image`, `Name`, `Class`, `Parent ID`, `Parent`, `ROI`, `Nearest Micronuclei Count`) %>%
    dplyr::summarize(across(everything(), 
                            list(mean = mean, 
                                sd = sd, 
                                min = min, 
                                max = max, 
                                sum = sum), 
                            .names = "{.col}_{.fn}"), 
                    .groups = "drop")
    # dplyr::summarize_all(
    #   list(
    #     mean = mean, 
    #     sd = sd, 
    #     min = min, 
    #     max = max, 
    #     sum = sum
    #   )
    # ) %>%
    # ungroup()

  print(head(features))

  df <- dplyr::left_join(counts, features)

  print(head(df))
    
  return(counts)
}

#' Summarize segmented detection file
#' 
#' @param f Input file
#' @export
segmented_detection_file_summary <- function(f) {
  detections <- data.table::fread(f)
  
  mask_cols <- c("tumor_mask", "stroma_mask", "whitespace_mask")
  # voa <- basename(dirname(f)) %>% stringr::str_replace("_", " ")
  
  df <- lapply(mask_cols, function(masktype) {
    detections_sub <- detections[as.vector(detections[,masktype,with=FALSE] == 1),]
    summary <- detection_file_summary(detections_sub, type = "table")
    data.frame(masktype=stringr::str_replace(masktype, "_mask$", ""), summary)
  }) %>% rbind.fill
  return(df)
}

#' Summarize thresholded detection file
#' 
#' @param f Input file
#' @param g Input file
#' @export
thresholded_detection_file_summary <- function(f, g) {
  detections <- readr::read_tsv(f)
  phenotypes <- readr::read_tsv(g)

  print(head(detections))
  print(head(phenotypes))

  detections <- dplyr::left_join(detections, phenotypes, by = "cell_id") %>%
    dplyr::mutate(Class = nucleus_phenotype)

  print(head(detections))

  summary <- detection_file_summary(detections, type = "table")

  print(head(summary))
  
  return(summary)
}

binned_detection_file_summary <- function(x, common_cols, grouping_cols, type = "file") {
  if (type == "file") {
    detections <- vroom::vroom(x)
  } else {
    detections <- x
  }
  
  # common_cols <- c("cell_id", "Object ID", "objectType", "index", "pn_index", "Nearest Primary Nucleus ID", "Proximity to nearest Primary Nucleus")
  # grouping_cols <- c("Image", "Name", "Class", "Parent ID", "Parent", "ROI ID", "ROI")

  # Identify available columns
  available_common_cols <- intersect(common_cols, colnames(detections))
  available_grouping_cols <- intersect(grouping_cols, colnames(detections))

  # Inform about missing columns (optional)
  missing_common_cols <- setdiff(common_cols, colnames(detections))
  missing_grouping_cols <- setdiff(grouping_cols, colnames(detections))

  if (length(missing_common_cols) > 0) {
    message("The following common columns are missing and will be ignored: ", paste(missing_common_cols, collapse = ", "))
  }
  if (length(missing_grouping_cols) > 0) {
    message("The following grouping columns are missing and will be ignored: ", paste(missing_grouping_cols, collapse = ", "))
  }

  print(head(detections))

  breaks <- c(seq(0, 5.0, 0.1), seq(6.0, 50.0, 1.0))

  counts <- detections %>%
    dplyr::select(-c(all_of(available_common_cols), contains("phenotype"), ends_with("state"))) %>%
    # Perform binning with custom breaks
    mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=breaks)) %>%
    # mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40))) %>%
    # mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=c(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20))) %>%
    group_by(across(all_of(c(available_grouping_cols, "area_bin")))) %>%
    dplyr::summarize(count = n()) %>%
    ungroup() %>%
    group_by(Image) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    ungroup()

  print(head(counts))
  
  return(counts)
}

thresholded_binned_detection_file_summary <- function(f, g) {
  detections <- readr::read_tsv(f)
  phenotypes <- readr::read_tsv(g)

  print(head(detections))
  print(head(phenotypes))

  detections <- dplyr::left_join(detections, phenotypes, by = "cell_id")

  print(head(detections))

  binned_counts <- detections %>%
    dplyr::select(-c("cell_id", contains("phenotype"), ends_with("state"))) %>%
    # Perform binning with custom breaks
    # mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40))) %>%
    mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40))) %>%
    # mutate(area_bin = cut(`Nucleus: Area µm^2`, breaks=c(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20))) %>%
    group_by(Image, Name, Class, Parent, ROI, area_bin) %>%
    dplyr::summarize(count = n()) %>%
    ungroup() %>%
    group_by(Image) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    ungroup()

  df <- binned_counts %>%
    as_tibble()
  
  return(df)
}


#' Summarize detection files (in serial, not recommended for > 10 files)
#' 
#' @param files
#' @export
summarize_detection_files <- function(files, db_path) {
  summary_table <- lapply(files, function(f) {
    stats <- detection_file_summary(f)
    voa <- parse_sample_ids(f)
    data.frame(voa=voa, stats)
  }) %>% rbind.fill
  summary_table$condensed_id <- ithi.meta::map_id(summary_table$voa, from = "voa", to = "condensed_id", db_path)
  return(summary_table)
}


#' Map channels
#' 
#' @param detections Detections
#' @export
map_channels <- function(detections) {
  
  # Map channel names
  markers <- any(str_detect(detections$Image, "cGAS_ENPP1_DAPI"))
  print(markers)

  if (markers) {
    detections <- detections %>%
      # # Recode values
      # dplyr::mutate(Name = str_replace(Name, "(Red|Channel 1)", "cGAS")) %>%
      # Rename columns
      dplyr::rename_with(~ str_replace(.x, "(Red|Channel 1)", "cGAS")) %>%
      dplyr::rename_with(~ str_replace(.x, "(Green|Channel 2)", "ENPP1")) %>%
      dplyr::rename_with(~ str_replace(.x, "(Blue|Channel 3)", "DAPI"))
  }

  markers <- any(str_detect(detections$Image, "CD8_STING_DAPI"))
  print(markers)

  if (markers) {
    detections <- detections %>%
      # Rename columns
      dplyr::rename_with(~ str_replace(.x, "(Red|Channel 1)", "CD8")) %>%
      dplyr::rename_with(~ str_replace(.x, "(Green|Channel 2)", "STING")) %>%
      dplyr::rename_with(~ str_replace(.x, "(Blue|Channel 3)", "DAPI"))
  }

  markers <- any(str_detect(detections$Image, "cGAS_STING_p53_panCK_CD8_DAPI_R1"))
  print(markers)

  if (markers) {
    detections <- detections %>%
      # Rename columns
      dplyr::rename_with(~ str_replace(.x, "DAPI", "DAPI")) %>%
      dplyr::rename_with(~ str_replace(.x, "Alexa488", "CD8")) %>%
      dplyr::rename_with(~ str_replace(.x, "Alexa546", "panCK")) %>%
      dplyr::rename_with(~ str_replace(.x, "Alexa594", "cGAS")) %>%
      dplyr::rename_with(~ str_replace(.x, "Alexa647", "p53")) %>%
      dplyr::rename_with(~ str_replace(.x, "CFP", "STING"))
  }

  return(detections)
}
