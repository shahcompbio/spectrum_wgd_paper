region_file_summary <- function(x) {

    regions <- readr::read_tsv(x)

    # regions <- regions %>%
    #     # Fix image ID
    #     dplyr::mutate(Image = str_replace_all(Image, ".tif", "")) %>%
    #     # Remove unclassified tissue regions
    #     filter(Parent != "Image")
    
    summary_regions <- regions %>%
        group_by(Image, Name, Class, Parent) %>%
        summarize_at(c("Area µm^2", "Perimeter µm"), sum) %>%
        ungroup

  return(summary_regions)
}