library(data.table)
library(dplyr)
library(signals)
library(stringi)

get_tumor_cells <- function(qc, cols = c("tp53_BAF", "fracLOH_17", "fracLOH_13", "breakpoints", "frac_nondiploid", "pct_cell_ploidy")){
  
  message("Get columns for PCA")
  data <- as.data.frame(qc)[cols]
  for(i in 1:ncol(data)){
    data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
  }
  
  message("Run PCA")
  pc <- prcomp(data, center = TRUE, scale = TRUE)
  df_transform = as.data.frame(pc$x)
  
  message("Run k-means")
  kmeans_ = kmeans(df_transform, centers = 2, nstart = 50)
  qc$kmeans_cluster <- kmeans_$cluster
  
  message("Use k-means clusters to identify tumor cells")
  qc <- qc %>% 
    group_by(kmeans_cluster) %>% 
    mutate(x = mean(frac_nondiploid)) %>% 
    ungroup() %>% 
    mutate(is_tumor_cell = ifelse(x == max(x), "Y", "N")) %>% 
    select(-x)
  
  message("Revaluate tumor cell ID if majority of chr17 is LOH")
  x <- mean(qc %>% filter(is_tumor_cell == "Y") %>% .$fracLOH_17)
  if (x > 0.9){
    qc <- qc %>% 
      mutate(is_tumor_cell = ifelse(fracLOH_17 > 0.9, "Y", "N"))
  }
  
  message("Add PC cols to qc data frame")
  qc <- cbind(as.data.frame(qc), as.data.frame(pc$x)) %>% as.data.table() #conversions probably not needed
  return(qc)
}

#read in files
hscn <- fread(snakemake@input$hscn)
qc <- fread(snakemake@input$qc)

#get BAF of TP53
tp53loh <- hscn[start == 7500001 & chr == "17"] %>% 
  select(cell_id, BAF) %>% 
  rename(tp53_BAF = BAF)

qc <- left_join(qc, tp53loh, by = "cell_id")

qc <- get_tumor_cells(qc)
qc$patient_id <- snakemake@wildcards$patient
cols_to_keep <- c("patient_id",
                  "cell_id",
                  "tp53_BAF", 
                  "fracLOH_17", 
                  "fracLOH_13", 
                  "breakpoints", 
                  "frac_nondiploid", 
                  "pct_cell_ploidy", 
                  "is_tumor_cell",
                  "PC1", "PC2", "PC3", "PC4", "PC5", "PC6")
table_out <- qc %>% select(all_of(cols_to_keep))

#write table
fwrite(table_out, file = snakemake@output$assignment)


png(filename = snakemake@output$heatmap,width = 1100)
plotHeatmap(hscn,
            tree = NULL,
            reorderclusters = TRUE,
            plottree = FALSE,
            clusters = table_out %>% mutate(clone_id = is_tumor_cell))
dev.off()
