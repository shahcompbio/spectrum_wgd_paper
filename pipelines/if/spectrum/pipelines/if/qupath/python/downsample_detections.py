import click
import sys
import pandas as pd
import logging

@click.command()
@click.option('--object_detections_tsv')
@click.option('--downsampled_object_detections_tsv')
def main(object_detections_tsv, downsampled_object_detections_tsv):

    # Load dataframe files
    def load_dataframe(file_path):
        try:
            return pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    print("Object detections loaded")
    object_detections = load_dataframe(object_detections_tsv)

    # Downsample each group to a maximum of 1E4 random rows or fewer if group size is smaller
    def downsample_detections(df, n=10000, random_state=42):
        downsampled = df.groupby('Class').apply(lambda x: x.sample(n=min(n, len(x)), random_state=random_state)).reset_index(drop=True)
        return downsampled
    
    downsampled_object_detections = downsample_detections(object_detections)

    # Export to TSV
    def export_to_tsv(dataframe, file_path):
        try:
            dataframe.to_csv(file_path, sep="\t", index=False)
            print(f"TSV exported to: {file_path}")
        except Exception as e:
            print(f"Error exporting TSV: {e}")

    export_to_tsv(downsampled_object_detections, downsampled_object_detections_tsv)

    print("File loaded, objects downsampled and exported successfully.")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
