import click
import sys
import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import shape
import logging
import pyogrio
import tqdm
import geojson
import csv
import ijson.backends.python as ijson

pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": 0})


def stream_geojson(filename, batch_size):
    with open(filename, 'r') as fp:
        features = ijson.items(fp, 'features.item', use_float=True)
        try:
            while True:
                data = []
                for _ in range(batch_size):
                    data.append(next(features))
                yield geojson.FeatureCollection(data)
        except StopIteration:
            if data:
                yield geojson.FeatureCollection(data)
            return


def read_geojson_chunks(filename, batch_size=10000):
    for chunk in stream_geojson(filename, batch_size):
        for feature in chunk['features']:
            feature['properties']['id'] = feature['id']
        chunk = gpd.GeoDataFrame.from_features(chunk['features'])
        yield chunk


from contextlib import contextmanager

class GeoJSONWriter:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.file.write('{"type": "FeatureCollection", "features": [')
        self.first_chunk = True

    def close(self):
        self.file.write(']}')
        self.file.close()

    def write_features(self, geopandas_df):
        # Convert features list to JSON string, omitting the outer brackets
        chunk_json = json.dumps(geopandas_df.to_geo_dict()['features'])[1:-1]

        # Handle comma separation between chunks
        if not self.first_chunk:
            chunk_json = ',' + chunk_json
        else:
            self.first_chunk = False

        # Write to file
        self.file.write(chunk_json)


@click.command()
@click.option('--object_detections_geojson')
@click.option('--region_annotations_geojson')
@click.option('--roi_annotations_geojson')
@click.option('--annotated_object_detections_geojson')
@click.option('--annotated_object_detections_tsv')
def main(object_detections_geojson, region_annotations_geojson, roi_annotations_geojson, annotated_object_detections_geojson, annotated_object_detections_tsv):

    print("Loading region annotations")
    region_annotations = gpd.read_file(region_annotations_geojson)

    print("Loading ROI annotations")
    roi_annotations = gpd.read_file(roi_annotations_geojson)

    geojson_writer = GeoJSONWriter(annotated_object_detections_geojson)

    print("Iterating through object detections")
    # Initialize header tracking
    expected_headers = None
    
    for idx, object_detections in enumerate(read_geojson_chunks(object_detections_geojson, batch_size=10000)):

        # Remove invalid detections
        object_detections = object_detections[object_detections.geometry.is_valid]

        # Assign parent annotation to each detection
        def assign_parents(detections, annotations, label = "Parent"):
            label_id = label + " ID"
            detections[label_id] = None
            detections[label] = None

            for index, detection in tqdm.tqdm(detections.iterrows()):
                for _, annotation in annotations.iterrows():
                    if annotation.geometry.contains(detection.geometry):
                        detections.at[index, label_id] = annotation["id"]  # Assumes annotations have an 'id' field
                        classification = json.loads(annotation["classification"])
                        detections.at[index, label] = classification["name"] # Assumes annotations have a 'classification' field
                        break
            return detections

        print("Assigning region annotations")
        annotated_object_detections = assign_parents(object_detections, region_annotations, label = "Parent")


        print("Assigning ROI annotations")
        annotated_object_detections = assign_parents(annotated_object_detections, roi_annotations, label = "ROI")

        # Export to GeoJSON
        print("Writing to geojson")
        geojson_writer.write_features(annotated_object_detections)

        # Export to TSV
        def create_dataframe(dataframe):
            # Step 1: Drop geometry and name if not needed
            dataframe.drop(columns=['geometry', 'name'], inplace=True)

            # Step 2: List of columns to check for nested structures
            nested_columns = ['classification', 'measurements']

            # Step 3: Parse and expand each nested column
            for col in nested_columns:
                if col in dataframe.columns:
                    # Parse the column as JSON if it's a string
                    dataframe[col] = dataframe[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                    
                    # Expand the column into separate columns
                    expanded_dataframe = dataframe[col].apply(pd.Series)
                    
                    # Combine the expanded DataFrame with the main DataFrame
                    dataframe = pd.concat([dataframe.drop(columns=[col]), expanded_dataframe], axis=1)

            # Step 4: Rename columns
            dataframe.rename(columns={'id': 'Object ID', 'name': 'Class'}, inplace=True)

            # Step 5: Drop color if not needed
            dataframe.drop(columns=['color'], inplace=True)

            return dataframe
        
        dataframe = create_dataframe(annotated_object_detections)

        # Writing to TSV
        if expected_headers is None:  # First batch
            expected_headers = list(dataframe.columns)  # Capture initial headers
            dataframe.to_csv(annotated_object_detections_tsv, sep='\t', index=False)
        else:
            current_headers = list(dataframe.columns)

            if expected_headers is None:  # Fallback check (shouldn't happen)
                logging.error("Expected headers were never initialized.")
                sys.exit(1)

            # Ensure all batches have the same columns
            if set(current_headers) != set(expected_headers):
                logging.warning("Inconsistent headers detected in batch {}. Adjusting headers...".format(idx))
                
                # Add missing columns with None values
                for col in expected_headers:
                    if col not in dataframe.columns:
                        dataframe[col] = None
                
                # Remove extra columns not in expected headers
                dataframe = dataframe[expected_headers]

            dataframe.to_csv(annotated_object_detections_tsv, mode='a', header=False, sep='\t', index=False)

        # break

    geojson_writer.close()

    print("Files processed, hierarchy resolved, and objects exported successfully.")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()