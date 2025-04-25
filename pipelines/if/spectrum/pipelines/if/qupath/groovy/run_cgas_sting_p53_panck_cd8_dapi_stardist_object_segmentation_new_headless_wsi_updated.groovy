//java -jar qupath-0.5.1.jar script --image /path_to_image/image.ome.tiff /path_to_script/run_analysis.groovy

def numThreads = 16

import qupath.lib.common.ThreadTools
ThreadTools.setParallelism(numThreads)

println "Number of threads set to: ${numThreads}"

import qupath.ext.stardist.StarDist2D

import qupath.lib.scripting.QP
import qupath.lib.gui.scripting.QPEx
import qupath.lib.io.GsonTools

import qupath.lib.gui.images.servers.RenderedImageServer
import qupath.lib.regions.RegionRequest

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get image data
println 'Get image data'
var imageData = getCurrentImageData()
var imageNameWithExt = imageData.getServer().getMetadata().getName()
def imageName = imageNameWithExt.take(imageNameWithExt.indexOf('.'))

// Set image type
setImageType('FLUORESCENCE');

println QP.getCurrentImageData()

// String roi_annotation_geojson_path = '/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/qupath/roi-annotation/annotations/' + imageName + '.geojson'
// def json = new File(roi_annotation_geojson_path)
// def roi_annotations = PathIO.readObjects(json)

// addObjects(roi_annotations);
// resolveHierarchy()

// Rename channels
setChannelNames(
  'DAPI', // DAPI (C1): DAPI
  'CD8', // Alexa488 (C2): CD8
  'panCK', // Alexa546 (C3): panCK
  'cGAS', // Alexa594 (C4): cGAS
  'p53', // Alexa647 (C5): p53
  'STING', // CFP (C6): STING
)

// Create parent annotation
createFullImageAnnotation(true)
selectAnnotations()

// Run object detection for primary nuclei using DAPI (channel 1)
println 'Run object detection for primary nuclei using DAPI (channel 1)'

// Specify the model file
def primary_nuclei_path_model = "/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/qupath/models/dsb2018_heavy_augment.pb"

var primary_nuclei_stardist = StarDist2D.builder(primary_nuclei_path_model)
  .nThreads(numThreads)        // Temporarix fix (see https://forum.image.sc/t/stardist-error-unable-to-load-bundle-null/45428/3)
  .threshold(0.5)              // Probability (detection) threshold
  .channels('DAPI')            // Select detection channel
  .normalizePercentiles(1, 99) // Percentile normalization
  .pixelSize(0.1625)           // Resolution for detection
  .cellExpansion(3.0)          // Approximate cells based upon nucleus expansion
  .cellConstrainScale(1.5)     // Constrain cell expansion using nucleus size
  .measureShape()              // Add shape measurements
  .measureIntensity()          // Add cell measurements (in all compartments)
  .includeProbability(true)    // Add probability as a measurement (enables later filtering)
  .classify("Primary nucleus") // Automatically assign all created objects as 'Primary nucleus'
  .build()

// selectObjectsByClassification('Tumor','Stroma','Vasculature','Glass')
var primary_nuclei_path_objects = getSelectedObjects()
if (primary_nuclei_path_objects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
primary_nuclei_stardist.detectObjects(imageData, primary_nuclei_path_objects)

def primary_nuclei_detections = getDetectionObjects()
primary_nuclei_detections.each{
  parent_id = it.getParent().getID().toString()
  it.setName(parent_id)

  def ml = it.getMeasurementList()
  def roi = it.getROI()
  // Get centroid in pixel coordinates
  ml.putMeasurement('Centroid X', roi.getCentroidX())
  ml.putMeasurement('Centroid Y', roi.getCentroidY())
  // // Get centroid in micron coordinates
  // def cal = getCurrentServer().getPixelCalibration()
  // double pixel_width = cal.getPixelWidthMicrons()
  // double pixel_height = cal.getPixelHeightMicrons()
  // ml.putMeasurement('Centroid X µm', roi.getCentroidX() * pixel_width)
  // ml.putMeasurement('Centroid Y µm', roi.getCentroidY() * pixel_height)
  ml.close()
}
fireHierarchyUpdate()

// Save detection objects for primary nuclei
String object_detection_results_primary_nuclei_tsv_path = 'object_detection_results_primary_nuclei.tsv'
saveDetectionMeasurements(object_detection_results_primary_nuclei_tsv_path)
String object_detection_results_primary_nuclei_geojson_path = 'object_detection_results_primary_nuclei.geojson'
exportObjectsToGeoJson(primary_nuclei_detections, object_detection_results_primary_nuclei_geojson_path, "FEATURE_COLLECTION")

// Run object detection for micronuclei using cGAS (Alexa594, channel 4)
println 'Run object detection for micronuclei using cGAS (Alexa594, channel 4)'

// Specify the model file
// def micronuclei_path_model = "/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/qupath/models/20211202_2D_StarDist_cGAS_min10_max99-95.pb"
// def micronuclei_path_model = "/data1/shahs3/users/vazquezi/projects/spectrum/pipelines/if/qupath/models/20231003_2D_StarDist_cGAS_min50_max99-95.pb"
// def micronuclei_path_model = "/data1/shahs3/users/vazquezi/projects/stardist/models/spectrum/20231003_2D_StarDist_cGAS_min50_max99-95.pb"
def micronuclei_path_model = "/data1/shahs3/users/vazquezi/projects/stardist/models/spectrum/20231003_2D_StarDist_cGAS_min50_max99-95.pb"

var micronuclei_stardist = StarDist2D.builder(micronuclei_path_model)
  .nThreads(numThreads)            // Temporarix fix (see https://forum.image.sc/t/stardist-error-unable-to-load-bundle-null/45428/3)
  .threshold(0.5)                  // Probability (detection) threshold
  .channels('cGAS')                // Select detection channel
  // .normalizePercentiles(10, 99.95) // Percentile normalization
  .preprocess(
    StarDist2D.imageNormalizationBuilder()
    .maxDimension(4096)   
    .percentiles(50, 99.95) 
    .build()
  )
  .pixelSize(0.1625)               // Resolution for detection
  .cellExpansion(1.0)              // Approximate cells based upon nucleus expansion
  .cellConstrainScale(1.0)         // Constrain cell expansion using nucleus size
  .measureShape()                  // Add shape measurements
  .measureIntensity()              // Add cell measurements (in all compartments)
  .includeProbability(true)        // Add probability as a measurement (enables later filtering)
  .classify("Micronucleus")        // Automatically assign all created objects as 'Micronucleus'
  .build()

// selectObjectsByClassification('Tumor','Stroma','Vasculature','Glass')
var micronuclei_path_objects = getSelectedObjects()

if (micronuclei_path_objects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
micronuclei_stardist.detectObjects(imageData, micronuclei_path_objects)

def micronuclei_detections = getDetectionObjects()
micronuclei_detections.each{
  parent_id = it.getParent().getID().toString()
  it.setName(parent_id)

  def ml = it.getMeasurementList()
  def roi = it.getROI()
  // Get centroid in pixel coordinates
  ml.putMeasurement('Centroid X', roi.getCentroidX())
  ml.putMeasurement('Centroid Y', roi.getCentroidY())
  // // Get centroid in micron coordinates
  // def cal = getCurrentServer().getPixelCalibration()
  // double pixel_width = cal.getPixelWidthMicrons()
  // double pixel_height = cal.getPixelHeightMicrons()
  // ml.putMeasurement('Centroid X µm', roi.getCentroidX() * pixel_width)
  // ml.putMeasurement('Centroid Y µm', roi.getCentroidY() * pixel_height)
  ml.close()
}
fireHierarchyUpdate()

// Save detection objects for micronuclei
String object_detection_results_micronuclei_tsv_path = 'object_detection_results_micronuclei.tsv'
saveDetectionMeasurements(object_detection_results_micronuclei_tsv_path)
String object_detection_results_micronuclei_geojson_path = 'object_detection_results_micronuclei.geojson'
exportObjectsToGeoJson(micronuclei_detections, object_detection_results_micronuclei_geojson_path, "FEATURE_COLLECTION")

// Combine overlapping primary nuclei and micronuclei
addObjects(primary_nuclei_detections)
resolveHierarchy()

def combined_detections = getDetectionObjects()

// Save combined detection objects
String object_detection_results_tsv_path = 'object_detection_results.tsv'
saveDetectionMeasurements(object_detection_results_tsv_path)
String object_detection_results_geojson_path = 'object_detection_results.geojson'
exportObjectsToGeoJson(combined_detections, object_detection_results_geojson_path, "FEATURE_COLLECTION")
