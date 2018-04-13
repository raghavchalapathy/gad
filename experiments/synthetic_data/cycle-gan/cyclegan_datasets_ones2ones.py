"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'one2one_mapping': 88,
    'one2one_mapping': 50
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'one2one_mapping': '.png',
    'one2one_mapping': '.png',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'one2one_mapping': '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/input/one2one_mapping/one2one_mapping_train.csv',
    'one2one_mapping_test': '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/input/one2one_mapping/one2one_mapping_test.csv',
}
