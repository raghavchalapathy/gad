"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'cats_n_dogs': 30,
    'cats_n_dogs': 30
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'cats_n_dogs': '.jpg',
    'cats_n_dogs': '.jpg'
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'cats_n_dogs': '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/input/cats_Vs_Dogs/cats_Vs_Dogs_train.csv',
    'cats_n_dogs_test': '/Users/raghav/Documents/Uni/group_anomalies_detection/groupAnomalies/models/tf-cyclegan/input/cats_Vs_Dogs/cats_Vs_Dogs_test.csv',
}
