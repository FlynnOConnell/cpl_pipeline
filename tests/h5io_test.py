import unittest
import tables
import pandas as pd
from pathlib import Path
import os
from cpl_extract.spk_io import create_empty_data_h5, create_hdf_arrays, merge_h5_files

class TestHDF5Operations(unittest.TestCase):

    def setUp(self):
        # Setup for tests, like creating mock data
        self.test_file = "test_file.h5"
        self.test_file_2 = "test_file_2.h5"
        self.merged_file = "merged_file.h5"
        self.rec_info = {"dig_in": [1, 2], "dig_out": [3, 4]}
        self.electrode_mapping = pd.DataFrame({'electrode': [3, 4], 'sampling_rate': [20000, 20000], 'units': ['V', 'V']})

    def test_create_empty_h5(self):
        create_empty_data_h5(self.test_file)
        self.assertTrue(Path(self.test_file).exists())

    def test_create_hdf_arrays(self):
        # Test creating arrays in an HDF5 file
        create_hdf_arrays(self.test_file, self.rec_info, self.electrode_mapping)
        with tables.open_file(self.test_file, 'r') as hf5:
            self.assertTrue("/raw" in hf5.root)
            self.assertTrue("/time" in hf5.root)

    def test_merge_h5_files(self):
        # Test merging two HDF5 files
        create_hdf_arrays(self.test_file, self.rec_info, self.electrode_mapping)
        create_hdf_arrays(self.test_file_2, self.rec_info, self.electrode_mapping)
        merge_h5_files([self.test_file, self.test_file_2])
        self.assertTrue(Path(self.merged_file).exists())
        # Additional checks for merged content

    def tearDown(self):
        # Cleanup after tests
        for file in [self.test_file, self.test_file_2, self.merged_file]:
            if Path(file).exists():
                os.remove(file)

if __name__ == '__main__':
    unittest.main()