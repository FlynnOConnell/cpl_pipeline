import argparse
from pathlib import Path

from icecream import ic

from cpl_pipeline import Dataset, load_dataset, load_pickled_object

def main():

    # data = load_dataset()

    #data = Dataset(Path().home() / 'data' / 'r35_session_1')
    data = load_dataset(Path().home() / 'data' / 'r35_session_1')
    data.initialize_parameters()

    data.extract_data()
    data.detect_spikes()

    data.cluster_spikes()
    data.sort_spikes()
    data.units_similarity()

    data.post_sorting()
    data.make_unit_plots()

if __name__ == "__main__":
    main()
