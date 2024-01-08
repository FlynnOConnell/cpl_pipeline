import argparse
from pathlib import Path

import cpl_pipeline as cpl

def main():

    # data = load_dataset()

    # data = cpl.Dataset(Path().home() / 'data' / 'r35' / 'session_1')
    data = cpl.load_dataset(Path().home() / 'data' / 'r35' / 'session_1')
    # data.initialize_parameters()

    data.extract_data()
    data.detect_spikes()

    data.cluster_spikes()
    data.sort_spikes()
    data.units_similarity()

    data.post_sorting()
    data.make_unit_plots()

if __name__ == "__main__":
    main()
