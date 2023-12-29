from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from icecream import ic

class DummyDataLoader:
    dataLoaded = pyqtSignal(object)

    def __init__(self, base_path,):
        self.base_path = Path(base_path)
        self.status = {}
        self.data = {}
        self.npy = None
        self.pickle_files = None
        self.npy_files = None
        self.dirs = []
        self.edirs = {}
        self.run()

    def run(self):
        self.pickle_files = list(self.base_path.glob('*.p'))
        self.npy_files = list(self.base_path.glob('*.npy'))
        self.dirs = [folder for folder in self.base_path.iterdir() if folder.is_dir()]

        if 'spike_detection' in self.dirs:
            self.status['spike_detection'] = True

        if 'spike_clustering' in self.dirs:
            self.status['spike_clustering'] = True

        if 'spike_sorting' in self.dirs:
            self.status['spike_sorting'] = True

        if len(list(self.pickle_files)) == 1:
            self.status['pk_file'] = True

        if len(list(self.npy_files)) >= 1:
            self.npy = list(self.npy_files)
            self.status['npy_file'] = True

        for edir in self.base_path.rglob("electrode*"):
            ic()
            if edir.is_file():
                self.edirs[edir.parent] = edir
                data_dir = edir / 'data'
                plots_dir = edir / 'plot'
            if edir.is_dir():
                self.edirs[edir.parent] = edir
                data_dir = edir / 'data'
                plots_dir = edir / 'plot'
            else:
                continue

            if data_dir.exists():
                self.status['data'] = True
            if plots_dir.exists():
                self.status['plots'] = True

        for electrode_dir in self.base_path.iterdir():
            if electrode_dir.is_dir():
                electrode_data = {}
                # Analysis params
                analysis_params_dir = electrode_dir / 'analysis_params'
                if analysis_params_dir.exists():
                    for param_file in analysis_params_dir.glob('*.json'):
                        electrode_data['analysis_params'] = {param_file.stem: str(param_file)}

                # Data
                data_dir = electrode_dir / 'data'
                if data_dir.exists():
                    data_files = {}
                    for data_file in data_dir.glob('*'):
                        data_files[data_file.stem] = str(data_file)
                    electrode_data['data'] = data_files

                # Plots
                plots_dir = electrode_dir / 'plots'
                if plots_dir.exists():
                    plot_files = {}
                    for plot_file in plots_dir.glob('*'):
                        plot_files[plot_file.stem] = str(plot_file)
                    electrode_data['plots'] = plot_files

if __name__ == "__main__":
    path = Path().home() / 'data' / 'r35'
    loader = DummyDataLoader(path)
    x = 5