from tqdm import tqdm

class ProgressBarManager:
    """Singleton class for managing progress bars."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressBarManager, cls).__new__(cls)
            cls._instance.pbar_files = None
            cls._instance.pbar_channels = None
            cls._instance.pbar_clusters = None
        return cls._instance

    def init_file_bar(self, total):
        self.pbar_files = tqdm(total=total, desc="Files", leave=True)

    def update_file_bar(self):
        self.pbar_files.update(1)

    def close_file_bar(self):
        self.pbar_files.close()

    def init_channel_bar(self, total):
        self.pbar_channels = tqdm(total=total, desc="Channels", leave=True)

    def update_channel_bar(self):
        self.pbar_channels.update(1)

    def close_channel_bar(self):
        self.pbar_channels.close()

    def init_cluster_bar(self, total):
        self.pbar_clusters = tqdm(total=total, desc="Clusters", leave=True)

    def update_cluster_bar(self):
        self.pbar_clusters.update(1)

    def close_cluster_bar(self):
        self.pbar_clusters.close()