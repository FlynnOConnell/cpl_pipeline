import os
import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_status(self, file_dir):
        status_path = os.path.join(file_dir, "status.npy")
        if os.path.exists(status_path):
            return np.load(status_path, allow_pickle=True).item()
        else:
            return {}

    def save_status(self, file_dir, status):
        status_path = os.path.join(file_dir, "status.npy")
        np.save(status_path, np.array([status], dtype=object))

    def process(self, file_num, overwrite=False):
        file_dir = os.path.join(self.base_path, f"file{file_num}")
        os.makedirs(file_dir, exist_ok=True)

        status = self.load_status(file_dir)

        for ch in range(1, 3):  # Loop through 2 channels
            ch_dir = os.path.join(file_dir, "Data", f"channel_{ch}")
            os.makedirs(ch_dir, exist_ok=True)
            for cluster in range(3, 5):  # Loop for 3 and 4 clusters
                cluster_dir = os.path.join(ch_dir, f"{cluster}_clusters")
                os.makedirs(cluster_dir, exist_ok=True)

                cluster_status_key = f"channel_{ch}_{cluster}_clusters"

                if not overwrite and status.get(cluster_status_key, False):
                    print(f"Skipping {cluster_dir}")
                    continue

                np_data = np.random.rand(10)  # Your actual data

                # Save your numpy data here
                np.save(os.path.join(cluster_dir, "data.npy"), np_data)

                # Generate and save plot
                plt.plot(np_data)
                plt.savefig(os.path.join(cluster_dir, "plot.png"))

                # Update status
                status[cluster_status_key] = True
                self.save_status(file_dir, status)


if __name__ == "__main__":
    base_path = "Base_path"
    dp = DataProcessor(base_path)
    dp.process(file_num=1, overwrite=False)
