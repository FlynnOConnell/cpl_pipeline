from pathlib import Path

from spk2extract.spk_io.spk_h5 import read_h5


def merge_h5(path_pre, path_post):
    pre_h5 = read_h5(path_pre)
    pre_data = pre_h5["data"]
    pre_events = pre_h5["event"]
    pre_metadata = pre_h5["metadata_channel"]
    pre_filemetadata = pre_h5["metadata_file"]
    pre_time = pre_metadata["LFP1_OB"]["channel_max_time"]
    recording_length_pre = pre_filemetadata["recording_length"]
    post_h5 = read_h5(path_post)
    post_data = post_h5["data"]
    post_events = post_h5["event"]
    post_metadata = post_h5["metadata_channel"]
    post_filemetadata = post_h5["metadata_file"]
    post_time = post_metadata["LFP1_OB"]["channel_max_time"]
    recording_length_post = post_filemetadata["recording_length"]

    x = pre_h5["data"]["LFP1_OB"]["LFP1_OB"][0]


if __name__ == "__main__":
    # get all files with "pre" in the name before .h5
    path = Path().home() / "spk2extract" / "h5"
    pre_files = list(path.glob("*pre*.h5"))
    post_files = list(path.glob("*post*.h5"))
    merge_h5(pre_files[0], post_files[0])