from pathlib import Path
import numpy as np

from spk2extract import spk_io


def get_common_name(path1: Path, path2: Path) -> str:
    common_name1 = path1.stem.split("_preinfusion")[0]
    common_name2 = path2.stem.split("_postinfusion")[0]

    if common_name1 == common_name2:
        return common_name1
    else:
        raise ValueError("Names do not match")


def check_substring_content(main_string, substring):
    """Checks if any combination of the substring is in the main string."""
    return substring.lower() in main_string.lower()


def merge_data(data_pre: dict, data_post: dict):
    combined = {}
    for key in data_pre.keys():
        if check_substring_content(key, "u") and not check_substring_content(
            key, "lfp"
        ):
            if key in data_post.keys():
                pre_times = data_pre[key]["times"]["times"]
                post_times = data_post[key]["times"]["times"]
                pre_spikes = data_pre[key]["spikes"]["spikes"]
                post_spikes = data_post[key]["spikes"]["spikes"]

                # Adjust the timestamps
                last_timestamp_pre = pre_times[-1]
                post_times += last_timestamp_pre
                pre_spikes = np.vstack((pre_spikes, post_spikes))
                pre_times = np.append(pre_times, post_times)
            combined[key] = data_pre[key]
        elif check_substring_content(key, "lfp"):
            if key in data_post.keys():

                pre_times = data_pre[key]["times"]["times"]
                post_times = data_post[key]["times"]["times"]
                pre_spikes = data_pre[key]["spikes"]["spikes"]
                post_spikes = data_post[key]["spikes"]["spikes"]

                # Adjust the timestamps
                last_timestamp_pre = pre_times[-1]
                post_times += last_timestamp_pre

                pre_times = np.append(pre_times, post_times)
                pre_spikes = np.append(pre_spikes, post_spikes)

                data_pre[key] = {"times": pre_times, "spikes": pre_spikes}
            combined[key] = data_pre[key]

    return combined


def merge_events(events_pre, events_post):
    # add the last timestamp of the 'pre' data to each timestamp in the 'post' data
    last_timestamp_pre = events_pre["events"][-1]
    events_post = events_post["events"]
    events_post += last_timestamp_pre
    events_pre = np.append(events_pre, events_post)
    return events_pre


def merge_metadata_file(metadata_file_pre, metadata_file_post):
    return {"pre": metadata_file_pre, "post": metadata_file_post}


def merge_h5(path_pre, path_post):
    pre_h5 = spk_io.read_h5(path_pre)
    post_h5 = spk_io.read_h5(path_post)
    merged_data = merge_data(pre_h5["spikedata"], post_h5["spikedata"])
    merged_events = merge_events(pre_h5["events"], post_h5["events"])
    merged_metadata = merge_metadata_file(
        pre_h5["metadata"]["channel"], post_h5["metadata"]["metadata"]
    )
    metadata_channel = pre_h5["metadata"]
    return merged_data, merged_events, merged_metadata, metadata_channel


if __name__ == "__main__":
    # get all files with "pre" in the name before .h5
    path = Path().home() / "spk2extract" / "h5"
    pre_files = list(path.glob("*pre*.h5"))
    post_files = list(path.glob("*post*.h5"))
    combined_metadata = {}

    data, events, m_file, m_channel = merge_h5(pre_files[0], post_files[0])
    savename = get_common_name(pre_files[0], post_files[0])
    savename = Path().home() / "spk2extract" / "combined" / f"{savename}.h5"
    savename.parent.mkdir(parents=True, exist_ok=True)
    spk_io.write_h5(
        savename,
        data=data,
        events=events,
        metadata_file=m_file,
        metadata_channel=m_channel,
    )

    x = 5
