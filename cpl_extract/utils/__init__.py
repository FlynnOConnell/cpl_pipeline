from . import math_tools, particles, tk_widgets, spike_sorting_GUI, data_reset

def pad_arrays_to_same_length(arr_list, max_diff=100):
    """
    Pads numpy arrays to the same length.

    Parameters:
    - arr_list (list of np.array): The list of arrays to pad
    - max_diff (int): Maximum allowed difference in lengths

    Returns:
    - list of np.array: List of padded arrays
    """
    lengths = [len(arr) for arr in arr_list]
    max_length = max(lengths)
    min_length = min(lengths)

    if max_length - min_length > max_diff:
        raise ValueError("Arrays differ by more than the allowed maximum difference")

    padded_list = []
    for arr in arr_list:
        pad_length = max_length - len(arr)
        padded_arr = np.pad(arr, (0, pad_length), "constant", constant_values=0)
        padded_list.append(padded_arr)

    return padded_list

def extract_common_key(filepath):
    parts = filepath.stem.split("_")
    return "_".join(parts[:-1])

def check_substring_content(main_string, substring) -> bool:
    """Checks if any combination of the substring is in the main string."""
    return substring.lower() in main_string.lower()

__all__ = [
    "math_tools",
    "particles",
    "tk_widgets",
    "spike_sorting_GUI",
    "data_reset",
    "pad_arrays_to_same_length",
    "extract_common_key",
    "check_substring_content",
]