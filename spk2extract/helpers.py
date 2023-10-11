"""General helper functions used throughout the project."""

def check_substring_content(main_string, substring):
    """Checks if any combination of the substring is in the main string."""
    return substring.lower() in main_string.lower()
