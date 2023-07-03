import glob
import re


def get_markdown_files(dir, exclude_dirs_patterns: list = None):
    """Recursively get the paths to all markdown files in a directory, excluding any directories in exclude_dirs.
    Directories in exclude_dirs are matched using glob syntax and may appear anywhere in the path.
    """
    if exclude_dirs_patterns is None:
        exclude_dirs_patterns = []
    paths = []
    for path in glob.glob(f"{dir}/**", recursive=True):
        if not path.endswith(".md"):
            continue
        # Skip any paths that match any of the exclude_dirs patterns anywhere in the path
        if any(re.search(pattern, path) for pattern in exclude_dirs_patterns):
            continue
        paths.append(path)
    return paths


def get_file_text(file: str) -> str:
    """Return the text of the given file."""
    with open(file, "r") as f:
        return f.read()
