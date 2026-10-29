"""Load pipeline configuration from a YAML file."""

import yaml

__all__ = ["load_config"]


def load_config(path):
    """Load a pipeline configuration from a YAML file into a plain dict.

    The returned dict is suitable for passing straight to
    :func:`pyAethra.load_and_run` or :func:`pyAethra.run_pipeline_from_dataframe`.

    Keys omitted from the file fall back to the pipeline's built-in defaults
    at run time, with one exception: the required column keys ``time_col``,
    ``mag_col``, and ``err_col`` must be present (either in the file or supplied
    another way). YAML ``null`` maps to Python ``None`` (e.g. ``group_col: null``
    means "one object per file").

    Parameters
    ----------
    path : str
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed configuration. An empty file yields an empty dict.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path!r} must contain a YAML mapping (key: value pairs), "
            f"got {type(data).__name__}."
        )
    return data
