"""Light interface to fsspec filesystem operations."""

import logging
import os
from pathlib import Path

from fsspec import get_mapper

logger = logging.getLogger(__name__)


def ls(path: str | Path, **kwargs) -> list:
    """List files in path.

    Parameters
    ----------
    path: str | Path
        Path to list directories from.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    content: list
        Names of files and directories in url.

    """
    logger.debug(f"Listing contents from {path}")
    fs = get_mapper(str(path), **kwargs).fs
    return fs.ls(path)


def exists(path: str | Path, **kwargs) -> bool:
    """Check if path exists.

    Parameters
    ----------
    path: str | Path
        Path to check.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    ans: bool
        True if path exists, False otherwise.

    """
    logger.debug(f"Checking if {path} exists")
    fs = get_mapper(str(path), **kwargs).fs
    return fs.exists(path)


def isdir(path: str | Path, **kwargs) -> bool:
    """Check if path is a directory.

    Parameters
    ----------
    path: str | Path
        Path to check.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    ans: bool
        True if path is a directory, False otherwise.

    """
    logger.debug(f"Checking if {path} is a directory")
    fs = get_mapper(str(path), **kwargs).fs
    return fs.isdir(path)


def isfile(path: str | Path, **kwargs) -> bool:
    """Check if path is a file.

    Parameters
    ----------
    path: str | Path
        Path to check.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    ans: bool
        True if path is file, False otherwise.

    """
    logger.debug(f"Checking if {path} is a file")
    fs = get_mapper(str(path), **kwargs).fs
    return fs.isfile(path)


def makedirs(path: str | Path, exist_ok: bool = False, **kwargs):
    """Recursively make directories.

    Parameters
    ----------
    path: str | Path
        Directory name.
    exist_ok: bool
        If False, will error if the target already exists.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    """
    logger.debug(f"Checking if {path} is a file")
    fs = get_mapper(str(path), **kwargs).fs
    fs.makedir(path, create_parents=exist_ok)


def rm(path: str | Path | list, recursive: bool = False, **kwargs):
    """Recursively make directories.

    Parameters
    ----------
    path: str | Path | list
        Path(s) to remove.
    recursive: bool
        Remove path recursively.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    """
    logger.debug(f"Removing path: {path}")
    fs = get_mapper(str(path), **kwargs).fs
    fs.rm(path, recursive=recursive)


def get(
    src: str | Path,
    dst: str | Path,
    recursive: bool = False,
    check: bool = False,
    **kwargs,
) -> str:
    """Download file or folder from src to dst.

    Parameters
    ----------
    src: str | Path
        Source path.
    dst: str | Path
        Full destination path or destination folder to download src to.
    recursive: bool
        Download  directories recursively.
    check: bool
        If True, check if file exists and if recursive is required before downloading.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    downloaded: str
        Full path of downloaded file, note this is different than dst if dst is a
        directory since in that case the src filename is prepended to dst.

    """
    if check and not isfile(src) and not isdir(src):
        raise FileNotFoundError(f"src {src} not found.")
    if check and isdir(src) and not recursive:
        raise IsADirectoryError(f"Set recursive=True to download directory {src}")
    if isdir(dst) and Path(src).name != Path(dst).name:
        dst = os.path.join(str(dst), Path(src).name)
    logger.debug(f"Downloading: {src} --> {dst}")
    fs = get_mapper(str(src), **kwargs).fs
    fs.get(src, dst, recursive=recursive)
    return dst


def put(
    src: str | Path,
    dst: str | Path,
    recursive: bool = False,
    check: bool = False,
    **kwargs,
):
    """Upload file or folder from src to dst.

    Parameters
    ----------
    src: str | Path
        Source path.
    dst: str | Path
        Full destination path or destination folder to download src to.
    recursive: bool
        Download  directories recursively.
    check: bool
        If True, check if file exists and if recursive is required before downloading.
    kwargs:
        Keyword arguments to pass to fsspec.get_mapper and instantiate an FSMap object.

    Returns
    -------
    downloaded: str
        Full path of downloaded file.

    """
    if check and not isfile(src) and not isdir(src):
        raise FileNotFoundError(f"src {src} not found.")
    if check and isdir(src) and not recursive:
        raise IsADirectoryError(f"Set recursive=True to upload directory {src}")
    if isdir(dst) and Path(src).name != Path(dst).name:
        dst = os.path.join(str(dst), Path(src).name)
    logger.debug(f"Uploading: {src} --> {dst}")
    fs = get_mapper(dst, **kwargs).fs
    if isdir(dst) and Path(src).name != Path(dst).name:
        dst = os.path.join(str(dst), Path(src).name)
    fs.put(src, dst, recursive=recursive)
