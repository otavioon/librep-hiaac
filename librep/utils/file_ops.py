import hashlib
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from librep.config.type_definitions import PathLike


class ChecksumError(Exception):
    pass


def download_url(url: str, file_path: PathLike):
    """Download an URL to a path

    Args:
        url (str): URL to perform download
        file_path (Path): Path to store downloaded data

    Returns:
        None
    """
    file_path = Path(file_path)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with file_path.open('wb') as file, \
        tqdm(
            desc=f"Downloading to {str(file_path)}",
            total=total, unit='iB', unit_scale=True,
            unit_divisor=1024,) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def unzip_file(file_path: PathLike, destination_dir: PathLike):
    file_path, destination_dir = Path(file_path), Path(destination_dir)
    with zipfile.ZipFile(file_path, 'r') as zf:
        for member in tqdm(
                zf.infolist(),
                desc=
                f"Extracting '{str(file_path)}' to directory '{str(destination_dir)}'..."
        ):
            zf.extract(member, destination_dir)


def file_checksum(file_path: PathLike) -> str:
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_unzip_check(url: str,
                         download_destination: PathLike,
                         checksum: str = None,
                         remove_on_check_error: bool = True,
                         unzip_dir: Path = None,
                         remove_downloads: bool = True):
    download_destination = Path(download_destination)
    download_url(url, download_destination)

    if checksum is not None:
        digest = file_checksum(download_destination)
        if digest != checksum:
            if remove_on_check_error:
                download_destination.unlink()
            raise ChecksumError

    if unzip_dir is not None:
        unzip_file(download_destination, unzip_dir)

    if remove_downloads:
        download_destination.unlink()


def json_dump(file_path: PathLike,
              data: dict,
              indent: int = 4,
              sort_keys: bool = True,
              **json_kwargs):
    file_path = Path(file_path)
    with file_path.open("w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, **json_kwargs)