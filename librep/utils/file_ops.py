import hashlib
import json
import gdown
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
    total = int(resp.headers.get("content-length", 0))
    with file_path.open("wb") as file, tqdm(
            desc=f"Downloading to {str(file_path)}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def unzip_file(file_path: PathLike, destination_dir: PathLike):
    file_path, destination_dir = Path(file_path), Path(destination_dir)
    with zipfile.ZipFile(file_path, "r") as zf:
        for member in tqdm(
                zf.infolist(),
                desc=
                f"Extracting '{str(file_path)}' to directory '{str(destination_dir)}'...",
        ):
            zf.extract(member, destination_dir)


def file_checksum(file_path: PathLike) -> str:
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_unzip_check(
    url: str,
    download_destination: PathLike,
    checksum: str = None,
    remove_on_check_error: bool = True,
    unzip_dir: Path = None,
    remove_downloads: bool = True,
):
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


class Downloader:

    def __init__(self, url: str):
        self.url = url

    def download(self, destination: PathLike) -> Path:
        """Download an URL to a destination path

        Args:
            destination (PathLike): Path to store downloaded data

        Returns:
            None
        """
        raise NotImplementedError


class WgetDownload(Downloader):

    def download(self, destination: PathLike) -> Path:
        """Download an URL to a path

        Args:
            url (str): URL to perform download
            file_path (Path): Path to store downloaded data

        Returns:
            None
        """
        destination = Path(destination)
        resp = requests.get(self.url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with destination.open("wb") as file, tqdm(
                desc=f"Downloading to {str(destination)}",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        return destination


class GoogleDriveDownloader(Downloader):

    def download(self, destination: PathLike) -> Path:
        destination = Path(destination)
        gdown.download(id=self.url,
                       output=str(destination),
                       use_cookies=True,
                       quiet=False)
        return destination


class Extractor:
    def __init__(self, destination_dir: PathLike):
        self.destination_dir = Path(destination_dir)

    def extract(self, compressed_file_path: PathLike) -> Path:
        raise NotImplementedError


class ZipExtractor(Extractor):
    def extract(self, compressed_file_path: PathLike) -> PathLike:
        compressed_file_path = Path(compressed_file_path)
        with zipfile.ZipFile(compressed_file_path, "r") as zf:
            for member in tqdm(
                    zf.infolist(),
                    desc=f"Extracting '{str(compressed_file_path)}' to directory '{str(self.destination_dir)}'...",
            ):
                zf.extract(member, self.destination_dir)
        return self.destination_dir


class Checksum:
    def __init__(self, hash_val: str):
        self.hash_val = hash_val

    def check(self, file_path: PathLike) -> bool:
        raise NotImplementedError


class MD5Checksum(Checksum):
    def check(self, file_path: PathLike) -> bool:
        file_path = Path(file_path)
        with file_path.open("rb") as f:
            hashlib.md5(f.read()).hexdigest() == self.hash_val


def download_extract_check(
    destination_download_file: PathLike,
    downloader: Downloader,
    extractor: Extractor,
    checker: Checksum = None,
    remove_on_check_error: bool = True,
    unzip_dir: Path = None,
    remove_downloads: bool = True
):
    destination = Path(destination_download_file)
    downloaded_file = downloader.download(destination)

    if checker is not None:
        if not checker.check(downloaded_file):
            if remove_on_check_error:
                downloaded_file.unlink()
            raise ChecksumError

    if extractor is not None:
        extractor.extract(downloaded_file)

    if remove_downloads:
        downloaded_file.unlink()


def json_dump(
    file_path: PathLike,
    data: dict,
    indent: int = 4,
    sort_keys: bool = True,
    **json_kwargs,
):
    file_path = Path(file_path)
    with file_path.open("w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, **json_kwargs)
