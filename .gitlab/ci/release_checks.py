"""Check the release distributions and installations used by GitLab CI."""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import zipfile
from email.parser import BytesParser
from importlib.metadata import version as installed_version
from importlib.util import find_spec
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


def expected_filenames(version: str) -> tuple[str, str]:
    return f"entise-{version}-py3-none-any.whl", f"entise-{version}.tar.gz"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check_metadata(version: str, dist: Path) -> None:
    wheel, sdist = expected_filenames(version)
    files = {path.name for path in dist.iterdir() if path.name != ".gitignore"}
    require(files == {wheel, sdist}, "Unexpected distribution filenames")

    name = f"entise-{version}"
    parser = BytesParser()
    with zipfile.ZipFile(dist / wheel) as archive:
        wheel_metadata = parser.parsebytes(archive.read(f"{name}.dist-info/METADATA"))
    with tarfile.open(dist / sdist) as archive:
        metadata_file = archive.extractfile(f"{name}/PKG-INFO")
        if metadata_file is None:
            raise SystemExit("Source distribution PKG-INFO is missing")
        with metadata_file:
            sdist_metadata = parser.parse(metadata_file)

    for metadata in (wheel_metadata, sdist_metadata):
        require(metadata["Name"] == "entise" and metadata["Version"] == version, "Metadata name/version mismatch")
        require(bool(metadata["Requires-Python"]), "Missing Requires-Python")
        require("pv-pvlib" in metadata.get_all("Provides-Extra", []), "Missing pv-pvlib extra")

    for field in ("Requires-Python", "Requires-Dist", "Provides-Extra"):
        require(
            sorted(wheel_metadata.get_all(field, [])) == sorted(sdist_metadata.get_all(field, [])),
            f"Metadata differs for {field}",
        )


def check_install(version: str, artifact: str) -> None:
    import entise
    from entise.core.registry import get_strategy, get_strategy_extra

    require(entise.__version__ == installed_version("entise") == version, "Installed EnTiSe version mismatch")
    require(Path(entise.__file__).resolve().is_relative_to(Path(sys.prefix)), "EnTiSe was imported outside its venv")
    require(get_strategy_extra("pvlib") == "pv-pvlib", "The PV strategy extra was not registered")
    if artifact.endswith("[pv-pvlib]"):
        require(get_strategy("pvlib").__name__ == "PVLib", "The installed PV strategy could not be loaded")
    else:
        require(find_spec("pvlib") is None, "The base distribution installed an optional dependency")
    print(f"Verified EnTiSe {version} from {artifact}")


def release_files(index: str, version: str) -> dict:
    url = f"{index}/pypi/entise/{version}/json"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=30) as response:
                return {item["filename"]: item for item in json.load(response)["urls"]}
        except HTTPError as error:
            if error.code != 404 or attempt == 5:
                raise
            time.sleep(10)
    raise RuntimeError("Release lookup did not return")


def download_release(version: str, manifest: Path, index: str, directory: Path) -> None:
    require(index in ("https://test.pypi.org", "https://pypi.org"), f"Unexpected index {index}")
    hashes = {filename: digest for digest, filename in map(str.split, manifest.read_text().splitlines())}
    require(set(hashes) == set(expected_filenames(version)), "Checksum manifest has unexpected files")
    files = release_files(index, version)
    require(set(files) == set(hashes), f"Files for EnTiSe {version} on {index} differ from the build")
    for filename, digest in hashes.items():
        item = files[filename]
        require(not item["yanked"] and item["digests"]["sha256"] == digest, f"Index checksum mismatch for {filename}")
        path = directory / filename
        with urlopen(item["url"], timeout=60) as response, path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        with path.open("rb") as downloaded:
            require(
                hashlib.file_digest(downloaded, "sha256").hexdigest() == digest, f"Download mismatch for {filename}"
            )
        print(f"Verified {filename} from {index} (SHA-256 {digest})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    metadata = commands.add_parser("metadata")
    metadata.add_argument("version")
    metadata.add_argument("dist", type=Path)
    installed = commands.add_parser("installed")
    installed.add_argument("version")
    installed.add_argument("artifact")
    download = commands.add_parser("download")
    download.add_argument("version")
    download.add_argument("manifest", type=Path)
    download.add_argument("index")
    download.add_argument("directory", type=Path)

    args = parser.parse_args()
    if args.command == "metadata":
        check_metadata(args.version, args.dist)
    elif args.command == "installed":
        check_install(args.version, args.artifact)
    else:
        download_release(args.version, args.manifest, args.index, args.directory)


if __name__ == "__main__":
    main()
