"""Download the pinned TorchGeo RESISC45 mirror; standard library only.

Source/checksum: https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/resisc45.py
Dataset author/license: https://gcheng-nwpu.github.io/ (CC BY-NC 4.0).
This prepares original files only; it does not change experiment splits.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import re
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

URL = ('https://huggingface.co/datasets/isaaccorley/resisc45/resolve/'
       '883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip')
SHA256 = 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53'
CLASSES = tuple('airplane airport baseball_diamond basketball_court beach bridge '
    'chaparral church circular_farmland cloud commercial_area dense_residential desert '
    'forest freeway golf_course ground_track_field harbor industrial_area intersection '
    'island lake meadow medium_residential mobile_home_park mountain overpass palace '
    'parking_lot railway railway_station rectangular_farmland river roundabout runway '
    'sea_ice ship snowberg sparse_residential stadium storage_tank tennis_court terrace '
    'thermal_power_station wetland'.split())
IMAGES_PER_CLASS = 700
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_dataset(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError('Dataset directory does not exist: ' + str(root))
    dirs = {p.name for p in root.iterdir() if p.is_dir()}
    if dirs != set(CLASSES):
        raise ValueError('Expected exactly {} class folders in {}. Missing: {}; extra: {}'.format(
            len(CLASSES), root, sorted(set(CLASSES)-dirs), sorted(dirs-set(CLASSES))))
    counts = {}
    for name in CLASSES:
        files = [p for p in (root/name).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        counts[name] = len(files)
        if len(files) != IMAGES_PER_CLASS or any(p.stat().st_size == 0 for p in files):
            raise ValueError('{}: expected {} nonempty images; found {}'.format(name, IMAGES_PER_CLASS, len(files)))
    return counts


def download_archive(url, destination, expected_sha256=SHA256, retries=3, timeout=60):
    """Resume a .part file, verify hash, then atomically publish the archive."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if retries < 1 or timeout <= 0:
        raise ValueError('retries and timeout must be positive')
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError('Existing archive has the wrong SHA256: {}. Move it aside before retrying.'.format(destination))
        print('Reuse verified archive:', destination, flush=True)
        return destination
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists() and sha256_file(partial) == expected_sha256:
        partial.replace(destination)
        return destination
    for attempt in range(1, retries+1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'resisc45-data-preparer/1.0', 'Accept-Encoding': 'identity',
                   'Cache-Control': 'no-cache'}
        if offset:
            headers['Range'] = 'bytes={}-'.format(offset)
        print('Download attempt {}/{}; resume at {:.1f} MiB'.format(attempt,retries,offset/2**20), flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                ctype = response.headers.get('Content-Type', '').lower()
                if 'text/html' in ctype or 'application/json' in ctype:
                    raise ValueError('Download returned a web/login/error page instead of the ZIP archive.')
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                    if not match or int(match[1]) != offset or int(match[2]) < offset or int(match[3]) <= int(match[2]):
                        raise ValueError('Invalid Content-Range; refusing to append incorrect bytes.')
                    total = int(match[3])
                    mode = 'ab'
                elif status == 200:
                    if offset:
                        print('Server does not support Range; restarting this archive download.', flush=True)
                    offset = 0
                    total = int(response.headers.get('Content-Length', '0'))
                    mode = 'wb'
                else:
                    raise OSError('Unexpected HTTP status: '+str(status))
                started = last = time.monotonic()
                received = offset
                with partial.open(mode) as out:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now-last >= 5:
                            suffix = '/{:.1f} MiB ({:.1f}%)'.format(total/2**20,100*received/total) if total else ' MiB'
                            speed = (received-offset)/2**20/max(now-started, .001)
                            print('Downloaded {:.1f}{}; {:.2f} MiB/s'.format(received/2**20,suffix,speed), flush=True)
                            last = now
                if total and received != total:
                    raise OSError('Incomplete download: {} / {} bytes'.format(received,total))
            actual = sha256_file(partial)
            if actual != expected_sha256:
                raise ValueError('SHA256 mismatch. Expected {}; got {}. The file was NOT extracted: {}. '
                    'Move this .part file aside before retrying.'.format(expected_sha256,actual,partial))
            partial.replace(destination)
            print('SHA256 verified:', actual, flush=True)
            return destination
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            print('Download failed:', error, flush=True)
            if attempt == retries:
                raise RuntimeError('Download failed after {} attempts. Partial file retained: {}. '
                    'Retry on a login node with internet access, or transfer this verified ZIP using --archive.'.format(retries,partial)) from error
            time.sleep(min(2*attempt, 6))
    raise RuntimeError('Download did not complete')


def extract_archive(archive, destination):
    """Validate the layout, stream verified ZIP members into a staging directory."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Refusing to overwrite a nonempty dataset path: '+str(destination))
    staging = destination.parent / ('.' + destination.name + '.extracting')
    if staging.is_symlink():
        raise ValueError('Extraction staging directory must not be a symbolic link')
    with zipfile.ZipFile(archive) as z:
        members, names = [], set()
        counts = Counter()
        for info in z.infolist():
            parts = PurePosixPath(info.filename.replace('\\','/')).parts
            if not parts or '..' in parts or PurePosixPath(info.filename).is_absolute() or any(':' in p for p in parts):
                raise ValueError('Unsafe ZIP member: '+info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Symbolic links are not allowed in the dataset ZIP')
            if info.is_dir():
                continue
            if parts[0] == 'NWPU-RESISC45':
                parts = parts[1:]
            if not parts:
                raise ValueError('Unexpected root file in dataset ZIP: '+info.filename)
            if len(parts) != 2 or parts[0] not in CLASSES or Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                # Non-image metadata is not installed into the ImageFolder.
                if Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS:
                    raise ValueError('Unexpected image layout: '+info.filename)
                continue
            relative = Path(*parts)
            if relative in names or info.file_size == 0:
                raise ValueError('Duplicate or empty image member: '+info.filename)
            names.add(relative)
            counts[parts[0]] += 1
            members.append((info,relative))
        if dict(counts) != {c:IMAGES_PER_CLASS for c in CLASSES}:
            raise ValueError('ZIP is not the complete expected dataset: '+str(dict(counts)))
        staging.mkdir(parents=True,exist_ok=True)
        for index,(info,relative) in enumerate(members, 1):
            target = staging/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.resolve().is_relative_to(staging.resolve()) or target.is_symlink():
                raise ValueError('Extraction path escapes staging directory')
            with z.open(info) as source, target.open('wb') as out:
                shutil.copyfileobj(source,out,1024*1024)
            if index % 3500 == 0 or index == len(members):
                print('Extracted {}/{} images'.format(index,len(members)),flush=True)
    validate_dataset(staging)
    if destination.exists():
        destination.rmdir()  # Empty only; never recursively remove user files.
    staging.rename(destination)
    return destination


def prepare(data_root, archive=None, url=URL, retries=3, timeout=60, check_only=False):
    root = Path(data_root).expanduser().resolve()
    if (root/'NWPU-RESISC45').is_dir():
        root = root/'NWPU-RESISC45'
    if root.exists() and any(root.iterdir()):
        counts = validate_dataset(root)
        print('Dataset ready: {} classes, {} images: {}'.format(len(counts),sum(counts.values()),root),flush=True)
        return root
    if check_only:
        validate_dataset(root)
        return root
    if archive:
        archive = Path(archive).expanduser().resolve()
        if not archive.is_file() or sha256_file(archive) != SHA256:
            raise ValueError('--archive must be the ZIP matching the pinned TorchGeo SHA256: '+str(archive))
    else:
        archive = download_archive(url,root.parent/'NWPU-RESISC45.zip',retries=retries,timeout=timeout)
    root = extract_archive(archive,root)
    counts = validate_dataset(root)
    manifest = {'url':url,'archive_sha256':SHA256,'archive_bytes':archive.stat().st_size,
                'data_root':str(root),'classes':len(counts),'images':sum(counts.values()),
                'source':'Hugging Face mirror used by TorchGeo; original dataset by NWPU',
                'license':'CC BY-NC 4.0, https://gcheng-nwpu.github.io/'}
    (root.parent/(root.name+'_download.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DATASET READY:',root,flush=True)
    print('{} classes x {} images = {}'.format(len(counts),IMAGES_PER_CLASS,sum(counts.values())),flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='./data/NWPU-RESISC45',help='The directory containing the 45 class folders')
    p.add_argument('--archive',help='Use a previously downloaded, checksum-matching ZIP without network access')
    p.add_argument('--url',default=URL,help='Alternative URL for the SAME checksum-matching ZIP')
    p.add_argument('--retries',type=int,default=3)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--check-only',action='store_true')
    a = p.parse_args()
    try:
        prepare(a.data_root,a.archive,a.url,a.retries,a.timeout,a.check_only)
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as error:
        p.exit(1,'Dataset preparation failed: '+str(error)+'\n')


if __name__=='__main__':
    main()
