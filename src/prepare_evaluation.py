
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import requests
from .dataset import atomic_json, image_digest
from PIL import Image


URLS = {
    'images': 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',
    'annotations': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


def download(url, path):
    """Range resume with bounded retries; a failed response never becomes final."""
    path = Path(path)
    if path.exists():
        return
    temp = path.with_suffix('.partial')
    for attempt in range(3):
        offset = temp.stat().st_size if temp.exists() else 0
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(30, 120)) as response:
                if response.status_code == 416 and temp.exists():
                    try:
                        with zipfile.ZipFile(temp) as archive:
                            complete = archive.testzip() is None
                    except zipfile.BadZipFile:
                        complete = False
                    if complete:
                        os.replace(temp, path)
                        return
                    temp.unlink()
                    continue
                response.raise_for_status()
                append = offset and response.status_code == 206
                with open(temp, 'ab' if append else 'wb') as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
            # CRC validation verifies the downloaded archives before promotion.
            with zipfile.ZipFile(temp) as archive:
                if archive.testzip() is not None:
                    raise ValueError('ZIP CRC validation failed')
            os.replace(temp, path)
            return
        except (requests.RequestException, zipfile.BadZipFile, ValueError):
            if attempt == 2:
                raise


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for kind, url in URLS.items():
        download(url, root / f'{kind}.zip')
    with zipfile.ZipFile(root / 'annotations.zip') as archive:
        annotation = json.loads(archive.read('annotations/captions_val2017.json'))
    captions = {}
    for item in annotation['annotations']:
        captions.setdefault(item['image_id'], []).append(item['caption'])
    unique, excluded_hashes, excluded_urls = {}, set(), set()
    completed_extraction = (root / 'source.json').exists()
    (root / 'val2017').mkdir(exist_ok=True)
    with zipfile.ZipFile(root / 'images.zip') as archive:
        for i, item in enumerate(sorted(annotation['images'], key=lambda x: x['id'])):
            relative = f"val2017/{item['file_name']}"
            blob = archive.read(relative)
            target = root / relative
            if not completed_extraction and not target.exists():
                temp = target.with_suffix('.tmp')
                temp.write_bytes(blob)
                temp.replace(target)
            with Image.open(io.BytesIO(blob)) as image:
                key = image_digest(image)
            excluded_hashes.update((key, hashlib.sha256(blob).hexdigest()))
            excluded_urls.add(item['coco_url'])
            row = {'image': relative, 'image_id': item['id'], 'captions': captions[item['id']],
                   'url': item['coco_url'], 'license_id': item['license']}
            if key in unique:
                existing = unique[key]
                existing['captions'] = list(dict.fromkeys(existing['captions'] + row['captions']))
                existing.setdefault('urls', [existing['url']]).append(row['url'])
            else:
                unique[key] = row
            if (i + 1) % 500 == 0:
                print(json.dumps({'audited_coco_images': i + 1, 'unique_images': len(unique)}), flush=True)
    records = list(unique.values())
    # Fixed split by ascending first image ID after exact-image deduplication.
    val, test = records[:1000], records[1000:]
    atomic_json(root / 'val.json', val)
    atomic_json(root / 'test.json', test)
    atomic_json(root / 'licenses.json', {'info': annotation['info'], 'licenses': annotation['licenses']})
    atomic_json(root / 'exclusions.json', {'image_hashes': sorted(excluded_hashes), 'urls': sorted(excluded_urls)})
    hashes = {}
    for kind in URLS:
        with open(root / f'{kind}.zip', 'rb') as stream:
            hashes[kind] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'dataset': 'COCO val2017', 'split': 'custom sorted unique image IDs: first 1000 validation, remainder test',
              'validation_images': len(val), 'test_images': len(test), 'source_urls': URLS,
              'archives_sha256': hashes, 'exact_image_duplicates_removed': len(annotation['images']) - len(records)}
    atomic_json(root / 'source.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/evaluation')
    print(json.dumps(prepare(p.parse_args().root), indent=2))
