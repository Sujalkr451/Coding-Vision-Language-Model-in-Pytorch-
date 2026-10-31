
import argparse
import json
from pathlib import Path
from .dataset import build_index, atomic_json


def prepare(root, max_metadata_files=None, exclusions=None, max_pairs=None):
    from huggingface_hub import HfApi, hf_hub_download
    import img2dataset
    root = Path(root)
    metadata = root / 'metadata'
    shards = root / 'shards'
    metadata.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    repo = 'mlfoundations/datacomp_small'
    lock = root / 'source.json'
    # Pin the metadata snapshot for resume; never silently change its file list.
    if lock.exists():
        source = json.loads(lock.read_text())
    else:
        info = HfApi().dataset_info(repo)
        files = sorted(x.rfilename for x in info.siblings if x.rfilename.endswith('.parquet'))
        source = dict(repo=repo, revision=info.sha, files=files,
                      license='CC-BY-4.0 metadata; images retain original rights')
        atomic_json(lock, source)
    files = source['files'][:max_metadata_files] if max_metadata_files else source['files']
    # Separate partial/full output trees so img2dataset shard IDs stay stable on resume.
    selection = 'full' if max_metadata_files is None else f'first-{max_metadata_files}'
    if max_pairs:
        selection += f'-limit-{max_pairs}'
    selected_metadata = metadata / selection
    selected_metadata.mkdir(exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    def fetch(file):
        return hf_hub_download(repo, file, repo_type='dataset', revision=source['revision'], local_dir=selected_metadata)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    # Preserve source hashes under a non-reserved name; img2dataset reserves sha256.
    # Filter held-out source hashes/URLs BEFORE resizing or face blurring.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    exclude = json.loads(Path(exclusions).read_text()) if exclusions else {}
    exclusion_sha = __import__('hashlib').sha256(json.dumps(exclude, sort_keys=True).encode()).hexdigest()
    policy = root / f'exclusions-{selection}.json'
    if policy.exists() and json.loads(policy.read_text())['sha256'] != exclusion_sha:
        raise ValueError('Exclusions changed: use a new preparation root')
    atomic_json(policy, {'sha256': exclusion_sha})
    filtered = root / f'input-{selection}'
    filtered.mkdir(exist_ok=True)
    for i, file in enumerate(files):
        target = filtered / f'{i:05d}.parquet'
        if target.exists():
            continue
        table = pq.read_table(selected_metadata / file, columns=['url', 'text', 'uid', 'sha256', 'face_bboxes'])
        keep = pc.and_(pc.invert(pc.is_in(table['sha256'], value_set=pa.array(exclude.get('image_hashes', []), type=pa.string()))),
                       pc.invert(pc.is_in(table['url'], value_set=pa.array(exclude.get('urls', []), type=pa.string()))))
        table = table.filter(keep)
        if max_pairs:
            table = table.slice(0, max_pairs)
        table = table.rename_columns(['url', 'text', 'uid', 'source_sha256', 'face_bboxes'])
        tmp = target.with_suffix('.tmp')
        pq.write_table(table, tmp)
        tmp.replace(target)
    output = shards / selection
    output.mkdir(exist_ok=True)
    img2dataset.download(
        url_list=str(filtered), input_format='parquet', url_col='url', caption_col='text',
        output_folder=str(output), output_format='webdataset', image_size=512,
        resize_mode='keep_ratio_largest', resize_only_if_bigger=True, encode_format='jpg',
        bbox_col='face_bboxes', save_additional_columns=['uid', 'source_sha256'],
        processes_count=8, thread_count=32, number_sample_per_shard=10000,
        retries=2, timeout=15, incremental_mode='incremental', enable_wandb=False)
    # Downloader parquet rows include failures, unlike the image-only tar audit.
    statuses = {}
    for status_file in sorted(output.glob('*.parquet')):
        table = pq.read_table(status_file, columns=['status'])
        for item in pc.value_counts(table['status']).to_pylist():
            status = str(item['values'])
            statuses[status] = statuses.get(status, 0) + item['counts']
    atomic_json(root / f'download-counts-{selection}.json',
                {'statuses': statuses, 'attempted': sum(statuses.values())})
    manifest = build_index(output, root / ('index' if selection == 'full' else f'index-{selection}'), exclusions)
    from .cache_tokens import cache_tokens
    cache_tokens(root / ('index' if selection == 'full' else f'index-{selection}'), str(root.parent / 'tokenizer'))
    atomic_json(root / f'preparation-{selection}.json', {'source': source, 'download_statuses': statuses, 'manifest': manifest})
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/datacomp_small')
    p.add_argument('--max-metadata-files', type=int)
    p.add_argument('--exclusions', required=True, help='Held-out evaluation hash/URL exclusion manifest')
    a = p.parse_args()
    print(json.dumps(prepare(a.root, a.max_metadata_files, a.exclusions), indent=2))
