
import json
from pathlib import Path
import subprocess
import modal

ROOT = Path(__file__).parent
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libglib2.0-0')
         .pip_install_from_requirements(str(ROOT / 'requirements.txt'))
         .pip_install('img2dataset==1.47.0', 'pyarrow==19.0.0', 'numpy==1.26.4', 'opencv-python-headless==4.10.0.84')
         .add_local_python_source('src')
         .add_local_file(str(ROOT / 'config.yaml'), '/app/config.yaml'))
app = modal.App('custom-clip-training', image=image)
data = modal.Volume.from_name('clip-data', create_if_missing=True)
runs = modal.Volume.from_name('clip-runs', create_if_missing=True)


@app.function(cpu=16, memory=32768, timeout=86400, volumes={'/data': data})
def prepare(max_metadata_files: int = 0):
    from src.prepare_data import prepare as build
    try:
        return build('/data/datacomp_small', max_metadata_files or None, '/data/evaluation/exclusions.json', max_pairs=512 if max_metadata_files else None)
    finally:
        data.commit()


@app.function(cpu=8, memory=16384, timeout=7200, volumes={'/data': data})
def prepare_evaluation():
    from src.prepare_evaluation import prepare
    try:
        return prepare('/data/evaluation')
    finally:
        data.commit()


@app.function(gpu='H100', cpu=8, memory=32768, timeout=900, volumes={'/runs': runs})
def capacity_benchmark():
    from src.benchmark import benchmark
    try:
        return benchmark('/runs/benchmark/h100.json')
    finally:
        runs.commit()


@app.function(gpu='H100', cpu=8, memory=32768, timeout=900,
              volumes={'/data': data, '/runs': runs})
def evaluate_weights():
    import torch
    torch.set_num_threads(4)
    from src.checkpoint import load_inference
    from src.eval import evaluate_retrieval
    from src.dataset import atomic_json
    model = load_inference('/runs/clip/checkpoints/best_inference.pth', 'cuda')
    result = evaluate_retrieval(model, '/data/evaluation/test.json', 'cuda', cache_dir='/data/tokenizer')
    atomic_json('/runs/evaluation/test.json', result)
    runs.commit()
    return result


def launch(config_json, gpus, mode, resume):
    import os
    from src.train import load_config
    config = load_config('/app/config.yaml')
    config.update(json.loads(config_json))
    config['modal']['num_gpus'] = gpus
    if mode == 'smoke':
        config['training'].update(batch_size_per_gpu=8, num_workers=2, max_steps=10,
                                  max_runtime_seconds=1200, allow_small_dataset=True,
                                  checkpoint_steps=5)
        config['dataset']['index_dir'] = '/data/datacomp_small/index-first-1-limit-512'
        config['output_dir'] = f'/runs/{mode}-{gpus}/checkpoints'
    elif mode != 'train':
        raise ValueError(mode)
    target = Path(config['output_dir'])
    if not target.is_relative_to('/runs'):
        raise ValueError('Modal output_dir must be on /runs')
    Path('/tmp/run.yaml').write_text(__import__('yaml').safe_dump(config))
    env = os.environ.copy()
    env['CLIP_MODAL_VOLUME'] = 'clip-runs'
    command = ['torchrun', '--standalone', f'--nproc_per_node={gpus}', '-m', 'src.train', '--config', '/tmp/run.yaml']
    if resume:
        checkpoint = target / 'last_model.pth'
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        command += ['--resume', str(checkpoint)]
    try:
        subprocess.run(command, env=env, check=True)
    finally:
        runs.commit()
        data.commit()


@app.function(gpu='H100', cpu=16, memory=65536, timeout=86400,
              volumes={'/data': data, '/runs': runs})
def train_one(config_json: str, mode: str, resume: bool):
    launch(config_json, 1, mode, resume)


@app.function(gpu='H100:4', cpu=32, memory=131072, timeout=86400,
              volumes={'/data': data, '/runs': runs})
def train_four(config_json: str, mode: str, resume: bool):
    launch(config_json, 4, mode, resume)


@app.local_entrypoint()
def main(stage: str = 'train', num_gpus: int = 4, resume: bool = False,
         approve_full_run: bool = False, config: str = 'config.yaml'):
    if stage not in ('evaluate', 'benchmark', 'prepare-eval', 'prepare', 'prepare-full', 'prepare-pilot', 'smoke', 'train'):
        raise ValueError('Choose prepare-eval, prepare, prepare-pilot, smoke, benchmark, train, or evaluate')
    if stage in ('train', 'prepare', 'prepare-full') and not approve_full_run:
        raise ValueError('Full preparation/training requires explicit --approve-full-run')
    if stage == 'evaluate':
        print(evaluate_weights.remote())
        return
    if stage == 'benchmark':
        print(capacity_benchmark.remote())
        return
    if stage == 'prepare-eval':
        print(prepare_evaluation.remote())
        return
    if stage.startswith('prepare'):
        print(prepare.remote(1 if stage == 'prepare-pilot' else 0))
        return
    if num_gpus not in (1, 4):
        raise ValueError('Modal launcher supports 1 or 4 H100s; local torchrun supports other counts')
    import yaml
    serialized = json.dumps(yaml.safe_load(Path(config).read_text()))
    (train_one if num_gpus == 1 else train_four).remote(serialized, stage, resume)
