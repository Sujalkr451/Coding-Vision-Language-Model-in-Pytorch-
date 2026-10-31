import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(directory):
    directory = Path(directory)
    events = [json.loads(x) for x in (directory / 'events.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, key in [('Training', 'train_loss'), ('Validation (batch negatives)', 'val_loss')]:
        rows = [x for x in events if key in x]
        axes[0].plot([x['global_step'] for x in rows], [x[key] for x in rows], label=label, marker='o' if len(rows) <= 50 else None)
    rows = [x for x in events if 'mean_recall' in x]
    axes[1].plot([x['global_step'] for x in rows], [x['mean_recall'] for x in rows], marker='o')
    axes[0].set(title='Contrastive loss', xlabel='Optimizer step', ylabel='Loss')
    axes[0].legend()
    axes[1].set(title='Held-out retrieval', xlabel='Optimizer step', ylabel='Mean Recall@1,5,10')
    fig.tight_layout()
    fig.savefig(directory / 'training_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    plot(p.parse_args().directory)
