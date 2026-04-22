"""
Sequential training script — runs VGG16 → AlexNet → ResNet → MobileNet one after another.
Each model gets full GPU memory. Logs per model to logs/<model>_train.log.

Usage:
    nohup env PYTHONPATH=/workspace/Recognition_Weather \
        /workspace/venv/bin/python train_all.py > logs/train_all.log 2>&1 &
"""

import os
import sys
import time
import subprocess

PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = [
    ('VGG16',     'src/baselines/vgg16.py'),
    ('AlexNet',   'src/baselines/alexnet.py'),
    ('ResNet50',  'src/baselines/resnet.py'),
    ('MobileNet', 'src/baselines/mobilenet.py'),
    ('ViT',       'src/baselines/vit.py'),
]

env = os.environ.copy()
env['PYTHONPATH'] = PROJECT_DIR

results = {}

for name, script in MODELS:
    log_path = os.path.join(LOG_DIR, f'{name.lower()}_train.log')
    print(f'\n{"="*60}')
    print(f'Starting {name} — log: {log_path}')
    print(f'{"="*60}')

    start = time.time()
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(
            [PYTHON, os.path.join(PROJECT_DIR, script)],
            env=env,
            cwd=PROJECT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

    elapsed = time.time() - start
    status = 'SUCCESS' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
    results[name] = {'status': status, 'time_min': elapsed / 60}
    print(f'{name} finished in {elapsed/60:.1f} min — {status}')

print(f'\n{"="*60}')
print('All models done. Summary:')
print(f'{"="*60}')
for name, r in results.items():
    print(f'  {name:<12} {r["status"]:<25} {r["time_min"]:.1f} min')
