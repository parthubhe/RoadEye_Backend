"""Offline architecture/gradient/serialization checks, plus a tiny end-to-end training option."""
import argparse
import gc
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import get_flops
from PIL import Image, ImageDraw

from architectures import write_config
from experiment_trainer import transfer_pretrained
from modules import register_modules

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-smoke', action='store_true')
    parser.add_argument('--weights', type=Path, default=ROOT.parents[1] / 'yolo26s.pt')
    parser.add_argument('--medium-weights', type=Path, default=ROOT / 'workspace/weights/yolo26m.pt')
    args = parser.parse_args()
    register_modules()
    torch.set_num_threads(2)
    report = []
    with tempfile.TemporaryDirectory(prefix='verify_', dir=ROOT) as temporary:
        tmp = Path(temporary)
        for exp, base in [('E1', 'E3'), ('E2', 'E3'), ('E3', 'E3'), ('E4', 'E3'), ('E5', 'E3'),
                          ('E6', 'E1'), ('E6', 'E2'), ('E6', 'E3'), ('E6', 'E4'), ('E6', 'E5')]:
            config = write_config(exp, tmp, base, pretrained=None)
            wrapper = YOLO(str(config))
            model = wrapper.model.cpu().float()
            transfer = None
            source = args.weights if model.yaml['scale'] == 's' else args.medium_weights
            if source.is_file():
                transfer = transfer_pretrained(model, source)
            flops = get_flops(model, imgsz=640)
            model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
            model.train()
            batch = {'img': torch.rand(2, 3, 64, 64), 'batch_idx': torch.tensor([0., 1.]),
                     'cls': torch.tensor([[0.], [1.]]),
                     'bboxes': torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.6, 0.2, 0.3]])}
            losses, _ = model.loss(batch)
            loss = losses.sum()
            assert torch.isfinite(loss), (exp, base, loss)
            loss.backward()
            trained = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert trained and all(torch.isfinite(g).all() for g in trained.values())
            for layer in model.model:
                if layer.__class__.__name__ in {'WeightedFusion', 'DirectionalConv'}:
                    assert all(p.grad is not None for p in layer.parameters())
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 128, 128))[0]
            assert output.shape[-1] == 6 and torch.isfinite(output).all()
            path = tmp / f'{exp}_{base}.pt'
            wrapper.save(path)
            loaded = YOLO(str(path)).model.cpu().float().eval()
            with torch.no_grad():
                restored = loaded(torch.zeros(1, 3, 128, 128))[0]
            assert restored.shape == output.shape and torch.isfinite(restored).all()
            strides = model.stride.tolist()
            assert strides == ([8., 16., 32., 4.] if (base if exp == 'E6' else exp) in {'E1', 'E3'} else [8., 16., 32.])
            item = {'experiment': exp, 'base': base if exp == 'E6' else None,
                    'parameters': sum(p.numel() for p in model.parameters()), 'strides': strides,
                    'estimated_gflops_640_unfused': flops,
                    'backward_finite': True, 'checkpoint_reload': True,
                    'pretrained_tensors_loaded': transfer['loaded_tensors'] if transfer else None}
            report.append(item)
            print(json.dumps(item), flush=True)
            del wrapper, model, loaded, trained, loss, losses, restored, output
            gc.collect()
        # Verify that standalone notebooks have valid Python cells and embedded sources.
        for notebook in ROOT.glob('E*.ipynb'):
            nb = json.loads(notebook.read_text(encoding='utf-8'))
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{notebook.name}:{i}', 'exec')
            assert nb['nbformat'] == 4
        if args.training_smoke:
            # Generated synthetic labels test execution, not accuracy.
            data = tmp / 'tiny'
            for split in ['train', 'valid', 'test']:
                for subdir in ['images', 'labels']:
                    (data / split / subdir).mkdir(parents=True)
                for i in range(2):
                    image = Image.new('RGB', (64, 64), (60, 60, 60))
                    ImageDraw.Draw(image).rectangle((24, 24, 40, 40), outline='white', width=2)
                    image.save(data / split / 'images' / f'{i}.jpg')
                    (data / split / 'labels' / f'{i}.txt').write_text(f'{i} 0.5 0.5 0.25 0.25\n')
            from dataset import NAMES
            import yaml
            (data / 'data.yaml').write_text(yaml.safe_dump({'names': NAMES}))
            from run_experiment import run
            run(['--experiment', 'E3', '--local-data', str(data), '--allow-count-mismatch',
                 '--workspace', str(tmp / 'workspace'), '--project', str(tmp / 'runs'),
                 '--epochs', '1', '--batch', '2', '--imgsz', '64', '--device', 'cpu',
                 '--workers', '0', '--no-amp', '--pretrained', str(args.weights)])
            runs = list((tmp / 'runs').glob('*/pretrained_transfer.json'))
            assert runs, 'Trainer lost the custom pretrained transfer'
            report.append({'synthetic_one_epoch_E3': 'passed', 'trainer_transfer_verified': True})
    (ROOT / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('All checks passed.')


if __name__ == '__main__':
    main()
