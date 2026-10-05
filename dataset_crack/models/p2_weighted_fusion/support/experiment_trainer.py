"""Preserve pretrained transfer across inserted layers, including inside Trainer rebuilds."""
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from modules import register_modules


def transfer_pretrained(target, source_path):
    source = YOLO(str(source_path)).model.cpu().float()
    mapping = target.yaml['source_layer_map']
    target_state, source_state = target.state_dict(), source.state_dict()
    transferred, skipped = {}, []
    for key, value in target_state.items():
        pieces = key.split('.')
        new_index = int(pieces[1])
        old_index = mapping.get(new_index, mapping.get(str(new_index)))
        source_key = '.'.join(['model', str(old_index), *pieces[2:]])
        if old_index is not None and source_key in source_state and source_state[source_key].shape == value.shape:
            transferred[key] = source_state[source_key]
        else:
            skipped.append(key)
    target.load_state_dict(transferred, strict=False)
    # Every unchanged backbone tensor must transfer; fail rather than silently start it from scratch.
    missing_backbone = [k for k in skipped if int(k.split('.')[1]) < 11]
    if missing_backbone:
        raise ValueError(f'Incompatible pretrained backbone: {missing_backbone[:5]}')
    report = {'source': str(source_path), 'loaded_tensors': len(transferred),
              'total_tensors': len(target_state), 'fresh_tensors': skipped,
              'loaded_state_elements': sum(v.numel() for v in transferred.values()),
              'all_backbone_tensors_loaded': True}
    print(f"Mapped pretrained transfer: {len(transferred)}/{len(target_state)} tensors; full backbone loaded")
    return report


class ExperimentTrainer(DetectionTrainer):
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume and overrides.get('data'):
            # A restarted cloud session may have the same YAML filename but a new cache root.
            self.args.data = overrides['data']

    def get_model(self, cfg=None, weights=None, verbose=True):
        register_modules()
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        source = model.yaml.get('pretrained_source')
        if weights is None and source:
            report = transfer_pretrained(model, source)
            (Path(self.save_dir) / 'pretrained_transfer.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return model
