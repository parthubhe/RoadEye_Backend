"""Build E1-E6 and explicitly map original layers for pretrained weight transfer."""
import copy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    'E1': 'p2', 'E2': 'weighted_fusion', 'E3': 'p2_weighted_fusion',
    'E4': 'yolo26m', 'E5': 'directional', 'E6': 'single_simam',
}


def build_config(experiment, simam_base='E3'):
    if experiment not in EXPERIMENTS or simam_base not in {'E1', 'E2', 'E3', 'E4', 'E5'}:
        raise ValueError('Expected E1-E6; E6 base must be E1-E5')
    base = simam_base if experiment == 'E6' else experiment
    p2, weighted = base in {'E1', 'E3'}, base in {'E2', 'E3'}
    cfg = yaml.safe_load((ROOT / 'yolo26_base.yaml').read_text(encoding='utf-8'))
    cfg['scale'] = scale = 'm' if base == 'E4' else 's'
    _, width, maximum = cfg['scales'][scale]
    scaled = lambda c: math.ceil(min(c, maximum) * width / 8) * 8
    original = cfg['backbone'] + cfg['head']
    nodes, channels, mapped, transfer = [], [], {}, {}

    def append(f, repeat, module, args, c):
        idx = len(nodes)
        nodes.append([f, repeat, module, args])
        channels.append(c)
        return idx

    def fusion(f, branch_channels):
        return append(f, 1, 'WeightedFusion', [branch_channels], sum(branch_channels))

    # Keep original backbone, then remap every absolute head reference as layers are inserted.
    for old_idx, (f, n, module, args) in enumerate(original[:-1]):
        args = copy.deepcopy(args)
        refs = [mapped[x] if x != -1 else -1 for x in f] if isinstance(f, list) else mapped.get(f, f)
        if module == 'Concat':
            incoming = [channels[i] for i in refs]
            c = sum(incoming)
        elif module == 'nn.Upsample':
            c = channels[refs]
        else:
            c = scaled(args[0])
        idx = append(refs, n, module, args, c)
        transfer[idx] = old_idx
        if module == 'Concat' and weighted:
            idx = fusion(-1, incoming)
        if old_idx == 16 and base == 'E5':
            idx = append(-1, 1, 'DirectionalConv', [c, 7], c)
        if old_idx == 16 and experiment == 'E6' and not p2:
            idx = append(-1, 1, 'SimAM', [c], c)
        mapped[old_idx] = idx

    detection = [mapped[16], mapped[19], mapped[22]]
    if p2:
        up = append(mapped[16], 1, 'nn.Upsample', [None, 2, 'nearest'], channels[mapped[16]])
        incoming = [channels[up], channels[mapped[2]]]
        append([up, mapped[2]], 1, 'Concat', [1], sum(incoming))
        if weighted:
            fusion(-1, incoming)
        finest = append(-1, 2, 'C3k2', [128, True], scaled(128))
        if experiment == 'E6':
            finest = append(-1, 1, 'SimAM', [scaled(128)], scaled(128))
        # Append P2 LAST: keeps P3-P5 head widths and indices compatible with pretrained YOLO.
        detection.append(finest)
    head = append(detection, 1, 'Detect', ['nc'], None)
    transfer[head] = 23
    cfg['backbone'], cfg['head'] = nodes[:11], nodes[11:]
    cfg['source_layer_map'] = transfer
    cfg['experiment'] = experiment
    cfg['simam_base'] = simam_base if experiment == 'E6' else None
    return cfg


def write_config(experiment, directory, simam_base='E3', pretrained='auto'):
    cfg = build_config(experiment, simam_base)
    cfg['pretrained_source'] = f"yolo26{cfg['scale']}.pt" if pretrained == 'auto' else pretrained
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f'_{simam_base}' if experiment == 'E6' else ''
    target = directory / f"yolo26{cfg['scale']}_{experiment}{suffix}.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return target
