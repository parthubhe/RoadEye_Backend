"""Append the September 2026 experiment audit, preserving the existing DOCX package."""
import csv
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'BFY_ModelComp_report_v2.docx'
NOTEBOOK = ROOT / 'e3-p2-weighted-fusion (1).ipynb'
RUNS = ROOT / 'dataset_crack/notebook/runs'
LEGACY = ROOT / 'dataset_crack/runs/pipe_proto'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
ET.register_namespace('w', W)
MARKER = '7. E1-E6 architecture experiments (September 2026 update)'


def element(name, attrs=None):
    return ET.Element(f'{{{W}}}{name}', {f'{{{W}}}{k}': str(v) for k, v in (attrs or {}).items()})


def paragraph(text, style=None):
    p = element('p')
    if style:
        prop = element('pPr')
        prop.append(element('pStyle', {'val': style}))
        if style.startswith('Heading'):
            prop.append(element('keepNext'))
        p.append(prop)
    run = element('r')
    t = element('t')
    t.text = text
    run.append(t)
    p.append(run)
    return p


def table(headers, rows):
    tbl = element('tbl')
    props = element('tblPr')
    props.append(element('tblW', {'w': 5000, 'type': 'pct'}))
    borders = element('tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        borders.append(element(side, {'val': 'single', 'sz': 4, 'color': 'B8C4D2'}))
    props.append(borders)
    tbl.append(props)
    for index, row in enumerate([headers, *rows]):
        tr = element('tr')
        trprop = element('trPr')
        trprop.append(element('cantSplit'))
        if index == 0:
            trprop.append(element('tblHeader'))
        tr.append(trprop)
        for value in row:
            tc = element('tc')
            p = paragraph(str(value))
            run = p.find(f'{{{W}}}r')
            rp = element('rPr')
            rp.append(element('sz', {'val': 19}))
            if index == 0:
                rp.append(element('b'))
                tcp = element('tcPr')
                tcp.append(element('shd', {'fill': 'E6EDF5'}))
                tc.append(tcp)
            run.insert(0, rp)
            tc.append(p)
            tr.append(tc)
        tbl.append(tr)
    return tbl


def notebook_summary():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    found = []
    for cell in nb['cells']:
        for output in cell.get('outputs', []):
            text = output.get('text', '')
            text = ''.join(text) if isinstance(text, list) else text
            for match in re.finditer(r'\{\s*"experiment"\s*:', text):
                try:
                    obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
                except ValueError:
                    continue
                if obj.get('experiment') == 'E3' and obj.get('split') == 'val':
                    found.append(obj)
    if len(found) != 1:
        raise ValueError(f'Expected one E3 validation summary, found {len(found)}')
    return found[0]


def csv_best(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))


def main():
    e3 = notebook_summary()
    e5path = RUNS / 'E5_directional_v5_normal_seed0_20260928_185759_471e27_val/summary.json'
    e5 = json.loads(e5path.read_text(encoding='utf-8'))
    baseline = csv_best(LEGACY / 'yolo26s_v5_cctv/results.csv')
    parallel = csv_best(LEGACY / 'yolo26s_v5_parallel_simam_cbam_se/results.csv')
    with ZipFile(REPORT) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    contents = {info.filename: data for info, data in entries}
    document = ET.fromstring(contents['word/document.xml'])
    body = document.find(f'{{{W}}}body')
    if any(''.join(p.itertext()) == MARKER for p in body):
        raise ValueError('Experiment update already exists; refusing to duplicate it.')
    for text in body.iter(f'{{{W}}}t'):
        if text.text == 'Architecture measured; training run pending':
            text.text = 'Completed 80 epochs; see Section 8 for validation results'

    additions = [paragraph(MARKER, 'Heading1')]
    additions.append(paragraph('Status recorded 29 September 2026. E1, E2 and E4 are planned/pending; no measured accuracy is available for them. E3 and E5 completed 80 epochs. E6 completed four epochs and stopped during epoch 5 with CUDA out-of-memory; it is not a completed accuracy comparison.'))
    additions.append(table(['Batch / ID', 'Architecture', 'Purpose', 'Status'], [
        ['A / E1', 'YOLO26s + P2 detection head', 'Test whether stride-4 features improve fine/small defect detection.', 'Pending; 80 epochs planned'],
        ['A / E2', 'YOLO26s + weighted residual PAN fusion', 'Test learned normalized weights for multi-scale feature fusion.', 'Pending; 80 epochs planned'],
        ['B / E3', 'YOLO26s + P2 + weighted residual PAN fusion', 'Test the combined P2/fusion design.', 'Completed: 80 epochs'],
        ['B / E4', 'Standard YOLO26m', 'Capacity control: test a larger backbone without custom attention.', 'Pending; 80 epochs planned'],
        ['C / E5', 'YOLO26s + directional residual block at P3', 'Depthwise 1x7 and 7x1 kernels plus pointwise fusion.', 'Completed: 80 epochs'],
        ['C / E6', 'E3 parent + one SimAM block at P2', 'Isolate parameter-free attention on the chosen parent.', 'Interrupted: OOM in epoch 5/80'],
    ]))
    additions.append(paragraph('Implementation scope: E2/E3 use weighted residual fusion within the YOLO PAN neck, not a full BiFPN reproduction. E5 is not deformable or dynamic-snake convolution. E6 defaults provisionally to E3 and allows a different E1-E5 parent. SPPF and C2PSA are inherited YOLO26 components, not newly invented contributions.'))
    additions.append(paragraph('Protocol and reproducibility', 'Heading2'))
    additions.append(paragraph('Normal v5: 9,486 training / 2,014 validation / 1,986 test images, with crack, corrosion_rust, sediment_deposit, root_intrusion and joint_defect. The experiment suite preserves the existing split; it does not use the model-error-filtered diagnostic subset. This update does not independently verify pipe identity separation or absence of near-duplicate leakage.'))
    additions.append(paragraph('Configured recipe: 80 epochs, 640-pixel input, batch 16, AdamW, lr0=0.002, lrf=0.01, cosine schedule, seed 0, deterministic mode, AMP, two workers and mosaic disabled for the final 10 epochs. Common augmentations: HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, horizontal flip 0.5 and mosaic 1.0; rotations, shear, perspective, vertical flip, mixup and copy-paste are zero.'))
    additions.append(paragraph('Ultralytics 8.4.120; shape-compatible pretrained tensors are transferred using explicit original-to-inserted-layer mappings. E3 and E6 logs confirm 696/942 tensors loaded and the complete backbone transferred. New modules/class outputs initialize fresh. The normal dataset is not offline CLAHE-preprocessed, but E3 logs show optional Albumentations CLAHE, blur, median blur and grayscale at p=0.01 each. Match these dependencies/settings across machines.'))

    additions.append(paragraph('8. Measured validation results', 'Heading1'))
    def csv_metrics(row):
        return [f"{float(row['metrics/' + key + '(B)']) * 100:.2f}%" for key in ('precision', 'recall', 'mAP50', 'mAP50-95')]
    def summary_metrics(row):
        return [f'{row[key] * 100:.2f}%' for key in ('precision', 'recall', 'mAP50', 'mAP50_95')]
    additions.append(table(['Model', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'Evidence'], [
        ['Plain YOLO26s', *csv_metrics(baseline), f"Training-val CSV, epoch {baseline['epoch']}"],
        ['Parallel SimAM+CBAM+SE', *csv_metrics(parallel), f"Training-val CSV, epoch {parallel['epoch']}"],
        ['E3', *summary_metrics(e3), 'Post-training validation summary'],
        ['E5', *summary_metrics(e5), 'Post-training validation summary'],
        ['E1 / E2 / E4', 'N/A', 'N/A', 'N/A', 'N/A', 'Pending; no invented results'],
        ['E6', 'N/A', 'N/A', 'N/A', 'N/A', 'Incomplete; OOM'],
    ]))
    additions.append(paragraph('All values above are validation percentages, not test accuracy. Historical CSV rows were selected by highest mAP50-95, not by independent maxima for each metric. E3/E5 use their explicit post-training validation summaries. Small differences in evaluation settings, batch size, precision and environment can affect results; reevaluate all best checkpoints identically before claiming a formal ranking. Earlier rounded baseline values in Section 1 are retained as historical entries.'))
    additions.append(paragraph('E3 ran on a Tesla T4 and reported 10.251 hours for 80 epochs. E5 results.csv records 11,734.7 seconds (about 3.26 hours) through epoch 80 locally, excluding subsequent final evaluations. These are different machines and are not an architecture speed benchmark. E6 logged approximately 11.4 GB GPU memory before the backward-pass OOM.'))
    additions.append(paragraph('9. Class-wise diagnosis', 'Heading1'))
    e5classes = {row['name']: row for row in e5['per_class']}
    additions.append(table(['Class', 'E3 AP50', 'E5 AP50', 'E3 AP50-95', 'E5 AP50-95'], [
        [row['name'], f"{row['AP50']*100:.2f}%", f"{e5classes[row['name']]['AP50']*100:.2f}%",
         f"{row['AP50_95']*100:.2f}%", f"{e5classes[row['name']]['AP50_95']*100:.2f}%"] for row in e3['per_class']
    ]))
    additions.append(paragraph('Cracks, sediment and joints remain the weakest classes in both completed experiments. Corrosion and roots exceed 92% AP50. Similar per-class outcomes suggest these architectural additions have not addressed the dominant errors; they do not prove the data are defective or that further architecture changes cannot help. Crack AP50-95 near 43% warrants investigating localization and annotation consistency as well as missed detections.'))
    additions.append(paragraph('10. Conclusions and next experiments', 'Heading1'))
    additions.append(paragraph('Neither E3 nor E5 currently demonstrates an improvement over plain YOLO26s. E3 exceeds E5 by about 1.19 percentage points in mAP50-95 but only 0.18 points in mAP50. Single-seed differences are preliminary. E6 is a resource-limited incomplete experiment, not evidence against SimAM accuracy.'))
    for text in [
        'Complete E1/E2/E4 to separate P2, fusion and model-capacity effects; rerun finalists across several seeds.',
        'Compare shared baseline/E3/E5 validation errors: missed defects, mislocalized boxes, texture false positives and annotation inconsistencies. Review a manageable sample and analogous training examples; do not delete difficult evaluation images based on model failure.',
        'Test 640 versus 800/960 validation resolution on existing checkpoints, then matched training if useful. Evaluate a controlled reduction in mosaic/scale augmentation rather than assuming more epochs or more modules will help.',
        'Retry E6 with a smaller batch or a larger-memory GPU and record the changed configuration. Keep the full test set unchanged for final evaluation, not repeated experiment selection.',
        'No 90% mAP50 guarantee is supported. Even perfect crack AP50 with all other E3 class APs unchanged would yield only about 88.35% overall mAP50; sediment and joints also need improvement.',
    ]:
        additions.append(paragraph(text))
    additions.append(paragraph('Evidence files', 'Heading2'))
    for path in [NOTEBOOK, e5path, LEGACY / 'yolo26s_v5_cctv/results.csv',
                 LEGACY / 'yolo26s_v5_parallel_simam_cbam_se/results.csv',
                 RUNS / 'E6_E3_single_simam_v5_normal_seed0_20260928_225358_f163f4/results.csv']:
        additions.append(paragraph(path.relative_to(ROOT).as_posix()))
    section = body.find(f'{{{W}}}sectPr')
    index = list(body).index(section) if section is not None else len(body)
    page_break = paragraph('')
    page_break.find(f'{{{W}}}r').append(element('br', {'type': 'page'}))
    for offset, node in enumerate([page_break, *additions]):
        body.insert(index + offset, node)
    contents['word/document.xml'] = ET.tostring(document, encoding='utf-8', xml_declaration=True)
    # The original minimal package omitted this relationship, so connect its existing styles.
    relations = ET.fromstring(contents['word/_rels/document.xml.rels'])
    style_type = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles'
    if not any(child.get('Type') == style_type for child in relations):
        ET.SubElement(relations, f'{{{REL}}}Relationship', {'Id': 'rIdExperimentStyles', 'Type': style_type, 'Target': 'styles.xml'})
    contents['word/_rels/document.xml.rels'] = ET.tostring(relations, encoding='utf-8', xml_declaration=True)
    backup = REPORT.with_name(REPORT.stem + '_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.docx')
    shutil.copy2(REPORT, backup)
    fd, tmp = tempfile.mkstemp(suffix='.docx', dir=REPORT.parent)
    os.close(fd)
    try:
        with ZipFile(tmp, 'w', ZIP_DEFLATED) as archive:
            for info, _ in entries:
                archive.writestr(info, contents[info.filename])
        with ZipFile(tmp) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                if name.endswith(('.xml', '.rels')):
                    ET.fromstring(archive.read(name))
        os.replace(tmp, REPORT)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()
    print(f'Updated: {REPORT}\nBackup: {backup}')


if __name__ == '__main__':
    main()
