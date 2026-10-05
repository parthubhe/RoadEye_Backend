"""Append verified dataset documentation and clearly labelled illustrative rows."""

from pathlib import Path
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET

SRC = Path('BFY_ModelComp.xlsx')
TMP = Path('BFY_ModelComp_updated.xlsx')
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
ET.register_namespace('', NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')


def text_cell(ref: str, value: str) -> ET.Element:
    cell = ET.Element(f'{{{NS}}}c', {'r': ref, 't': 'inlineStr'})
    inline = ET.SubElement(cell, f'{{{NS}}}is')
    node = ET.SubElement(inline, f'{{{NS}}}t')
    node.text = value
    return cell


def number_cell(ref: str, value: str) -> ET.Element:
    cell = ET.Element(f'{{{NS}}}c', {'r': ref})
    node = ET.SubElement(cell, f'{{{NS}}}v')
    node.text = value
    return cell


def row(number: int, values: list[tuple[str, str, bool]]) -> ET.Element:
    item = ET.Element(f'{{{NS}}}row', {'r': str(number)})
    for col, value, is_number in values:
        item.append(number_cell(f'{col}{number}', value) if is_number else text_cell(f'{col}{number}', value))
    return item


def main() -> None:
    with zipfile.ZipFile(SRC, 'r') as source:
        sheet = ET.fromstring(source.read('xl/worksheets/sheet1.xml'))
        sheet_data = sheet.find(f'{{{NS}}}sheetData')
        if sheet_data is None:
            raise RuntimeError('Worksheet has no sheetData')
        # Keep the workbook non-destructive: do not append the same sections twice.
        existing = ''.join(sheet.itertext())
        if 'Dataset evolution and split policy' in existing:
            raise RuntimeError('Documentation section already exists in workbook')
        entries = [
            row(27, [('A', 'Dataset evolution and split policy (verified process)', False)]),
            row(28, [('A', 'Version', False), ('B', 'Dataset processing', False), ('C', 'Split and leakage control', False), ('D', 'Status', False)]),
            row(29, [('A', 'v2', False), ('B', 'Merged sewer inspection sources into a common five-class taxonomy.', False), ('C', 'Group-level assignments kept related pipe sequences together; train/validation/test were separated by group to reduce same-pipe frame leakage.', False), ('D', 'Verified historical dataset policy', False)]),
            row(30, [('A', 'v3', False), ('B', 'Added broader CCTV and close-up material, exposing substantial domain and viewpoint variation.', False), ('C', 'Mixed domains made generalisation harder when a visual domain was over-represented in one split.', False), ('D', 'Verified project observation', False)]),
            row(31, [('A', 'v4', False), ('B', 'Separated CCTV pipe-view data from close-up/domain-mismatched material and tested CLAHE preprocessing.', False), ('C', 'The split was reorganised to make the evaluation domain explicit rather than mixing visually incompatible sources.', False), ('D', 'Verified project dataset build', False)]),
            row(32, [('A', 'v5', False), ('B', 'Returned to the v2-style group split and removed only identified close-up mismatch prefixes; retained crack, corrosion_rust, sediment_deposit, root_intrusion, and joint_defect.', False), ('C', 'Related frames from the same pipe/inspection group are kept in one split, preventing near-duplicate CCTV frames from entering the test set and inflating or destabilising evaluation.', False), ('D', 'Verified: 9,486 train / 2,014 validation / 1,986 test images', False)]),
            row(33, [('A', 'v5 variants', False), ('B', 'Normal and CLAHE versions use the same image/label split assignments; CLAHE is applied only to image intensity, not annotations.', False), ('C', 'This makes normal-versus-CLAHE comparisons paired and reproducible.', False), ('D', 'Verified preprocessing setup', False)]),
            row(35, [('A', 'Hypothetical / illustrative YOLO11n results — NOT RUN RESULTS', False)]),
            row(36, [('A', 'Model', False), ('B', 'Dataset', False), ('C', 'Precision', False), ('D', 'Recall', False), ('E', 'mAP50', False), ('F', 'mAP50-95', False), ('G', 'Use', False)]),
            row(37, [('A', 'YOLO11n', False), ('B', 'v2', False), ('C', '0.642', True), ('D', '0.588', True), ('E', '0.617', True), ('F', '0.341', True), ('G', 'Illustrative placeholder only; not measured', False)]),
            row(38, [('A', 'YOLO11n', False), ('B', 'v3', False), ('C', '0.601', True), ('D', '0.557', True), ('E', '0.584', True), ('F', '0.309', True), ('G', 'Illustrative placeholder only; not measured', False)]),
            row(40, [('A', 'Important reporting note', False), ('B', 'Rows 37–38 are hypothetical placeholders requested for planning and must not be cited as experimental evidence. Replace them with N/A or recovered logs before submitting a paper.', False)]),
        ]
        for item in entries:
            sheet_data.append(item)
        sheet_bytes = ET.tostring(sheet, encoding='utf-8', xml_declaration=True)
        with zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                target.writestr(info, sheet_bytes if info.filename == 'xl/worksheets/sheet1.xml' else source.read(info.filename))
    shutil.move(TMP, SRC)
    print(f'Updated {SRC}')


if __name__ == '__main__':
    main()
