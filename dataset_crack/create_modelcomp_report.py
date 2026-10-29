"""Create a clean Word report from the verified model-comparison records."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

OUT = Path("BFY_ModelComp_report_v2.docx")


def p(text: str, style: str = "Normal") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ['<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr>']
    for r, values in enumerate([headers] + rows):
        out.append('<w:tr>')
        for value in values:
            bold = '<w:b/>' if r == 0 else ''
            out.append(f'<w:tc><w:p><w:r>{bold}<w:t xml:space="preserve">{escape(str(value))}</w:t></w:r></w:p></w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


def document_xml() -> str:
    body = []
    body.append(p('RoadEye Sewer Defect Detection — Dataset and Model Comparison', 'Title'))
    body.append(p('Consolidated research record', 'Subtitle'))
    body.append(p('This report consolidates the previous comparison sheet and the v2-to-v5 dataset documentation. Measured values are listed separately from illustrative placeholders.'))

    body.append(p('1. Measured YOLO26 results', 'Heading1'))
    body.append(table(['Model', 'Dataset', 'Precision', 'Recall', 'mAP50', 'mAP50-95'], [
        ['YOLO26s baseline', 'v5 normal', '0.805', '0.761', '0.826', '0.655'],
        ['YOLO26s + SimAM', 'v5 CLAHE', '0.813', '0.747', '0.824', '0.644'],
        ['YOLO26s + CBAM', 'v5 CLAHE', '0.765', '0.781', '0.826', '0.635'],
        ['YOLO26s + SE', 'v5 CLAHE', '0.776', '0.763', '0.823', '0.628'],
    ]))
    body.append(p('Unseen test split comparison', 'Heading2'))
    body.append(table(['Model', 'Precision', 'Recall', 'mAP50', 'mAP50-95'], [
        ['Baseline', '0.8144', '0.7442', '0.8214', '0.6505'],
        ['SimAM', '0.7835', '0.7301', '0.8085', '0.6303'],
        ['CBAM', '0.7850', '0.7424', '0.8155', '0.6250'],
        ['SE', '0.7895', '0.7484', '0.8153', '0.6296'],
    ]))

    body.append(p('2. Dataset evolution and research methodology', 'Heading1'))
    body.append(table(['Version', 'Processing', 'Split and leakage control', 'Status'], [
        ['v2', 'Merged sewer inspection sources into a common five-class taxonomy.', 'Group-level assignments kept related pipe sequences together; train, validation, and test were separated by group.', 'Verified historical policy'],
        ['v3', 'Added broader CCTV and close-up material, increasing viewpoint and domain variation.', 'Mixed CCTV and close-up domains made generalisation more difficult.', 'Verified project observation'],
        ['v4', 'Separated CCTV pipe-view data from close-up/domain-mismatched material and tested CLAHE.', 'Evaluation domains were made explicit instead of mixing visually incompatible sources.', 'Verified dataset build'],
        ['v5', 'Restored the v2-style group split and removed only identified close-up mismatch prefixes.', 'Related frames from the same pipe/inspection group remain in one split, reducing near-duplicate CCTV leakage into test.', '9,486 train / 2,014 validation / 1,986 test'],
        ['v5 variants', 'Normal and CLAHE versions use identical image/label assignments; CLAHE changes image intensity only.', 'Paired split enables a controlled normal-versus-CLAHE comparison.', 'Verified preprocessing setup'],
    ]))
    body.append(p('Current five-class taxonomy: crack, corrosion_rust, sediment_deposit, root_intrusion, and joint_defect.'))
    body.append(p('The v5 construction was intended to ensure that a sequence of frames from one pipe inspection does not appear across train and test splits. This reduces leakage and gives a more realistic estimate of performance on unseen inspections.'))

    body.append(p('3. Architecture, losses, and computational profile', 'Heading1'))
    body.append(p('SPP (Spatial Pyramid Pooling) applies pooling at multiple receptive-field sizes and concatenates the resulting features. It gives the detector local and wider pipe-context information without changing the spatial output size. SPPF (Spatial Pyramid Pooling - Fast) is a computationally cheaper serial implementation that reuses one pooling operation repeatedly. Both are architectural modules, not attention mechanisms.'))
    body.append(p('DIoU (Distance-IoU) adds a centre-distance penalty to IoU loss, encouraging predicted and ground-truth box centres to align. CIoU (Complete-IoU) extends this with centre distance and aspect-ratio consistency. These losses are useful for localization, but they are not automatically enabled by changing a YAML file. In YOLO26, using them requires confirming the installed Ultralytics loss implementation and modifying or extending the training code; an SPP/SPPF block can be inserted through a compatible model YAML and custom module registration.'))
    body.append(table(['Model', 'Layers', 'Parameters', 'GFLOPs @ 640', 'Classes', 'Checkpoint / status'], [
        ['YOLO26s v5 baseline', '260', '9,951,734', '22.76', '5', '20.3 MB best.pt; trained 80 epochs'],
        ['YOLO26s parallel SimAM+CBAM+SE', '275', '10,021,857', '23.61', '5', 'Architecture measured; training run pending'],
    ]))
    body.append(p('The parallel attention design adds approximately 70,123 parameters (+0.7%) and 0.85 GFLOPs (+3.7%) relative to the measured YOLO26s baseline. The reported GFLOPs are model-construction estimates at 640 × 640, not a substitute for hardware FPS benchmarking.'))
    body.append(p('Baseline training configuration: 640 × 640 images, batch size 16, 80 epochs, AdamW optimizer, initial learning rate 0.002, cosine learning-rate schedule, deterministic seed 0, and six data-loader workers. The baseline completed in approximately 3.19 hours on the RTX 4070 Laptop GPU used for this project.'))

    body.append(p('4. Comparison with published work', 'Heading1'))
    body.append(table(['Published model', 'Reported result', 'Comparison'], [
        ['Improved YOLOv5s, Processes 2023', '80.5% mAP@0.5', 'Our 82.6% is approximately +2.1 points'],
        ['Improved YOLOv8n, PLOS One 2025', '83.6% mAP@0.5', 'Our 82.6% is approximately -1.0 point'],
        ['SAW-YOLOv8l, Sustainability 2026', '86.2% mAP@0.5; 69.0% mAP@0.5:0.95', 'Our result is lower; datasets and protocols differ'],
        ['Improved YOLOv4, Applied Sciences 2023', '92.3% reported mAP', 'Different four-class, 2,700-image dataset'],
        ['DefectTR transformer, 2022', '60.2% reported mAP', 'Datasets and evaluation protocols differ substantially'],
    ]))

    body.append(p('5. Hypothetical YOLO11n placeholders', 'Heading1'))
    body.append(p('The following values are illustrative placeholders only. YOLO11n artifacts were deleted and these runs were not recovered. They must not be cited as measured experimental results.'))
    body.append(table(['Model', 'Dataset', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'Use'], [
        ['YOLO11n', 'v2', '0.642', '0.588', '0.617', '0.341', 'Illustrative only; not measured'],
        ['YOLO11n', 'v3', '0.601', '0.557', '0.584', '0.309', 'Illustrative only; not measured'],
    ]))
    body.append(p('Reporting note: replace the hypothetical rows with N/A or recovered logs before submitting a paper or claiming experimental evidence.'))

    body.append(p('6. Interpretation and limitations', 'Heading1'))
    body.append(p('The v5 results are not directly comparable with papers that use different class taxonomies, image domains, split policies, or evaluation protocols. The group-level split is a methodological safeguard against near-duplicate leakage, not a guarantee of higher accuracy.'))
    body.append(p('CLAHE and attention mechanisms were evaluated as controlled variants. The best validation mAP50 is shared by the baseline and CBAM at approximately 0.826, while the unseen test split favours the baseline at 0.8214.'))

    body.append('<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>')
    return '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + ''.join(body) + '</w:body></w:document>'


CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'''
STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w: basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:i/><w:sz w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style></w:styles>'''.replace('<w: basedOn', '<w:basedOn')


def main() -> None:
    with ZipFile(OUT, 'w', ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CONTENT_TYPES)
        z.writestr('_rels/.rels', RELS)
        z.writestr('word/_rels/document.xml.rels', DOC_RELS)
        z.writestr('word/document.xml', document_xml())
        z.writestr('word/styles.xml', STYLES)
    print(f'Created {OUT.resolve()}')


if __name__ == '__main__':
    main()
