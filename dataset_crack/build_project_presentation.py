"""Rebuild the project PDF as an editable report and a concise research deck.

Run with the system Python 3.13 environment (python-docx, python-pptx, PyMuPDF,
Pillow and matplotlib are required). This script never changes the source PDF.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches as DInches, Pt as DPt, RGBColor as DRGB
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "BtechProject_Report.pdf"
OUT_DOC = ROOT / "BtechProject_Report_updated.docx"
SOURCE_DOC = ROOT / "BtechProject_Report_converted.docx"
OUT_PPT = ROOT / "RoadEye_Crack_Detection_Presentation.pptx"
MODEL_DOC = ROOT / "BFY_ModelComp2_report_updated.docx"
ASSETS = ROOT / "dataset_crack" / "presentation_assets"
ASSETS.mkdir(exist_ok=True)

RED = RGBColor(155, 27, 42)
INK = RGBColor(38, 42, 48)
MUTED = RGBColor(91, 98, 106)
LIGHT = RGBColor(239, 240, 242)
WHITE = RGBColor(255, 255, 255)

LITERATURE = [
    ("Huang et al., Processes 2023", "YOLOv5s + involution/GSConv/CBAM/distillation", "80.5", "48.7", "Six classes; 75 FPS reported", "https://doi.org/10.3390/pr11082508"),
    ("Sha et al., PLOS ONE 2025", "Lightweight YOLOv8 cross-scale fusion", "83.6", "—", "Different dataset; 1.95 M parameters", "https://doi.org/10.1371/journal.pone.0330677"),
    ("Hu et al., Sustainability 2026", "SAW-YOLOv8l", "86.2", "69.0", "Mixed Sewer-ML/industrial images", "https://doi.org/10.3390/su18083685"),
    ("Automatic Detection, Appl. Sci. 2023", "YOLOv4 + SPP + DIoU", "92.3", "—", "2,700 images; four classes", "https://doi.org/10.3390/app13074589"),
]

MODELS = [
    ("YOLO26n v4 + CLAHE", "v4 CCTV", 25.86, 13.38, "validation"),
    ("YOLO26s baseline", "v5 normal", 82.54, 65.61, "validation"),
    ("YOLO26s + SimAM", "v5 CLAHE", 82.45, 64.47, "validation"),
    ("YOLO26s + CBAM", "v5 CLAHE", 82.66, 63.50, "validation"),
    ("YOLO26s + SE", "v5 CLAHE", 82.24, 62.82, "validation"),
    ("Parallel SimAM+CBAM+SE", "v5 normal", 82.13, 63.88, "validation"),
    ("E3 P2 + weighted PAN", "v5 normal", 82.19, 65.01, "validation"),
    ("E5 directional P3", "v5 normal", 82.01, 63.82, "validation"),
    ("RF-DETR Nano 40e", "v5 normal", 89.56, 74.84, "validation"),
    ("RF-DETR Nano 50e", "v5 normal", 90.65, 76.51, "validation"),
    ("YOLO26s v6 initial", "v6 filtered", 82.54, 65.61, "validation"),
    ("YOLO26s v6 hard-image removal", "v6 filtered", 65.61, 52.63, "validation"),
]

TOP_TEST = [
    ("RF-DETR Nano, 50e", "normal / 512", 86.34, 83.89, 89.04, 73.85),
    ("YOLO26s baseline, 80e", "normal / 640", 81.44, 74.42, 82.14, 65.05),
    ("YOLO26s E3, 80e", "normal / 640", 78.40, 74.66, 80.78, 63.83),
    ("YOLO26s + CBAM, 80e", "CLAHE / 640", 78.50, 74.24, 81.55, 62.50),
]

CLASS_EXAMPLES = [
    ("crack", "sewage_000004.jpg"),
    ("corrosion_rust", "sewer_o8use_000033.jpg"),
    ("sediment_deposit", "sewage_defect_detection_000573.jpg"),
    ("root_intrusion", "sewer_o8use_000154.jpg"),
    ("joint_defect", "sewage_defect_detection_000132.jpg"),
]


def extract_pdf_image(doc: fitz.Document, page_index: int, image_index: int, name: str) -> Path:
    xref = doc[page_index].get_images(full=True)[image_index][0]
    item = doc.extract_image(xref)
    out = ASSETS / f"{name}.{item['ext']}"
    out.write_bytes(item["image"])
    return out


def annotated_examples() -> list[tuple[str, Path, str]]:
    result = []
    src = ROOT / "dataset_crack" / "final_dataset_v5" / "test"
    for label, name in CLASS_EXAMPLES:
        image_path = src / "images" / name
        label_path = src / "labels" / f"{Path(name).stem}.txt"
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing held-out example: {image_path} / {label_path}")
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        wanted = {"crack": 0, "corrosion_rust": 1, "sediment_deposit": 2, "root_intrusion": 3, "joint_defect": 4}[label]
        count = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5 or int(float(parts[0])) != wanted:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2) * img.width
            y1 = (yc - bh / 2) * img.height
            x2 = (xc + bw / 2) * img.width
            y2 = (yc + bh / 2) * img.height
            draw.rectangle((x1, y1, x2, y2), outline=(220, 40, 56), width=max(3, img.width // 180))
            count += 1
        out = ASSETS / f"example_{label}.jpg"
        img.save(out, quality=94)
        result.append((label, out, f"{name} · {count} ground-truth box{'es' if count != 1 else ''}"))
    return result


def make_chart() -> Path:
    labels = [m[0] for m in MODELS]
    ap50 = [m[2] for m in MODELS]
    ap95 = [m[3] for m in MODELS]
    colors = ["#b7bcc2" if "v4" in m[1] or "v6" in m[1] else ("#9b1b2a" if "RF-DETR" in m[0] else "#50606f") for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    yy = range(len(labels))
    for ax, values, title in zip(axes, (ap50, ap95), ("Validation mAP@0.50", "Validation mAP@0.50:0.95")):
        ax.barh(list(yy), values, color=colors, height=0.65)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=13, color="#262a30")
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9, colors="#4c5660")
        for i, value in enumerate(values):
            ax.text(value + 1, i, f"{value:.1f}", va="center", fontsize=8, color="#30363c")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(list(yy), labels=labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False)
    fig.subplots_adjust(left=.24, right=.97, bottom=.12, top=.91, wspace=.13)
    out = ASSETS / "all_completed_models_validation.png"
    fig.savefig(out, dpi=210, facecolor="white")
    plt.close(fig)
    return out


def add_doc_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = str(text)
    for row in rows:
        for cell, text in zip(table.add_row().cells, row):
            cell.text = str(text)


def build_source_transcription() -> None:
    """Archive every PDF page as editable text with its original raster figures."""
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = DInches(.65)
    sec.left_margin = sec.right_margin = DInches(.8)
    doc.styles["Normal"].font.name = "Aptos"
    doc.styles["Normal"].font.size = DPt(10)
    with fitz.open(PDF) as pdf:
        for page_index, source_page in enumerate(pdf):
            if page_index:
                doc.add_page_break()
            for block in source_page.get_text("blocks", sort=True):
                raw = block[4]
                clean = re.sub(r"\s+", " ", raw).strip()
                if not clean or clean == str(page_index):
                    continue
                if clean.startswith("Chapter ") or clean in ("Abstract", "References", "Appendix A"):
                    doc.add_heading(clean, level=1)
                elif re.match(r"^\d+\.\d+(?:\.\d+)?\s+[^.]+$", clean) and len(clean) < 100:
                    doc.add_heading(clean, level=2)
                else:
                    doc.add_paragraph(clean)
            for image_index, info in enumerate(source_page.get_images(full=True)):
                item = pdf.extract_image(info[0])
                path = ASSETS / f"source_p{page_index+1:02d}_{image_index}.{item['ext']}"
                path.write_bytes(item["image"])
                with Image.open(path) as im:
                    if im.width < 250 or im.height < 150:
                        width = DInches(1.2)
                    else:
                        width = DInches(min(5.6, max(3.2, im.width / 240)))
                doc.add_picture(str(path), width=width)
    doc.save(SOURCE_DOC)


def build_word(logo: Path, robot: Path, architecture: Path, material: Path, examples: list[tuple[str, Path, str]]) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = DInches(.72)
    section.bottom_margin = DInches(.72)
    section.left_margin = DInches(.82)
    section.right_margin = DInches(.82)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = DPt(10.5)
    styles["Normal"].paragraph_format.space_after = DPt(6)
    for s in ("Title", "Heading 1", "Heading 2"):
        styles[s].font.name = "Aptos Display"
        styles[s].font.color.rgb = DRGB(38, 42, 48)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Pipeline Inspection Mobile Robot for AI-enabled Crowd Management and Smart City Infrastructure Monitoring System").bold = True
    p.runs[0].font.size = DPt(20)
    p = doc.add_paragraph("Updated research report · sewer defect detection", style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(logo), width=DInches(1.15))
    p = doc.add_paragraph("B.Tech · Robotics and Automation / Artificial Intelligence and Machine Learning")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Aryan Sil · Ashmit Katale · Priyansh Pacheriwala · Parth Ubhe · Rajat Dwivedi")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Department of Robotics and Automation\nSymbiosis Institute of Technology, Pune · 2026–27")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("This project combines a mobile pipeline-inspection prototype with five-class visual defect detection for crack, corrosion/rust, sediment deposit, root intrusion and joint defect. The original proposal described YOLO11 experiments; those historical checkpoints and exact numerical logs are unavailable locally, so they are recorded as archived work rather than assigned invented scores. The reproducible current comparison uses YOLO26s variants and RF-DETR Nano on the v5 CCTV-focused dataset (9,486 train / 2,014 validation / 1,986 held-out test images). RF-DETR Nano after 50 total epochs reached 90.65% validation mAP@0.50 and 89.04% held-out test mAP@0.50, with 73.85% test mAP@0.50:0.95. The prototype photo documents chassis construction; the end-to-end robotic field performance and live-stream FPS still require formal measurement.")
    doc.add_heading("1. Introduction and motivation", level=1)
    doc.add_paragraph("Underground sewer inspection is often manual, slow and hazardous. Defects can be difficult to distinguish in low light, blur and textured pipe backgrounds. The project aims to combine safe remote viewing, a low-cost mobile platform, obstacle awareness and automated defect localization into one inspection workflow. Its smart-city motivation is faster maintenance triage; crowd-management language in the original proposal is broader context, not a measured function of this detector.")
    doc.add_paragraph("The proposal considers compact field robotics, wireless video, AI-assisted inspection and automated reports. Claimed advantages such as reduced inspection time and autonomous operation are design goals until measured on the assembled system.")
    doc.add_heading("2. Literature review", level=1)
    doc.add_paragraph("Published studies use different datasets, classes, splits and evaluation protocols. The figures below provide context, not a like-for-like leaderboard. Our 89.04% result is a held-out v5 test mAP@0.50; the 92.3% YOLOv4 paper reports a higher result on its own four-class dataset.")
    add_doc_table(doc, ["Study", "Approach", "mAP50", "mAP50:95", "Protocol note"], [[a, b, c + "%", d + "%" if d != "—" else d, e] for a,b,c,d,e,_ in LITERATURE])
    for _,_,_,_,_,url in LITERATURE:
        doc.add_paragraph(url, style="Caption")
    doc.add_heading("3. Problem, objectives and scope", level=1)
    doc.add_paragraph("Research questions: Can a compact robot obtain useful sewer POV footage? Can five frequent defects be detected robustly in that footage? How should attention and contrast enhancement be evaluated without split leakage? How well does the detector generalize to held-out pipe imagery? The objectives are to assemble a mobile platform, capture video, train and compare defect detectors, and expose inference through a usable interface.")
    doc.add_paragraph("Scope: four-wheel chassis with geared motors, controller, ultrasonic obstacle sensing and an ESP32-CAM-style video path; server-side image/video/live inference; five-class defect localization. Constraints include narrow-pipe mobility, lighting, blur, wireless attenuation, battery endurance and the possibility that images from the same inspection are correlated. GPS/IMU/report-generation elements from the initial block diagram remain proposed unless separately documented as implemented.")
    doc.add_heading("4. System and methodology", level=1)
    doc.add_paragraph("The source proposal combines locomotion, sensing, video transmission, AI inference, user display and result storage. The current software server accepts images, uploaded videos and live streams, with model selection; this is not proof of real-time robot deployment. The original system block diagram is retained as a proposed architecture.")
    doc.add_picture(str(architecture), width=DInches(4.0))
    doc.add_heading("4.1 Prototype and power budget", level=2)
    doc.add_paragraph("The photographed prototype uses a four-wheel chassis. The initial materials plan lists four 12 V, 100 RPM geared motors, motor driver, controller, ultrasonic sensing, ESP32-CAM/video components and an 11.1 V, 5,000 mAh battery. The original calculation uses 27 kg·cm ≈ 2.65 N·m per motor and 10.47 rad/s at 100 RPM, giving approximately 27.7 W mechanical per motor and 110.8 W for four. At an assumed 80% efficiency, estimated electrical demand is 138.5 W or 12.48 A at 11.1 V. These are assumptions, not measured current draw or runtime.")
    doc.add_picture(str(robot), width=DInches(5.0))
    doc.add_paragraph("Prototype photo extracted from the source report (same robot shown in the supplied image).", style="Caption")
    doc.add_picture(str(material), width=DInches(5.5))
    doc.add_paragraph("Original estimated expenditure: ₹13,750 (parts table from the source report).", style="Caption")
    doc.add_heading("4.2 Dataset construction", level=2)
    doc.add_paragraph("The v5 dataset merges sewer_o8use, sewage, sewage_defect_detection and sewer_defects into five labels. Unsupported categories were not forced into another class. Close-up pipeline_crack_detection and pipe_inspection imagery was excluded from the CCTV POV benchmark because it is visually a different domain. V2-style source/group-aware splitting was restored for v5 to reduce near-duplicate frames from an inspection appearing across train and test. This reduces leakage risk but does not certify perfect pipe-identity independence. sewer_o8use was oversampled four times in training only; validation and test were not oversampled. Counts: 9,486 / 2,014 / 1,986.")
    add_doc_table(doc, ["Class", "Held-out test example"], [[label, name] for label,name in CLASS_EXAMPLES])
    for label, path, caption in examples:
        doc.add_picture(str(path), width=DInches(3.5))
        doc.add_paragraph(f"{label}: {caption}. Red box = ground-truth label, not model prediction.", style="Caption")
    doc.add_heading("4.3 Preprocessing and training", level=2)
    doc.add_paragraph("Normal v5 is compared with a matched offline CLAHE variant. CLAHE operates on CIE-Lab luminance with clip limit 2.0 and an 8×8 tile grid; JPEG quality is 95. The underlying split and label coordinates are unchanged. YOLO26s variants use 640-pixel input, 80 epochs, AdamW, cosine learning-rate schedule, batch 16, seed 0 and standard image-space augmentation (HSV, translation, scaling, flip and mosaic, disabled in the final ten epochs). RF-DETR Nano uses 512-pixel input and 50 total epochs with effective batch 16 by gradient accumulation. Comparison across model families is indicative because input sizes and evaluators differ.")
    doc.add_paragraph("Architecture experiments include CBAM, SimAM, squeeze-and-excitation (SE), their parallel combination, E3 (a P2 head plus weighted PAN fusion), E5 (directional residual P3 block) and an incomplete E6 run that stopped with GPU out-of-memory. SPPF and C2PSA were inherited components, not novel inventions; TTA was described in the old proposal but is not counted as a verified improvement here.")
    doc.add_heading("5. Experimental results", level=1)
    doc.add_paragraph("The table reports completed validation runs. Historical YOLO11 baseline/CBAM/SimAM/combined variants appear in the original proposal, but their local checkpoints and exact metrics are unavailable. They are deliberately excluded from quantitative rankings.")
    add_doc_table(doc, ["Model", "Dataset", "Val mAP50", "Val mAP50:95"], [[a,b,f"{c:.2f}%",f"{d:.2f}%"] for a,b,c,d,_ in MODELS])
    doc.add_heading("5.1 Held-out test comparison", level=2)
    add_doc_table(doc, ["Model", "Pixels / input", "P", "R", "Test mAP50", "Test mAP50:95"], [[a,b,"—" if c is None else f"{c:.2f}%","—" if d is None else f"{d:.2f}%",f"{e:.2f}%",f"{f:.2f}%"] for a,b,c,d,e,f in TOP_TEST])
    doc.add_paragraph("RF-DETR test precision/recall are 86.34%/83.89%; 90.65% is its validation mAP50, not test accuracy. The best YOLO baseline test mAP50 is 82.14%, a 6.90-point gap on this test split. The difference is not solely attributable to architecture because the input resolution and evaluation backends differ.")
    doc.add_heading("5.2 Class-wise analysis and negative experiments", level=2)
    add_doc_table(doc, ["Class", "RF-DETR val AP50:95 (epoch 50)"], [[a,b] for a,b in [("crack","53.07%"),("corrosion_rust","88.91%"),("sediment_deposit","70.79%"),("root_intrusion","91.83%"),("joint_defect","77.52%")]])
    doc.add_paragraph("Crack is the weakest class. Thin boundaries, ambiguous texture, blur, poor contrast and inconsistent box extent are plausible error sources that need image-level verification. Removing model-selected hard examples from v6 did not improve the scientifically meaningful benchmark: one filtered validation run fell to 65.61% mAP50; a highly inflated 98.37% result came from a model-error-selected test subset and must not be presented as held-out generalization.")
    doc.add_heading("6. Integration, limitations and next work", level=1)
    doc.add_paragraph("A FastAPI inference interface was built for still images, uploaded video and camera/live stream inputs, with selectable trained models. End-to-end robot field trials, latency/FPS measurements, power draw, battery runtime, navigation reliability and fully independent inspection-site validation remain to be documented. The original seven-week plan covers literature review, design, dataset/model development, integration, testing and reporting; those are planning milestones, not evidence that every feature was completed.")
    doc.add_paragraph("Next steps: freeze the held-out test split; audit cracks and annotations; measure FPS on the target PC and robot video path; repeat leading models across seeds; test an external sewer inspection source; collect robot-native imagery; and report failure cases as well as mAP. Do not optimize on the test set or delete false negatives solely to increase a metric.")
    doc.add_heading("References", level=1)
    for a,b,c,d,e,url in LITERATURE:
        doc.add_paragraph(f"{a}. {url}")
    doc.add_paragraph("Original project proposal: BtechProject_Report.pdf, supplied by the team. Its earlier bibliography and YOLO11 plots are historical source material; individual entries and numerical YOLO11 runs require verification before publication.")
    doc.add_heading("Appendix: evidence and status", level=1)
    doc.add_paragraph("RF-DETR 50e: dataset_crack/runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/{best_validation_metrics.json,full_v5_test_metrics.json}. YOLO test values: dataset_crack/models/README.md and associated test folders. The original PDF has archived YOLO11 training curves but no recoverable checkpoint-level metric table. E1/E2/E4 are pending; E6 is incomplete due to OOM.")
    doc.save(OUT_DOC)


def update_modelcomp() -> Path:
    backup = ROOT / "BFY_ModelComp2_report_before_YOLO11_update.docx"
    if not backup.exists():
        shutil.copy2(MODEL_DOC, backup)
    doc = Document(MODEL_DOC)
    replacements = {
        "No RF-DETR test-set result was found in this run (run_test=false).": "The held-out v5 test evaluation was subsequently completed: 89.04% mAP50, 73.85% mAP50:95, 86.34% precision and 83.89% recall.",
        "The YOLO11n rows are illustrative placeholders, not actual runs.": "The source proposal mentions historical YOLO11 baseline, CBAM, SimAM and combined runs; exact checkpoints/logs are unavailable locally. No illustrative numerical scores are retained.",
        "5. Hypothetical YOLO11n placeholders": "5. Historical YOLO11 variants (metrics unavailable)",
        "The following values are illustrative placeholders only. YOLO11n artifacts were deleted and these runs were not recovered. They must not be cited as measured experimental results.": "The source PDF contains YOLO11 curves and names four YOLO11 variants. Their exact runs and weights are unavailable locally, so no numerical values are assigned and they are excluded from quantitative comparison.",
        "Reporting note: replace the hypothetical rows with N/A or recovered logs before submitting a paper or claiming experimental evidence.": "Reporting note: report these variants as historical only until original logs, weights and evaluation protocol are recovered.",
        "Re-evaluate both checkpoints on the identical untouched test protocol before making a paper claim.": "Both checkpoints have held-out v5 test evaluations, but cross-family input resolution and evaluator settings still differ; the 6.90-point test mAP50 gap is indicative, not architecture-only evidence.",
        "run RF-DETR once on the untouched v5 test split, log per-class AP and the exact evaluation settings": "use the completed RF-DETR evaluation on the untouched v5 test split, log per-class test AP and the exact evaluation settings",
        "No RF-DETR test-set result was found": "RF-DETR test results are now available",
    }
    for p in doc.paragraphs:
        if not p.text:
            continue
        updated = p.text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != p.text:
            p.text = updated
    table = doc.tables[7]
    rows = [
        ["YOLO11 baseline", "historical project dataset", "N/A", "N/A", "N/A", "N/A", "Archived curves in source PDF; checkpoint/log unavailable"],
        ["YOLO11 + CBAM", "historical project dataset", "N/A", "N/A", "N/A", "N/A", "Mentioned in source proposal; no recoverable metrics"],
        ["YOLO11 + SimAM", "historical project dataset", "N/A", "N/A", "N/A", "N/A", "Mentioned in source proposal; no recoverable metrics"],
        ["YOLO11 + CBAM + SimAM", "historical project dataset", "N/A", "N/A", "N/A", "N/A", "Mentioned in source proposal; no recoverable metrics"],
    ]
    table.rows[0].cells[-1].text = "Evidence / status"
    for row_index, values in enumerate(rows, start=1):
        row = table.rows[row_index] if row_index < len(table.rows) else table.add_row()
        for cell, value in zip(row.cells, values):
            cell.text = value
    lit = doc.tables[6]
    for row in lit.rows[1:]:
        row.cells[2].text = "RF-DETR: 89.04% held-out test mAP50 on v5; different data/protocol, so not a direct win claim."
    if not any(p.text == "October 2026 held-out test update" for p in doc.paragraphs):
        doc.add_heading("October 2026 held-out test update", level=1)
        doc.add_paragraph("RF-DETR Nano 50e reached 89.04% test mAP50 and 73.85% test mAP50:95 on the unchanged 1,986-image v5 test split. The YOLO26s baseline reached 82.14% / 65.05%; E3 reached 80.78% / 63.83%. Historical YOLO11 variants have no recoverable numerical results and are marked N/A, replacing the previous illustrative placeholders.")
    doc.save(MODEL_DOC)
    return backup


def tx(slide, text: str, x: float, y: float, w: float, h: float, size: int = 20,
       bold: bool = False, color=INK, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = "Aptos"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color


def page(slide, number: int) -> None:
    tx(slide, str(number), 12.25, 7.07, .40, .25, size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def title(slide, text: str, number: int) -> None:
    tx(slide, text, .72, .45, 11.9, .6, size=30, bold=True)
    page(slide, number)


def bullet(slide, lines: list[str], x=.82, y=1.45, w=11.6, h=5.4, size=22) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + line
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = INK
        p.space_after = Pt(16)


def table_slide(slide, headers: list[str], rows: list[list[str]], x=.65, y=1.38, w=12.0, h=4.8, font=15) -> None:
    sh = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = sh.table
    for j, label in enumerate(headers):
        table.cell(0,j).text = label
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            table.cell(i,j).text = value
    for i in range(len(rows)+1):
        for j in range(len(headers)):
            cell = table.cell(i,j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(249,249,249)
            cell.margin_left = Inches(.08)
            cell.margin_right = Inches(.06)
            cell.margin_top = Inches(.04)
            cell.margin_bottom = Inches(.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Aptos"
                p.font.size = Pt(font)
                p.font.color.rgb = INK
                p.font.bold = i == 0


def fit_image(slide, path: Path, x: float, y: float, w: float, h: float) -> None:
    im = Image.open(path)
    ratio = im.width / im.height
    box_ratio = w / h
    if ratio > box_ratio:
        actual_w, actual_h = w, w / ratio
    else:
        actual_h, actual_w = h, h * ratio
    slide.shapes.add_picture(str(path), Inches(x + (w-actual_w)/2), Inches(y + (h-actual_h)/2), width=Inches(actual_w), height=Inches(actual_h))


def build_ppt(logo: Path, robot: Path, architecture: Path, chart: Path, examples: list[tuple[str, Path, str]]) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    def new():
        return prs.slides.add_slide(blank)

    s = new()
    tx(s, "Pipeline Inspection Mobile Robot for AI-enabled Crowd Management and Smart City Infrastructure Monitoring System", 1.1, .78, 11.1, 1.35, size=27, bold=True, align=PP_ALIGN.CENTER)
    fit_image(s, logo, 5.7, 2.3, 1.9, 1.6)
    tx(s, "B.Tech project · Robotics and Automation / AI & ML", 1.8, 4.22, 9.7, .4, size=18, align=PP_ALIGN.CENTER)
    tx(s, "Aryan Sil  ·  Ashmit Katale  ·  Priyansh Pacheriwala  ·  Parth Ubhe  ·  Rajat Dwivedi", .8, 5.1, 11.7, .5, size=15, align=PP_ALIGN.CENTER)
    tx(s, "Symbiosis Institute of Technology, Pune  ·  2026–27", 2.3, 6.12, 8.8, .35, size=16, align=PP_ALIGN.CENTER)
    page(s,1)

    s = new(); title(s,"Why inspect sewers with a robot?",2)
    bullet(s,["Manual CCTV review is slow and exposes personnel to hazardous environments.","Low light, blur and repetitive pipe texture make subtle defects hard to localize.","The project joins a mobile camera platform, remote inference and a five-class detector."],size=23)

    s = new(); title(s,"Research questions and objectives",3)
    bullet(s,["Can a compact four-wheel platform obtain useful pipe-view footage?","Can five frequent defect types be localized reliably on unseen inspection images?","Do CLAHE or attention modules improve a controlled YOLO baseline?","Build the prototype, compare detectors and expose image/video/live inference."],size=21)

    s = new(); title(s,"What prior work reports",4)
    table_slide(s,["Study","Method","mAP@50","mAP@50:95","Context"],[[a.split(',')[0],b,c+"%",d+"%" if d!='—' else d,e] for a,b,c,d,e,_ in LITERATURE],font=12)
    tx(s,"Our RF-DETR: 89.04% test mAP@50, 73.85% test mAP@50:95 on v5. It exceeds some reported scores numerically, not in a matched benchmark; the YOLOv4 paper reports 92.3% on a different four-class set.",.75,6.35,11.7,.55,size=14,color=MUTED)

    s = new(); title(s,"System concept and current implementation",5)
    fit_image(s,architecture,.7,1.25,5.5,5.8)
    bullet(s,["Proposed: locomotion → sensing/video → wireless link → AI detection → display/report.","Implemented software: FastAPI image, video and live-camera inference with model selection.","Robot-to-server field FPS and autonomous navigation remain unverified."],x=6.35,y=1.5,w=6.0,h=4.7,size=17)

    s = new(); title(s,"Built prototype",6)
    fit_image(s,robot,.78,1.28,7.2,5.65)
    bullet(s,["Four-wheel chassis with geared-drive concept.","Initial design includes controller, ultrasonic sensing and camera stream.","Approximate materials budget: ₹13,750.","Battery/runtime values in the proposal are estimates, not measurements."],x=8.15,y=1.48,w=4.5,h=4.9,size=17)

    s = new(); title(s,"Power budget: design estimate",7)
    table_slide(s,["Quantity","Source assumption","Estimated value"],[
        ["Drive","4 × 12 V, 100 RPM geared motors","27 kg·cm ≈ 2.65 N·m each"],
        ["Speed","ω = 2πN/60","10.47 rad/s"],
        ["Mechanical","P = τω","27.7 W/motor; 110.8 W total"],
        ["Electrical","80% efficiency assumed","138.5 W ≈ 12.48 A at 11.1 V"],
        ["Battery","11.1 V, 5,000 mAh, 3C","15 A nominal continuous rating"],
    ],font=16)
    tx(s,"Actual current draw, thermal behavior and runtime still require bench and field measurements.",.74,6.55,11.5,.4,size=15,color=MUTED)

    s = new(); title(s,"Dataset evolution: v2 → v5",8)
    table_slide(s,["Version","Decision","Research effect"],[
        ["v2","Common five-class taxonomy; group-aware split","Reduced adjacent-frame leakage risk"],
        ["v3","Broader CCTV + close-up mixture","Exposed domain mismatch"],
        ["v4","Separated CCTV and close-up; tested CLAHE","Isolated viewpoint/contrast effects"],
        ["v5","Restored v2-style group split; excluded close-up domains","CCTV-focused 9,486 / 2,014 / 1,986 split"],
    ],font=16)
    tx(s,"Sources retained: sewer_o8use, sewage, sewage_defect_detection, sewer_defects. Train-only 4× sewer_o8use oversampling; no test oversampling.",.73,6.35,11.8,.42,size=15,color=MUTED)

    s = new(); title(s,"Five detection classes · held-out test images",9)
    positions=[(.72,1.5),(5.05,1.5),(9.38,1.5),(2.89,4.0),(7.22,4.0)]
    for (label,path,caption),(x,y) in zip(examples,positions):
        fit_image(s,path,x,y,3.2,1.63)
        tx(s,label.replace('_',' '),x,y+1.68,3.2,.28,size=15,bold=True,align=PP_ALIGN.CENTER)
    tx(s,"Red boxes are ground truth, not predictions. One test image per class.",.75,6.8,11.8,.28,size=13,color=MUTED)

    s = new(); title(s,"Preprocessing and experimental controls",10)
    bullet(s,["Normal v5 and offline CLAHE v5 share image/label identities; CLAHE adjusts Lab luminance (clip 2.0, 8×8 grid).","YOLO26s: 640 px, 80 epochs, AdamW, cosine schedule, seed 0; HSV/translate/scale/flip/mosaic, final ten epochs without mosaic.","RF-DETR Nano: 512 px, 50 total epochs, effective batch 16 with accumulation.","Held-out test remains unchanged; model-selected deletion is not a valid benchmark."],size=19)

    s = new(); title(s,"Models actually evaluated",11)
    bullet(s,["YOLO26s baseline, SimAM, CBAM, SE, parallel SimAM+CBAM+SE.","E3: P2 head + weighted PAN fusion. E5: directional residual P3 block.","RF-DETR Nano: 40 and 50 total epochs. E6 ended at epoch 5 with OOM.","Original YOLO11 baseline/CBAM/SimAM/combined curves are historical; checkpoints and exact metrics are unavailable, so no scores are fabricated."],size=18)

    s = new(); title(s,"Completed model comparison · validation",12)
    fit_image(s,chart,.52,1.15,12.25,5.52)
    tx(s,"Grey: different v4/v6 data; descriptive only. YOLO11 and incomplete E6 have no numerical bar.",.75,6.76,11.8,.23,size=11,color=MUTED)

    s = new(); title(s,"Top models · unchanged v5 test set",13)
    table_slide(s,["Model","Input","Precision","Recall","mAP@50","mAP@50:95"],[[a,b,"—" if c is None else f"{c:.2f}%","—" if d is None else f"{d:.2f}%",f"{e:.2f}%",f"{f:.2f}%"] for a,b,c,d,e,f in TOP_TEST],font=16)
    tx(s,"RF-DETR reaches 90.65% on validation, but 89.04% on held-out test. Cross-family evaluator/input differences remain.",.77,6.4,11.5,.45,size=15,color=MUTED)

    s = new(); title(s,"What still limits accuracy?",14)
    table_slide(s,["Class","RF-DETR val AP@50:95","Reading"],[
        ["crack","53.07%","Main localization and miss problem"],
        ["corrosion/rust","88.91%","Strong"],
        ["sediment deposit","70.79%","Moderate"],
        ["root intrusion","91.83%","Strong"],
        ["joint defect","77.52%","Moderate"],
    ],font=17)
    tx(s,"Likely issues to audit: thin/ambiguous cracks, blur, pipe texture false positives and box consistency.",.75,6.55,11.6,.35,size=15,color=MUTED)

    s = new(); title(s,"Negative results and research integrity",15)
    bullet(s,["Attention variants and E3 were near or below the plain YOLO26s held-out baseline.","A v6 subset selected from prior model errors gave an inflated 98.37% mAP@50: it is not a valid untouched test result.","After removing hard training examples, validation fell to 65.61% mAP@50.","Do not delete false negatives or tune on test images merely to raise the reported number."],size=20)

    s = new(); title(s,"Scope, limitations and next steps",16)
    bullet(s,["Measure robot-camera latency, real FPS, power draw, endurance and navigation in actual pipe conditions.","Audit crack boxes and failure cases; collect robot-native footage and an external site-level test set.","Repeat the leading models across seeds; compare under harmonized inference settings.","Keep the test split frozen, and report per-class AP plus failures alongside headline mAP."],size=20)

    s = new(); title(s,"Original seven-week work plan",17)
    table_slide(s,["Weeks","Planned work","Current evidence"],[
        ["1","Literature and requirements","Reviewed; model comparison updated"],
        ["2–3","Hardware selection, chassis and circuits","Prototype photographed; bench measurements pending"],
        ["4","Control firmware and ESP32-CAM stream","Integration requires documented field test"],
        ["5","YOLO11 variants, CLAHE and TTA","Archived YOLO11 plots; current YOLO26/RF-DETR results verified"],
        ["6–7","System integration, FPS and reporting","FastAPI interface built; end-to-end robot FPS pending"],
    ],font=15)
    tx(s,"The schedule in the source proposal is a plan, not evidence that each hardware/software milestone was completed.",.74,6.55,11.7,.42,size=14,color=MUTED)

    s = new(); title(s,"Takeaway",18)
    tx(s,"89.04%",.8,1.5,4.7,1.2,size=54,bold=True,color=RED)
    tx(s,"RF-DETR Nano held-out test mAP@50",.85,2.75,6.2,.6,size=22)
    tx(s,"Five-class CCTV-focused benchmark · robot prototype · deployable inference interface",.83,4.05,11.7,1.0,size=27,bold=True)
    tx(s,"Next milestone: prove the complete robot-to-detection loop and generalization on new inspections.",.85,5.83,11.6,.6,size=18,color=MUTED)

    s = new(); title(s,"References and evidence",19)
    for i,(a,b,c,d,e,url) in enumerate(LITERATURE):
        tx(s,f"{a} — {url}",.75,1.35+i*.78,11.9,.45,size=14)
    tx(s,"Project evidence: source PDF; dataset_crack/models/README.md; RF-DETR full_v5_test_metrics.json and best_validation_metrics.json; YOLO test run outputs.",.75,5.0,11.8,1.0,size=14,color=MUTED)
    tx(s,"Metric comparison across publications is contextual only: datasets, taxonomies, splits and evaluators differ.",.75,6.25,11.8,.45,size=14,color=MUTED)
    prs.save(OUT_PPT)


def main() -> None:
    with fitz.open(PDF) as pdf:
        logo = extract_pdf_image(pdf,0,0,"symbiosis_logo")
        architecture = extract_pdf_image(pdf,9,0,"proposed_system_architecture")
        material = extract_pdf_image(pdf,10,0,"materials_budget")
        robot = extract_pdf_image(pdf,12,0,"robot_prototype")
    examples = annotated_examples()
    chart = make_chart()
    build_source_transcription()
    build_word(logo,robot,architecture,material,examples)
    backup = update_modelcomp()
    build_ppt(logo,robot,architecture,chart,examples)
    print(json.dumps({"source_conversion":str(SOURCE_DOC),"report":str(OUT_DOC),"presentation":str(OUT_PPT),"modelcomp":str(MODEL_DOC),"modelcomp_backup":str(backup),"slides":19},indent=2))


if __name__ == "__main__":
    main()
