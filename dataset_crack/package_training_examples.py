"""Select clear, diverse v5 training examples and package their original images."""
from pathlib import Path
import csv
import shutil
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'final_dataset_v5' / 'train'
OUT = ROOT / 'training_examples_final_25'
NAMES = ['crack', 'corrosion_rust', 'sediment_deposit', 'root_intrusion', 'joint_defect']


def main():
    if (OUT.exists() and any(OUT.iterdir())) or OUT.with_suffix('.zip').exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    candidates = {i: [] for i in range(5)}
    for label in sorted((SOURCE / 'labels').glob('*.txt')):
        boxes = []
        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, x, y, w, h = map(float, p)
                if int(c) in candidates and min(w, h) >= .035 and .008 <= w*h <= .85:
                    boxes.append((int(c), x, y, w, h))
        if not boxes:
            continue
        image = next((SOURCE / 'images' / (label.stem + ext) for ext in ['.jpg', '.png', '.jpeg'] if (SOURCE / 'images' / (label.stem + ext)).exists()), None)
        if image is None:
            continue
        frame = cv2.imread(str(image), cv2.IMREAD_REDUCED_COLOR_4)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 320))
        sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
        exposure = float(np.mean((small > 20) & (small < 235)))
        if exposure < .55:
            continue
        H, W = gray.shape
        for c, x, y, w, h in boxes:
            crop = gray[max(0,int((y-h/2)*H)):min(H,int((y+h/2)*H)), max(0,int((x-w/2)*W)):min(W,int((x+w/2)*W))]
            if crop.size < 100:
                continue
            contrast = float(crop.std())
            score = np.log1p(sharp) + np.log1p(contrast) + 2*exposure + min(w*h,.3)*4 - .5*(len(boxes)-1)
            thumb = cv2.resize(gray,(24,24)).astype(np.float32)
            thumb = (thumb-thumb.mean()) / max(float(thumb.std()),1)
            candidates[c].append((float(score), image, label, (x,y,w,h),thumb))
    OUT.mkdir(exist_ok=True)
    selected, used = [], set()
    for c, name in enumerate(NAMES):
        count = 0
        sources = {}
        thumbs = []
        for score, image, label, box, thumb in sorted(candidates[c], key=lambda r: -r[0]):
            family = image.stem.rsplit('_',1)[0]
            if image.name in used:
                continue
            if any(float(np.mean((thumb-old)**2)) < .25 for old in thumbs):
                continue
            selected.append((c,image,label,box,score))
            thumbs.append(thumb)
            used.add(image.name)
            sources[family] = sources.get(family,0)+1
            count += 1
            if count == 5:
                break
        if count != 5:
            raise RuntimeError(f'Insufficient examples for {name}')
    sheet = Image.new('RGB',(1200,5*220),'white')
    draw = ImageDraw.Draw(sheet)
    for idx,(c,image,label,box,score) in enumerate(selected):
        folder = OUT / NAMES[c]
        folder.mkdir(exist_ok=True)
        shutil.copy2(image,folder/image.name)
        with Image.open(image) as im:
            preview = ImageOps.contain(im.convert('RGB'),(232,175))
        px,py=(idx%5)*240,(idx//5)*220
        sheet.paste(preview,(px,py+22))
        draw.text((px+4,py+3),NAMES[c],fill='black')
        draw.text((px+4,py+199),image.stem,fill='black')
    sheet.save(OUT / 'contact_sheet.jpg',quality=93)
    with (OUT/'image_sources.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['class','filename','original_training_path'])
        for c,image,*_ in selected:
            writer.writerow([NAMES[c],image.name,str(image)])
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(OUT.parent))
    print(f'Packaged {len(selected)} distinct training images: {OUT.with_suffix(".zip")}')
    print(f'Contact sheet: {OUT / "contact_sheet.jpg"}')


if __name__ == '__main__':
    main()
