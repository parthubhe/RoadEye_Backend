"""Create review sheets, then package explicitly visually selected examples."""
import csv
import json
import random
import shutil
import zipfile
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'final_dataset_v5/train'
REVIEW=ROOT/'dry_training_example_review'
NAMES=['crack','corrosion_rust','sediment_deposit','root_intrusion','joint_defect']

def package():
    chosen={'crack':'sewer_defects_001177','corrosion_rust':'sewer_o8use_009004',
            'sediment_deposit':'sewer_o8use_005414','root_intrusion':'sewer_o8use_002632',
            'joint_defect':'sewer_o8use_009487'}
    out=ROOT/'training_examples_5_no_visible_water'
    if (out.exists() or out.with_suffix('.zip').exists()) and '--replace' not in sys.argv:
        raise FileExistsError(out)
    for name,stem in chosen.items():
        classes={int(line.split()[0]) for line in (TRAIN/'labels'/f'{stem}.txt').read_text().splitlines() if line.strip()}
        assert NAMES.index(name) in classes,(name,stem)
        assert (TRAIN/'images'/f'{stem}.jpg').is_file()
    if out.exists():
        resolved=out.resolve()
        assert resolved.parent==ROOT.resolve() and resolved.name=='training_examples_5_no_visible_water'
        backup=REVIEW/f'previous_package_{datetime.now():%Y%m%d_%H%M%S}'
        backup.parent.mkdir(exist_ok=True)
        shutil.move(str(resolved),str(backup))
    out.mkdir()
    sheet=Image.new('RGB',(1500,1000),'white');draw=ImageDraw.Draw(sheet)
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    from PIL import ImageFont
    font=ImageFont.truetype(str(fontpath),22) if fontpath.exists() else ImageFont.load_default()
    rows=[]
    for i,(name,stem) in enumerate(chosen.items()):
        target=out/f'{name}__{stem}.jpg';source=TRAIN/'images'/f'{stem}.jpg'
        shutil.copy2(source,target)
        with Image.open(source) as im:thumb=ImageOps.contain(im.convert('RGB'),(480,425))
        x,y=(i%3)*500,(i//3)*500
        sheet.paste(thumb,(x+(500-thumb.width)//2,y+35))
        draw.text((x+8,y+5),name,fill='black',font=font)
        draw.text((x+8,y+465),stem,fill='black',font=font)
        rows.append({'class':name,'filename':target.name,'original_training_path':str(source),
                     'visual_check':'Inspected original full frame; no visible standing or flowing water.'})
    sheet.save(out/'contact_sheet.jpg',quality=96)
    with (out/'sources.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):z.write(p,p.relative_to(out.parent))
    print(f'5 original training images: {out}')
    print(out.with_suffix('.zip'))

def main():
    REVIEW.mkdir(exist_ok=True)
    groups={i:[] for i in range(5)}
    for label in sorted((TRAIN/'labels').glob('*.txt')):
        classes={int(line.split()[0]) for line in label.read_text().splitlines() if len(line.split())==5}
        for c in classes:
            if c in groups:groups[c].append(label.stem)
    rng=random.Random(312)
    records={}
    for c,name in enumerate(NAMES):
        if name not in ['joint_defect','sediment_deposit']:continue
        rng.shuffle(groups[c])
        chosen=[]
        for stem in groups[c]:
            if (TRAIN/'images'/f'{stem}.jpg').exists():chosen.append(stem)
            if len(chosen)==30:break
        records[name]=chosen
        sheet=Image.new('RGB',(1500,1800),'white');d=ImageDraw.Draw(sheet)
        for i,stem in enumerate(chosen):
            with Image.open(TRAIN/'images'/f'{stem}.jpg') as im:
                thumb=ImageOps.contain(im.convert('RGB'),(294,266))
            x,y=(i%5)*300,(i//5)*300
            sheet.paste(thumb,(x,y));d.text((x+3,y+270),f'{i}: {stem}',fill='black')
        sheet.save(REVIEW/f'{name}.jpg',quality=95)
    (REVIEW/'candidates.json').write_text(json.dumps(records,indent=2))
    print(REVIEW)

if __name__=='__main__':
    package() if '--package' in sys.argv else main()
