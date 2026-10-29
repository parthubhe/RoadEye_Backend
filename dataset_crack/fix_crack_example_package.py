"""Replace automatically selected crack examples with visually checked fractures."""
import csv
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'training_examples_final_25'
SOURCE=ROOT/'final_dataset_v5'/'train'/'images'
NAMES=['crack','corrosion_rust','sediment_deposit','root_intrusion','joint_defect']
CRACKS=['sewer_defects_012115.jpg','sewer_defects_002088.jpg',
        'sewer_defects_004160.jpg','sewer_defects_006413.jpg','sewage_defect_detection_011018.jpg']

def main():
    for name in CRACKS:
        if not (SOURCE/name).is_file():
            raise FileNotFoundError(name)
    with (OUT/'image_sources.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    archive=ROOT/'crack_example_review'/f'replaced_{datetime.now():%Y%m%d_%H%M%S}'
    archive.mkdir(parents=True,exist_ok=True)
    for old in (OUT/'crack').iterdir():
        if old.is_file():
            shutil.move(str(old),str(archive/old.name))
    for name in CRACKS:
        shutil.copy2(SOURCE/name,OUT/'crack'/name)
    rows=[{'class':'crack','filename':name,'original_training_path':str(SOURCE/name)} for name in CRACKS]+[r for r in rows if r['class']!='crack']
    with (OUT/'image_sources.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['class','filename','original_training_path'])
        writer.writeheader()
        writer.writerows(rows)
    sheet=Image.new('RGB',(1500,1250),'white')
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        with Image.open(OUT/r['class']/r['filename']) as im:
            preview=ImageOps.contain(im.convert('RGB'),(294,210))
        x,y=(i%5)*300,(i//5)*250
        sheet.paste(preview,(x,y+18))
        draw.text((x+3,y+2),r['class'],fill='black')
        draw.text((x+3,y+230),Path(r['filename']).stem,fill='black')
    sheet.save(OUT/'contact_sheet.jpg',quality=95)
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(OUT.parent))
    for name in NAMES:
        count=len(list((OUT/name).glob('*.jpg')))
        assert count==5,(name,count)
    print('Updated package: 5 images per class, 25 total.')

if __name__=='__main__':
    main()
