"""Make labeled contact sheets for manual selection of training crack examples."""
from pathlib import Path
import random
import json
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'final_dataset_v5'/'train'
OUT=ROOT/'crack_example_review'

def main():
    OUT.mkdir(exist_ok=True)
    records=[]
    rng=random.Random(42)
    for prefix in ['sewer_defects_', 'sewage_defect_detection_', 'sewage_']:
        items=[]
        for label in sorted((BASE/'labels').glob(prefix+'*.txt')):
            boxes=[list(map(float,line.split())) for line in label.read_text().splitlines() if len(line.split())==5 and line.split()[0]=='0']
            if boxes:
                items.append((label,boxes))
        rng.shuffle(items)
        for label,boxes in items[:24]:
            image=BASE/'images'/(label.stem+'.jpg')
            if image.exists():
                records.append({'image':str(image),'boxes':boxes})
    for page in range((len(records)+17)//18):
        sheet=Image.new('RGB',(1500,1200),'white')
        draw=ImageDraw.Draw(sheet)
        for local,r in enumerate(records[page*18:(page+1)*18]):
            im=Image.open(r['image']).convert('RGB')
            d=ImageDraw.Draw(im)
            W,H=im.size
            for _,x,y,w,h in r['boxes']:
                d.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),outline='red',width=2)
            thumb=ImageOps.contain(im,(490,165))
            x,y=(local%3)*500,(local//3)*200
            sheet.paste(thumb,(x,y))
            draw.text((x+3,y+168),f"{page*18+local}: {Path(r['image']).stem}",fill='black')
        sheet.save(OUT/f'page_{page}.jpg',quality=93)
    (OUT/'candidates.json').write_text(json.dumps(records,indent=2))
    print(f'{len(records)} candidates in {OUT}')

if __name__=='__main__':
    main()
