"""Render saved predictions at consistent display scale without rerunning inference."""
import json
import sys
import zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from infer_training_examples_rfdetr import NAMES,COLORS,PACKAGE

def main():
    out=Path(sys.argv[1]).resolve()
    data=json.loads((out/'predictions.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    sheet=Image.new('RGB',(2500,2350),'white')
    sd=ImageDraw.Draw(sheet)
    sd.text((12,10),'RF-DETR Nano | 50 epochs | confidence >= 0.25 | TRAINING examples',fill='black',font=font)
    for i,row in enumerate(data['images']):
        with Image.open(PACKAGE/row['class']/row['filename']) as original:
            W,H=original.size
            im=ImageOps.contain(original.convert('RGB'),(900,900))
        sx,sy=im.width/W,im.height/H
        d=ImageDraw.Draw(im)
        for box in row['detections']:
            color=COLORS[NAMES.index(box['class'])]
            x1,y1,x2,y2=box['xyxy']
            corners=(x1*sx,y1*sy,x2*sx,y2*sy)
            d.rectangle(corners,outline=color,width=4)
            text=f"{box['class']} {box['confidence']:.2f}"
            length=d.textlength(text,font=font)
            x=max(0,min(corners[0],im.width-length-4))
            y=max(0,corners[1]-23)
            d.rectangle((x,y,x+length+4,y+23),fill='black')
            d.text((x+2,y+1),text,fill=color,font=font)
        im.save(out/row['class']/row['filename'],quality=96)
        thumb=ImageOps.contain(im,(490,380))
        x,y=(i%5)*500,45+(i//5)*460
        sd.text((x+5,y),row['class'],fill='black',font=font)
        sheet.paste(thumb,(x+(500-thumb.width)//2,y+24))
        sd.text((x+5,y+407),row['filename'],fill='black',font=font)
        sd.text((x+5,y+430),f"{len(row['detections'])} detections",fill='black',font=font)
    sheet.save(out/'predictions_contact_sheet.jpg',quality=96)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    print(out/'predictions_contact_sheet.jpg')

if __name__=='__main__':main()
