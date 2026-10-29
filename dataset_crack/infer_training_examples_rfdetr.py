"""Run RF-DETR50 on the selected training examples and render predictions."""
import csv
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rfdetr import RFDETRNano

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'training_examples_final_25'
CHECKPOINT=ROOT/'runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7/checkpoint_best_total.pth'
NAMES=['crack','corrosion_rust','sediment_deposit','root_intrusion','joint_defect']
COLORS=['#ff6633','#00ccff','#ffcc00','#00dd88','#ee66ff']

def main():
    out=ROOT/f'training_examples_rfdetr50_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=RFDETRNano(pretrain_weights=str(CHECKPOINT),num_classes=5,resolution=512,device=device)
    if device=='cuda':
        model.inference(compile=False,dtype=torch.float16,inplace=True)
    with (PACKAGE/'image_sources.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else ImageFont.load_default()
    sheet=Image.new('RGB',(2500,2850),'#f5f5f5')
    sd=ImageDraw.Draw(sheet)
    sd.text((12,10),'RF-DETR Nano | 50 epochs | confidence >= 0.25 | TRAINING examples',fill='black',font=font)
    results=[]
    for i,row in enumerate(rows):
        im=Image.open(PACKAGE/row['class']/row['filename']).convert('RGB')
        if device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        pred=model.predict(im,threshold=.25)
        if device=='cuda': torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        if isinstance(pred,list): pred=pred[0]
        annotated=im.copy()
        d=ImageDraw.Draw(annotated)
        boxes=[]
        for xyxy,conf,cls in zip(pred.xyxy,pred.confidence,pred.class_id):
            cls=int(cls)
            if not 0<=cls<5: continue
            box=[float(v) for v in xyxy]
            color=COLORS[cls]
            d.rectangle(box,outline=color,width=3)
            text=f'{NAMES[cls]} {float(conf):.2f}'
            x,y=box[:2]
            y=max(0,y-19)
            bounds=d.textbbox((x,y),text,font=font)
            d.rectangle(bounds,fill='black')
            d.text((x,y),text,fill=color,font=font)
            boxes.append({'class':NAMES[cls],'confidence':float(conf),'xyxy':box})
        folder=out/row['class']
        folder.mkdir(exist_ok=True)
        annotated.save(folder/row['filename'],quality=95)
        thumb=ImageOps.contain(annotated,(490,480))
        x,y=(i%5)*500,50+(i//5)*560
        sd.text((x+5,y),f"Folder: {row['class']}",fill='black',font=font)
        sheet.paste(thumb,(x+(500-thumb.width)//2,y+25))
        sd.text((x+5,y+508),row['filename'],fill='black',font=font)
        sd.text((x+5,y+530),f'{len(boxes)} detections',fill='black',font=font)
        results.append({**row,'inference_ms':elapsed,'detections':boxes})
        print(f'[{i+1}/25] {row["filename"]}: {len(boxes)} detections',flush=True)
    sheet.save(out/'predictions_contact_sheet.jpg',quality=96)
    (out/'predictions.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT),'confidence':.25,'resolution':512,'split':'train','images':results},indent=2))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    print(f'Results: {out}',flush=True)

if __name__=='__main__':main()
