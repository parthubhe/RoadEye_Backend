"""Create an operating-point confusion matrix on the unchanged full v5 test split."""
import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRNano

ROOT=Path(__file__).resolve().parents[1]
NAMES=['crack','corrosion_rust','sediment_deposit','root_intrusion','joint_defect']
RUN=ROOT/'runs/pipe_proto/rfdetr_nano_v5_normal_512_seed0_20261003_193042_66b2b7'

def match(pred, truth):
    if len(pred)==0 or len(truth)==0:return []
    a=pred[:,None,:];b=truth[None,:,:]
    inter=np.maximum(0,np.minimum(a[:,:,2:],b[:,:,2:])-np.maximum(a[:,:,:2],b[:,:,:2])).prod(2)
    area_a=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)
    area_b=np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)
    ious=inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-9)
    candidates=np.argwhere(ious>=.5)
    candidates=sorted(candidates,key=lambda pair:-ious[tuple(pair)])
    pairs=[];usedp=set();usedt=set()
    for p,t in candidates:
        if int(p) not in usedp and int(t) not in usedt:
            pairs.append((int(p),int(t)));usedp.add(int(p));usedt.add(int(t))
    return pairs

def main():
    out=RUN/f'test_confusion_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(exist_ok=False)
    checkpoint=RUN/'checkpoint_best_total.pth'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=RFDETRNano(pretrain_weights=str(checkpoint),num_classes=5,resolution=512,device=device)
    if device=='cuda':model.inference(compile=False,dtype=torch.float16,inplace=True)
    images=sorted((ROOT/'final_dataset_v5/test/images').glob('*'))
    images=[p for p in images if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp','.webp'}]
    if len(images)!=1986:raise ValueError(f'Expected 1986 test images, got {len(images)}')
    matrix=np.zeros((6,6),dtype=np.int64)
    for i,path in enumerate(images):
        im=Image.open(path).convert('RGB');W,H=im.size
        gt=[];classes=[]
        for line in (ROOT/'final_dataset_v5/test/labels'/f'{path.stem}.txt').read_text().splitlines():
            if not line.strip():continue
            c,x,y,w,h=map(float,line.split())
            if w<=0 or h<=0:continue
            classes.append(int(c));gt.append([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])
        pred=model.predict(im,threshold=.25)
        if isinstance(pred,list):pred=pred[0]
        valid=(pred.class_id>=0)&(pred.class_id<5)
        boxes=np.asarray(pred.xyxy)[valid];pc=np.asarray(pred.class_id)[valid].astype(int)
        pairs=match(boxes,np.asarray(gt).reshape(-1,4))
        for p,t in pairs:matrix[pc[p],classes[t]]+=1
        for p in set(range(len(pc)))-{p for p,t in pairs}:matrix[pc[p],5]+=1
        for t in set(range(len(gt)))-{t for p,t in pairs}:matrix[5,classes[t]]+=1
        if (i+1)%100==0:print(f'[{i+1}/{len(images)}]',flush=True)
    names=NAMES+['background']
    with (out/'confusion_matrix.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['predicted / true']+names)
        for name,row in zip(names,matrix):writer.writerow([name]+row.tolist())
    stats=[]
    for c,name in enumerate(NAMES):
        tp=int(matrix[c,c]);fp=int(matrix[c].sum()-tp);fn=int(matrix[:,c].sum()-tp)
        stats.append({'class':name,'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)})
    report={'split':'test','images':len(images),'confidence':.25,'matching_iou':.5,
            'matching':'class-agnostic greedy descending IoU, one-to-one',
            'precision_mode':'FP16 on CUDA; FP32 on CPU','checkpoint':str(checkpoint),
            'rows':'predicted','columns':'true','class_order':names,'matrix':matrix.tolist(),'class_metrics':stats,
            'note':'Operating-point diagnostic; different protocol from saved RF-DETR AP/F1 sweep. Background/background is undefined and displayed as zero.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,8))
    plot=ax.imshow(matrix,cmap='Blues')
    ax.set_xticks(range(6),names,rotation=35,ha='right');ax.set_yticks(range(6),names)
    for row in range(6):
        for col in range(6):
            ax.text(col,row,str(matrix[row,col]),ha='center',va='center',color='white' if matrix[row,col]>matrix.max()/2 else 'black')
    ax.set_xlabel('True');ax.set_ylabel('Predicted')
    ax.set_title('RF-DETR Nano 50e — full v5 test\nconfidence 0.25, matching IoU 0.50')
    fig.colorbar(plot,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True);print(f'OUTPUT: {out}',flush=True)

if __name__=='__main__':main()
