"""Command-line options, training, checkpoint recovery, and evaluation."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from classifiers import AAMC, FootprintMarginClassifier
from models import HeightEncoder, RetrievalEncoder
from data import TrainingData
from processing import spat2freq
from evaluation import write_evaluation

DEFAULTS=dict(device='cpu',epochs=5,updates_per_epoch=2000,batch_size=64,seed=20260919,
              model_lr=1e-4,head_lr=1e-2,frequency=True,crop_mode='predicted',
              renormalize_descriptor=False,resume=None,amp=True)
KEYS=set(DEFAULTS)|{'task','output_dir','data','backbone_checkpoint','checkpoint','height_checkpoint','query_dir','gallery_dir','region','dataset'}

def build_options(supplied, training):
    if set(supplied)-KEYS:raise ValueError('Unknown options: '+str(sorted(set(supplied)-KEYS)))
    c={**DEFAULTS,**supplied}
    if c.get('task') not in ('height','vpr'):raise ValueError('task must be height or vpr')
    if 'batch_size' not in supplied and c['task']=='vpr':c['batch_size']=32
    if (c.get('region') or c.get('dataset')) and not c.get('data'):raise ValueError('--region/--dataset requires --data with a data root')
    if c.get('region'):
        c['data']=str(Path(c['data'])/'Crops'/c.pop('region')/c['task']/'train')
    if c.get('dataset'):
        dataset=c.pop('dataset')
        root=Path(c['data'])
        region='QD' if dataset.startswith('QD') else dataset
        c.setdefault('query_dir',str(root/'Query'/dataset))
        if c['task']=='vpr':c.setdefault('gallery_dir',str(root/'Crops'/region/'vpr'/'gallery'))
    if c['crop_mode'] not in ('none','predicted','gt'):raise ValueError('Invalid crop_mode')
    for key in ('epochs','updates_per_epoch','batch_size'):
        if type(c[key]) is not int or c[key]<=0:raise ValueError(key+' must be a positive integer')
    required=['output_dir']
    if training:required+=['data']
    else:
        required+=['checkpoint']
        if not c.get('data') and not c.get('query_dir'):raise ValueError('Provide --data or --query-dir')
        if c['task']=='vpr' and not c.get('data') and not c.get('gallery_dir'):raise ValueError('--gallery-dir is required')
    if training and not c['resume']:required+=['backbone_checkpoint']
    for key in required:
        if not c.get(key):raise ValueError('Missing option: '+key)
    return c

def parse_options(training, argv=None):
    parser=argparse.ArgumentParser(description='Train altitude classification or FAMC place recognition' if training else 'Evaluate a checkpoint')
    parser.add_argument('--task',choices=['height','vpr'],required=True)
    parser.add_argument('--data',required=training,help='Data root or crop directory')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--frequency',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    if training:
        parser.add_argument('--region',choices=['CT01','CT02','QD'],help='Use the region-specific Crops directory under the data root')
        parser.add_argument('--backbone-checkpoint')
        parser.add_argument('--resume')
        parser.add_argument('--epochs',type=int,default=5)
        parser.add_argument('--updates-per-epoch',type=int,default=2000)
        parser.add_argument('--batch-size',type=int)
        parser.add_argument('--model-lr',type=float,default=1e-4)
        parser.add_argument('--head-lr',type=float,default=1e-2)
        parser.add_argument('--seed',type=int,default=20260919)
    else:
        parser.add_argument('--dataset',choices=['CT01','CT02','QD01','QD02'],help='Use the dataset-specific Query directory and gallery under the data root')
        parser.add_argument('--query-dir',help='Query image directory')
        parser.add_argument('--gallery-dir',help='Gallery image directory')
        parser.add_argument('--height-checkpoint',help='Altitude checkpoint for predicted-scale normalization')
        parser.add_argument('--checkpoint',required=True)
        parser.add_argument('--crop-mode',choices=['none','predicted','gt'],default='predicted')
        parser.add_argument('--renormalize-descriptor',action='store_true')
    supplied={k:v for k,v in vars(parser.parse_args(argv)).items() if v is not None}
    try:return build_options(supplied,training)
    except ValueError as exc:parser.error(str(exc))

def construct(c,classes=0):
    if c['task']=='height':model,head=HeightEncoder(),AAMC(1024,12,s=100,m=.2)
    else:model,head=RetrievalEncoder(),FootprintMarginClassifier(3072,classes,s=30,m_min=0,m_max=.4)
    return model,head

def initialize_backbone(model,c):
    state=torch.load(c['backbone_checkpoint'],map_location='cpu',weights_only=False)
    if c['task']=='height':
        target=model[0].model
        target.load_state_dict({k:v for k,v in state.items() if k in target.state_dict()},strict=True)
    else:
        model.backbone.load_state_dict({k.removeprefix('features.'):v for k,v in state.items() if k.startswith('features.')},strict=True)

def freeze(model,c):
    if c['task']=='height':
        for module in (model[0].model.conv1,model[0].model.bn1,model[0].model.layer1):module.requires_grad_(False)
    else:
        for module in model.backbone[:5]:module.requires_grad_(False)

def atomic_save(state,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def train(c):
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    data=TrainingData(c);classes=len(data.places) if data.places is not None else 0
    model,head=construct(c,classes);freeze(model,c)
    if not c['resume']:initialize_backbone(model,c)
    model.to(c['device']);head.to(c['device'])
    opt=torch.optim.Adam([dict(params=[p for p in model.parameters() if p.requires_grad],lr=c['model_lr']),dict(params=head.parameters(),lr=c['head_lr'])])
    enabled=c['amp'] and str(c['device']).startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=enabled)
    epoch=offset=step=0;loss_sum=0.
    if c['resume']:
        saved=torch.load(c['resume'],map_location='cpu',weights_only=False)
        identity={k:v for k,v in c.items() if k not in ('resume','output_dir')}
        if identity!={k:v for k,v in saved['config'].items() if k not in ('resume','output_dir')}:raise ValueError('Resume options differ from the original training options')
        model.load_state_dict(saved['model']);head.load_state_dict(saved['classifier' if c['task']=='height' else 'head'])
        opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        epoch,offset,step,loss_sum=[saved[k] for k in ('epoch','offset','step','loss_sum')]
        torch.set_rng_state(saved['rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if enabled:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pth').exists() and not c['resume']:raise ValueError('The output directory contains a checkpoint; use --resume or a new directory')
    def state():
        return dict(config=c,model=model.state_dict(),**{'classifier' if c['task']=='height' else 'head':head.state_dict()},optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,offset=offset,step=step,loss_sum=loss_sum,rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if enabled else None)
    while epoch<c['epochs']:
        model.train();head.train()
        while offset<c['updates_per_epoch']:
            # Sample RNG is local; augmentation uses a deterministic per-update seed.
            seed=c['seed']+epoch*c['updates_per_epoch']+offset
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed);images,labels,iou=data.batch_at(seed)
            images=images.to(c['device']);labels=labels.to(c['device']);opt.zero_grad(set_to_none=True)
            if c['task']=='height' and c['frequency']:images=spat2freq(images)
            with torch.autocast(device_type=torch.device(c['device']).type,enabled=enabled):
                features=model(images) if c['task']=='height' else torch.cat([model(x) for x in images.split(16)])
            logits=head(features.float(),labels)[0] if iou is None else head(features,labels,iou.to(c['device']))[0]
            loss=torch.nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            offset+=1;step+=1;loss_sum+=float(loss.detach())
            if step%20==1:print(f'epoch={epoch+1} step={step} loss={float(loss.detach()):.6f}',flush=True)
            if step%200==0:atomic_save(state(),out/'checkpoint.pth')
        ep=epoch+1
        record=dict(epoch=ep,step=step,train_loss=loss_sum/c['updates_per_epoch'])
        # Epoch files and latest checkpoint both contain complete restart state.
        epoch+=1;offset=0;loss_sum=0.
        folder=out/f'epochs/epoch_{ep:03d}'
        atomic_save(state(),folder/'checkpoint.pth')
        (folder/'metrics.json').write_text(json.dumps(record,indent=2))
        atomic_save(state(),out/'checkpoint.pth')
    (out/'complete.json').write_text(json.dumps(dict(success=True,updates=step),indent=2))

def evaluate(c):
    saved=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
    if c['task']=='height':
        model,head=construct(c)
        head.load_state_dict(saved['classifier'],strict=True)
        head=head.to(c['device'])
    else:
        model,head=RetrievalEncoder(),None
    model.load_state_dict(saved['model'],strict=True)
    return write_evaluation(model.to(c['device']),head,c,c['output_dir'])
