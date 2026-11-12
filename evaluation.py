"""Image inference and metric computation for a checkpoint."""
import json
from pathlib import Path
import torch
from data import read_rows, read_image
from processing import predict_height, spat2freq, encode_images
from retrieval import rank_places, localization_distances

@torch.inference_mode()
def evaluate_model(model, head, config):
    model.eval()
    if head is not None:head.eval()
    device=config['device']; task=config['task']; rows=read_rows(config.get('query_dir') or config['data'],'test')
    def height_predictions(encoder,prototypes):
        heights=[]
        for start in range(0,len(rows),32):
            x=torch.stack([read_image(r,(336,448)) for r in rows[start:start+32]]).to(device)
            if config['frequency']:x=spat2freq(x)
            z=encode_images(encoder,x,device,config.get('amp',True))
            heights.extend(predict_height(z,prototypes).tolist())
        return heights
    if task=='height':
        heights=height_predictions(model,head.weight)
        predictions=[dict(r,predicted_height_m=float(h),error_m=abs(h-r['height_m'])) for r,h in zip(rows,heights)]
        errors=[r['error_m'] for r in predictions]
        metrics=dict(n=len(rows),mae_m=sum(errors)/len(errors),**{f'P_lt_{t}m':sum(e<t for e in errors)*100/len(errors) for t in (25,50,100)})
        return metrics,predictions
    height_model=prototypes=None
    if config['crop_mode']=='predicted':
        if not config.get('height_checkpoint'):
            raise ValueError('Predicted-altitude cropping requires --height-checkpoint')
        from models import HeightEncoder
        saved=torch.load(config['height_checkpoint'],map_location='cpu',weights_only=False)
        height_model=HeightEncoder().to(device).eval()
        height_model.load_state_dict(saved['model'],strict=True)
        prototypes=saved['classifier']['weight'].to(device)
        heights=height_predictions(height_model,prototypes)
        del height_model,prototypes,saved
    else:heights=[r['height_m'] if config['crop_mode']=='gt' else None for r in rows]
    gallery=read_rows(config.get('gallery_dir') or config['data'],'gallery'); ids=sorted({r['place_id'] for r in gallery})
    xy={}
    for r in gallery:
        point=(r['east_m'],r['north_m'])
        if r['place_id'] in xy and xy[r['place_id']]!=point:raise ValueError('Coordinates differ for the same place')
        xy[r['place_id']]=point
    # Keep one descriptor per gallery image on CPU; place scores are streamed per query.
    features=[]
    for start in range(0,len(gallery),16):
        x=torch.stack([read_image(r,(224,224)) for r in gallery[start:start+16]]).to(device)
        features.append(encode_images(model,x,device,config.get('amp',True)).cpu())
    features=torch.cat(features); predictions=[]
    g=torch.nn.functional.normalize(features,dim=1) if config['renormalize_descriptor'] else features
    lookup={place:i for i,place in enumerate(ids)}
    inverse=torch.tensor([lookup[r['place_id']] for r in gallery],device=device)
    g=g.to(device)
    for start in range(0,len(rows),16):
        group=rows[start:start+16]; hs=heights[start:start+16]
        images=torch.stack([read_image(r,(224,224),predicted_height=h) for r,h in zip(group,hs)])
        q=encode_images(model,images,device,config.get('amp',True))
        if config['renormalize_descriptor']:q=torch.nn.functional.normalize(q,dim=1)
        indices,values=rank_places(q,g,inverse,5)
        for r,h,rank,scores in zip(group,hs,indices.tolist(),values.tolist()):
            top=[ids[i] for i in rank]
            distances=localization_distances([xy[i] for i in top],[r['east_m'],r['north_m']])
            predictions.append(dict(r,**({'predicted_height_m':h} if config['crop_mode']=='predicted' else {}),top_place_ids=top,scores=scores,distances_m=distances))
    metrics=dict(n=len(rows),**{f'R@{k}_{t}m':sum(min(r['distances_m'][:k])<=t for r in predictions)*100/len(rows) for t in (25,50,100) for k in (1,5)})
    return metrics,predictions

def write_evaluation(model,head,config,folder):
    metrics,predictions=evaluate_model(model,head,config)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2))
    (folder/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    return metrics
