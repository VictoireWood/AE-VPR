"""Single-image altitude prediction, cropping, descriptor extraction, and place ranking."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from models import load_models
from processing import image_tensor, spat2freq, predict_height, normalize_height, encode_images
from retrieval import rank_places
from data import read_rows, read_image

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--height-checkpoint', required=True)
    parser.add_argument('--retrieval-checkpoint', required=True)
    parser.add_argument('--query-image', required=True)
    parser.add_argument('--gallery-dir', required=True, help='Directory containing gallery crops')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--frequency', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--renormalize-descriptor', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--output', default='outputs/prediction.json')
    c = vars(parser.parse_args(argv))
    device = c.get('device', 'cpu')
    height, prototypes, retrieval = load_models(c['height_checkpoint'], c['retrieval_checkpoint'], device)
    with Image.open(c['query_image']) as source:
        image = source.convert('RGB')
    with torch.inference_mode():
        x = image_tensor(image, (336, 448)).unsqueeze(0).to(device)
        if c.get('frequency', True):
            x = spat2freq(x)
        prediction = float(predict_height(encode_images(height,x,device,c['amp']), prototypes)[0])
        crop, box = normalize_height(image, prediction)
        query = encode_images(retrieval,image_tensor(crop, (224, 224)).unsqueeze(0),device,c['amp'])
        if c.get('renormalize_descriptor', False):
            query = torch.nn.functional.normalize(query, dim=1)
        gallery_rows=read_rows(c['gallery_dir'],'gallery')
        place_ids=sorted({r['place_id'] for r in gallery_rows})
        lookup={v:i for i,v in enumerate(place_ids)}
        features=[]
        for start in range(0,len(gallery_rows),16):
            images=torch.stack([read_image(r,(224,224)) for r in gallery_rows[start:start+16]]).to(device)
            features.append(encode_images(retrieval,images,device,c['amp']).cpu())
        features=torch.cat(features)
        if c['renormalize_descriptor']:features=torch.nn.functional.normalize(features,dim=1)
        ids, scores = rank_places(query.cpu(),features,
            [lookup[r['place_id']] for r in gallery_rows],c['top_k'])
    output = dict(predicted_height_m=prediction, crop_box=list(box), place_ids=[place_ids[i] for i in ids[0].tolist()], scores=scores[0].tolist())
    target = Path(c['output']); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False))

if __name__ == '__main__':
    main()
