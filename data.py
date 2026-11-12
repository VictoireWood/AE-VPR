"""Scan crop directories and read altitude, angle, and coordinates from filenames."""
import re
from functools import lru_cache
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from processing import image_tensor, normalize_height
from geometry import Footprints

REGION_EPSG = {'CT01':32650,'CT02':32651,'QD':32651,'QD01':32651,'QD02':32651}

@lru_cache(maxsize=16)
def coordinate_transform(epsg, inverse=False):
    from pyproj import Transformer
    return Transformer.from_crs(epsg if inverse else 4326,4326 if inverse else epsg,always_xy=True)


def image_metadata(path):
    path=Path(path)
    fields=path.stem.strip('@').split('@')
    row=dict(path=str(path))
    try:
        if len(fields)!=4 or not path.stem.startswith('@') or not path.stem.endswith('@'):
            raise ValueError('Expected @angle@altitude@E@N@ filenames')
        angle,height,east,north=map(float,fields)
        row.update(angle_degrees=angle,height_m=height,east_m=east,north_m=north)
        if not all(math.isfinite(v) for v in row.values() if isinstance(v,float)):
            raise ValueError('Non-finite value')
        if row['height_m']<=0:raise ValueError('Altitude must be positive')
    except (ValueError,TypeError) as exc:
        raise ValueError('Cannot read metadata from image filename: '+str(path)+'; expected @angle@altitude@E@N@.png') from exc
    row['place_id']=f"{row['east_m']:.6f},{row['north_m']:.6f}"
    for parent in path.parents:
        match=re.fullmatch(r'(\d{8}|\d{6}|\d{4})(?:_.*)?',parent.name)
        if match or parent.name=='latest':
            row['date']=match.group(1) if match else 'latest'
            break
    return row

def read_rows(path, split):
    path=Path(path)
    if not path.is_dir():raise ValueError('Expected an image directory: '+str(path))
    roles={'train':('train',),'test':('test','query'),'gallery':('gallery','database')}
    selected=next((path/name for name in roles[split] if (path/name).is_dir()),None)
    if selected is None:
        if any((path/name).is_dir() for names in roles.values() for name in names):
            raise ValueError('The data directory is missing the '+split+' subdirectory')
        selected=path
    images=sorted(p for p in selected.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff') and p.is_file())
    if not images:raise ValueError('No images found: '+str(selected))
    return [dict(image_metadata(p),split=split) for p in images]

def read_image(row, size, augment=False, predicted_height=None):
    with Image.open(row['path']) as source:
        image=source.convert('RGB')
        if predicted_height is not None:
            image,_=normalize_height(image,predicted_height)
        if augment:
            image=T.ColorJitter(.2,.3,.3,.2)(image)
        return image_tensor(image,size)

class TrainingData:
    def __init__(self, config):
        self.task=config['task']; self.rows=read_rows(config['data'],'train')
        self.batch=config['batch_size']; self.places=None
        if self.task=='height':
            for r in self.rows:
                if not math.isfinite(r['height_m']) or not 100<=r['height_m']<700:
                    raise ValueError('Altitude classification training requires heights in [100, 700)')
            if len(self.rows)<self.batch:raise ValueError('Fewer training images than batch_size')
        else:
            places={}
            for r in self.rows:
                point=(r['east_m'],r['north_m'])
                if not all(math.isfinite(v) for v in point):raise ValueError('Place coordinates must be finite')
                if r['place_id'] in places and places[r['place_id']]!=point:
                    raise ValueError('Coordinates differ for the same place')
                places[r['place_id']]=point
            self.places=[dict(place_id=k,east_m=v[0],north_m=v[1]) for k,v in places.items()]
            ids=list(places)
            self.indices={v:i for i,v in enumerate(ids)}
            self.geometry=Footprints([[r['east_m'],r['north_m']] for r in self.places])
            for r in self.rows:
                if not math.isfinite(r['height_m']) or r['height_m']<=0 or not math.isfinite(r['angle_degrees']):
                    raise ValueError('Invalid footprint parameters')
            if len(self.rows)<self.batch:raise ValueError('Fewer training images than batch_size')
    def batch_at(self, seed):
        rng=np.random.default_rng(seed)
        if self.task=='height':
            rows=[self.rows[int(i)] for i in rng.choice(len(self.rows),self.batch,replace=False)]
            return torch.stack([read_image(r,(336,448),True) for r in rows]),torch.tensor([int((r['height_m']-100)//50) for r in rows]),None
        rows=[self.rows[int(i)] for i in rng.choice(len(self.rows),self.batch,replace=False)]
        images=torch.stack([read_image(r,(224,224),True) for r in rows])
        labels=torch.tensor([self.indices[r['place_id']] for r in rows])
        overlaps=np.stack([self.geometry.overlap(self.indices[r['place_id']],r['height_m'],r['angle_degrees']) for r in rows])
        return images,labels,torch.from_numpy(overlaps)
