"""Generate map crops with altitude, angle, and coordinates encoded in filenames."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from data import coordinate_transform, image_metadata, REGION_EPSG

# Read georeferencing tags with Pillow and decode maps with OpenCV.
Image.MAX_IMAGE_PIXELS = None


class OutsideMap(ValueError):
    """The footprint and interpolation margin extend beyond the map."""


def map_metadata(path):
    # Compute metres per pixel separately for each axis using local WGS84 curvature.
    with Image.open(path) as image:
        tags = image.tag_v2
        sx, sy, _ = tags[33550]
        tie = tags[33922]
        keys = tuple(tags[34735])
        entries = [keys[i:i + 4] for i in range(4, len(keys), 4)]
        if (2048, 0, 1, 4326) not in entries:
            raise ValueError('Expected a WGS84 geographic GeoTIFF (EPSG:4326)')
        if 34264 in tags or sx <= 0 or sy <= 0:
            raise ValueError('Expected a north-up map without rotation or shear')
        if (1025, 0, 1, 2) in entries:
            raise ValueError('Expected PixelIsArea georeferencing rather than PixelIsPoint')
        lon, lat = tie[3] - tie[0] * sx, tie[4] + tie[1] * sy
        w, h = image.size
    phi = math.radians(lat - h * sy / 2)
    e2 = 6.69437999014e-3
    d = 1 - e2 * math.sin(phi) ** 2
    mx = sx * math.pi / 180 * 6378137 / math.sqrt(d) * math.cos(phi)
    my = sy * math.pi / 180 * 6378137 * (1 - e2) / d ** 1.5
    return dict(width=w, height=h, lon=lon, lat=lat, sx=sx, sy=sy, mx=mx, my=my)


def crop_view(image, center, height, angle, mx, my, size=(224, 224)):
    """Generate a BGR crop using a supersampled affine transform; size is (width, height)."""
    w, h = size
    if not all(math.isfinite(float(x)) for x in (*center, height, angle, mx, my)):
        raise ValueError('Crop parameters must be finite')
    if min(height, mx, my, w, h) <= 0:
        raise ValueError('Altitude, pixel scales, and output dimensions must be positive')
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    a = np.array([[c / mx, -s / mx], [s / my, c / my]]) @ np.diag(
        [height * 2048 / 1200 / w, height * 1536 / 1200 / h])
    center = np.asarray(center, dtype=np.float64)
    half = np.array([w / 2, h / 2])
    radius = abs(a) @ half
    if np.any(center - radius < 4) or np.any(center + radius > np.array(image.shape[1::-1]) - 4):
        raise OutsideMap('The footprint or Lanczos interpolation margin exceeds the map boundary')
    factor = max(1, math.ceil(float(np.linalg.svd(a, compute_uv=False).max())))
    mat = np.column_stack([a / factor, center - a @ (half - .5 / factor)])
    # Warp the source window containing the footprint; the caller loads each map once.
    lo = np.floor(center - radius - 4).astype(int)
    hi = np.ceil(center + radius + 4).astype(int)
    mat[:, 2] -= lo
    patch = image[lo[1]:hi[1], lo[0]:hi[0]]
    result = cv2.warpAffine(patch, mat, (w * factor, h * factor),
                            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
    if factor > 1:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
    return result


def centres_from_maps(metadata, epsg, spacing=None, bounds=None):
    """Generate crop centres on a map-aligned grid with a rotation margin."""
    if spacing is None:
        spacing=(2048 / 1200 * 100 / 4,1536 / 1200 * 100 / 4)
    if np.isscalar(spacing):spacing=(float(spacing),float(spacing))
    if len(spacing)!=2 or any(not math.isfinite(float(v)) or v<=0 for v in spacing):
        raise ValueError('spacing_m must contain one or two positive values')
    reference=metadata[0]
    radius=math.hypot(2048,1536)/1200*200/2
    rx,ry=radius/reference['mx']+8,radius/reference['my']+8
    xs=np.arange(rx,reference['width']-1-rx,spacing[0]/reference['mx'])
    ys=np.arange(ry,reference['height']-1-ry,spacing[1]/reference['my'])
    transform=coordinate_transform(epsg)
    points=[]
    for cy in ys:
        for cx in xs:
            lon=reference['lon']+(cx+.5)*reference['sx']
            lat=reference['lat']-(cy+.5)*reference['sy']
            east,north=transform.transform(lon,lat)
            if bounds and not(bounds[0]<=east<bounds[2] and bounds[1]<=north<bounds[3]):continue
            # Keep the common physical area when maps have different extents.
            if all(m['lon']<=lon<m['lon']+m['width']*m['sx'] and
                   m['lat']-m['height']*m['sy']<lat<=m['lat'] for m in metadata):
                points.append((float(east),float(north)))
    return points


def prepare(config):
    mode=config['mode']
    if mode not in ('vpr','height','gallery'):raise ValueError('mode must be vpr, height, or gallery')
    sources=config['sources']
    if not sources or len({str(s['date']) for s in sources})!=len(sources):
        raise ValueError('Map sources must have unique dates')
    metadata=[map_metadata(s['path']) for s in sources]
    epsg=config.get('epsg') or ((32600 if metadata[0]['lat']>=0 else 32700)+int((metadata[0]['lon']+180)//6)+1)
    if config.get('centres_dir'):
        root=Path(config['centres_dir'])
        if not root.is_dir():raise ValueError('centres_dir must be an existing crop directory')
        xy={}
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'):
                row=image_metadata(path)
                xy[row['place_id']]=(row['east_m'],row['north_m'])
        points=list(xy.values())
    else:
        points=centres_from_maps(metadata,epsg,config.get('spacing_m'),config.get('bounds'))
    if not points:raise ValueError('No crop centres are available; adjust the bounds or grid spacing')
    inverse=coordinate_transform(epsg,inverse=True)
    lonlat=[inverse.transform(e,n) for e,n in points]
    split=config.get('split') or ('gallery' if mode=='gallery' else 'train')
    out=Path(config['output_dir'])
    rng=np.random.default_rng(config.get('seed',20260919))
    size=(480,360) if mode=='height' else (224,224)
    saved=skipped=duplicates=0;retained=set();written=set()
    cv2.setNumThreads(1)
    for source,meta in zip(sources,metadata):
        image=cv2.imread(str(source['path']),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(meta['height'],meta['width']):
            raise ValueError('Cannot read the map or its dimensions do not match: '+str(source['path']))
        folder=out/split/str(source['date']);folder.mkdir(parents=True,exist_ok=True)
        for (east,north),(lon,lat) in zip(points,lonlat):
            center=((lon-meta['lon'])/meta['sx']-.5,(meta['lat']-lat)/meta['sy']-.5)
            if mode=='gallery':views=[(150.,0.)]
            elif mode=='height':
                views=[(float(h),float(a)) for h in config['heights_m'] for a in config.get('angles_degrees',[0])]
                if not views or any(not 100<=h<700 for h,_ in views):raise ValueError('Altitude classification requires 100 <= height < 700')
            else:
                count=config.get('views_per_place_date',4)
                if type(count) is not int or count<1:raise ValueError('views_per_place_date must be a positive integer')
                views=[(150. if rng.random()<.5 else float(rng.uniform(100,200)),float(rng.integers(12)*30)) for _ in range(count)]
            for height,angle in views:
                try:crop=crop_view(image,center,height,angle,meta['mx'],meta['my'],size)
                except OutsideMap:
                    skipped+=1
                    continue
                # Use float repr to preserve crop parameters; directories store dates and filenames store geometry.
                name='@'+'@'.join(str(float(v)) for v in (angle,height,east,north))+'@.png'
                target=folder/name
                if target in written:
                    duplicates+=1
                    continue
                if target.exists():raise ValueError('Crop already exists: '+str(target))
                if not cv2.imwrite(str(target),crop):raise OSError('Failed to write image: '+str(target))
                written.add(target);retained.add((east,north));saved+=1
        print(f"date={source['date']} saved={saved} skipped={skipped} duplicate_views={duplicates}",flush=True)
    return dict(mode=mode,images=saved,skipped=skipped,duplicate_views=duplicates,
                places_requested=len(points),places_retained=len(retained),epsg=epsg)


def parse_options(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['vpr','height','gallery'],required=True)
    parser.add_argument('--data',help='Data root containing Map/ and Query/')
    parser.add_argument('--region',choices=['CT01','CT02','QD'])
    parser.add_argument('--map',dest='maps',action='append',nargs=2,metavar=('DATE','TIFF'))
    parser.add_argument('--output-dir',help='Defaults to DATA/Crops/REGION/height or vpr')
    parser.add_argument('--split',choices=['train','test','gallery'])
    parser.add_argument('--spacing-m',type=float,nargs='+',help='Centre spacing in metres: one value for both axes or EAST NORTH; default 42.6666667 32')
    parser.add_argument('--bounds',type=float,nargs=4,metavar=('E_MIN','N_MIN','E_MAX','N_MAX'))
    parser.add_argument('--centres-dir',help='Reuse centres from existing crop filenames')
    parser.add_argument('--seed',type=int,default=20260919)
    parser.add_argument('--views-per-place-date',type=int,default=4)
    parser.add_argument('--heights-m',type=float,nargs='+',default=list(range(100,700,5)))
    parser.add_argument('--angles-degrees',type=float,nargs='+',default=[0])
    options=vars(parser.parse_args(argv))
    if options['spacing_m'] is not None:
        if len(options['spacing_m']) not in (1,2):parser.error('--spacing-m takes one value or EAST NORTH')
        if len(options['spacing_m'])==1:options['spacing_m']*=2
    maps=options.pop('maps')
    root=options.pop('data');region=options.pop('region')
    if region:options['epsg']=REGION_EPSG[region]
    if maps:
        options['sources']=[dict(date=date,path=path) for date,path in maps]
    else:
        if not root or not region:parser.error('Provide --data and --region, or specify --map')
        folder=Path(root)/'Map'/region
        paths=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.tif','.tiff'))
        if not paths:parser.error('No maps found: '+str(folder))
        options['sources']=[dict(date=p.stem.strip('@').split('@')[0],path=str(p)) for p in paths]
        # Select map sources for altitude estimation or VPR.
        if options['mode']=='height':
            date='202306'
            options['sources']=[s for s in options['sources'] if s['date']==date]
        elif region in ('CT01','CT02'):
            options['sources']=[s for s in options['sources'] if s['date']!='202306']
        if not options['sources']:parser.error('No maps match this task; use --map to specify another source')
    if not options['output_dir']:
        if not root or not region:parser.error('With --map, provide --output-dir or both --data and --region')
        kind='height' if options['mode']=='height' else 'vpr'
        options['output_dir']=str(Path(root)/'Crops'/region/kind)
    if options['mode']=='gallery' and not options['centres_dir']:
        centres=Path(options['output_dir'])/'train'
        if centres.is_dir():options['centres_dir']=str(centres)
    if options['mode']=='height' and not options['heights_m']:parser.error('height mode requires --heights-m')
    return options


if __name__=='__main__':
    print(json.dumps(prepare(parse_options()),ensure_ascii=False))
