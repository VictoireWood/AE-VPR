"""Place-score pooling across dates and distance-threshold recall."""
import torch

def rank_places(query, gallery, place_ids, top_k=5):
    if gallery.ndim != 2 or query.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError('Descriptors must be two-dimensional matrices with matching feature dimensions')
    ids = torch.as_tensor(place_ids, device=query.device)
    if ids.ndim != 1 or len(ids) != len(gallery):
        raise ValueError('The number of gallery place IDs does not match the number of images')
    if len(gallery)==0 or not isinstance(top_k,int) or top_k<=0:
        raise ValueError('The gallery must be nonempty and top_k must be a positive integer')
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    scores = query @ gallery.to(query.device).T
    pooled = scores.new_full((len(query),len(unique)),-float('inf'))
    pooled.scatter_reduce_(1,inverse.expand(len(query),-1),scores,reduce='amax',include_self=True)
    order = torch.argsort(pooled,dim=1,descending=True,stable=True)[:,:min(top_k,len(unique))]
    return unique[order],pooled.gather(1,order)

def recalls(distances, thresholds=(25, 50, 100), ranks=(1, 5)):
    d = torch.as_tensor(distances)
    if d.ndim != 2 or len(d) == 0:
        raise ValueError('Distances must be a nonempty [queries, ranks] matrix')
    return {f'R@{k}_{t}m': float((d[:, :k].min(1).values <= t).float().mean()*100)
            for t in thresholds for k in ranks if k <= d.shape[1]}


def localization_distances(reference_xy, query_xy):
    """Euclidean distances in the dataset's fixed UTM zone."""
    import numpy as np
    return np.linalg.norm(np.asarray(reference_xy,dtype=np.float64)-np.asarray(query_xy,dtype=np.float64),axis=-1).tolist()
