"""Input transforms, frequency mapping, and predicted-altitude cropping."""
import math
import torch
from torchvision import transforms as T

NORMALIZE = T.Normalize([.485, .456, .406], [.229, .224, .225])

def image_tensor(image, size):
    return T.Compose([T.Resize(size, antialias=True), T.ToTensor(), NORMALIZE])(image.convert('RGB'))

def spat2freq(x, base=1.5):
    if base <= 1:
        raise ValueError('base must be greater than 1')
    return torch.log1p(torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1)).abs()) / math.log(base)

def normalize_height(image, predicted_height, reference_height=150.):
    if not math.isfinite(predicted_height) or predicted_height <= 0:
        raise ValueError('Predicted altitude must be finite and positive')
    ratio = min(1., reference_height / predicted_height)
    w, h = image.size
    cw, ch = max(1, round(w * ratio)), max(1, round(h * ratio))
    x, y = (w-cw)//2, (h-ch)//2
    box = (x, y, x+cw, y+ch)
    return image.crop(box), box

def predict_height(features, prototypes):
    similarity = torch.nn.functional.normalize(features.float(), dim=1) @ torch.nn.functional.normalize(prototypes.float(), dim=1).T
    return 125 + 50 * similarity.argmax(1)


def encode_images(model, images, device, amp=True):
    """Use CUDA mixed precision for feature extraction and FP32 for scoring."""
    enabled=amp and torch.device(device).type=='cuda'
    with torch.autocast(device_type=torch.device(device).type,enabled=enabled):
        return model(images.to(device)).float()
