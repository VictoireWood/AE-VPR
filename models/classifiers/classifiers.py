
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def _resolve_dims(in_features=None, out_features=None, embedding_size=None, classnum=None):
    """Support both local and AdaFace-style constructor argument names."""
    if in_features is None:
        in_features = embedding_size
    if out_features is None:
        out_features = classnum
    if in_features is None or out_features is None:
        raise TypeError("classifier dimensions must be provided")
    return in_features, out_features


# Based on https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py#L10
class AAMC(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        """Implementation of Additive Angular Margin Classifier:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
    def forward(self, input, label):    # NOTE 这里的input就对应head/metrics里的embedding
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m   # NOTE 三角函数和角公式
        phi = torch.where(cosine > 0, phi, cosine)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s    # NOTE 相当于交叉熵损失，也就是-log的部分
        # print(output)
        
        return output, cosine


# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py
class LMCC(nn.Module):
    def __init__(self, in_features=None, out_features=None, s=30.0, m=0.40, embedding_size=None, classnum=None):
        """Implementation of Large Margin Cosine Classifier:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
        """
        super().__init__()
        in_features, out_features = _resolve_dims(in_features, out_features, embedding_size, classnum)
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    @staticmethod
    def cosine_sim(x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps), ip
    
    def forward(self, input, label):
        cosine, prod = self.cosine_sim(input, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output, prod
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class LinearLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, embedding_size=None, classnum=None):
        super().__init__()
        in_features, out_features = _resolve_dims(in_features, out_features, embedding_size, classnum)
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, inputs, label=None):
        """param 'label' is not used, but having it makes it easier to replace LMCC with
        LinearLayer in other parts of the code, instead of having lots of if-else"""
        output = self.linear(inputs)
        return output, output

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

# based on https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py
class ACLC(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weight, std=0.01)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer('t', torch.zeros(1))

    def forward(self, input, label):
        input = l2_norm(input, axis = 1)
        weight_norm = l2_norm(self.weight, axis = 0)
        cos_theta = torch.mm(input, weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, input.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s
    


    '''class ACLC(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s'''
    

# AdaFace

class QAMC(nn.Module):
    def __init__(self,
                 embedding_size=None,
                 classnum=None,
                 in_features=None,
                 out_features=None,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 quality_alpha=0.5,
                 ):
        super().__init__()
        embedding_size, classnum = _resolve_dims(in_features, out_features, embedding_size, classnum)
        self.classnum = classnum
        self.weight = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.quality_alpha = quality_alpha

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        # print('\n\AdaFace with the following property')
        # print('self.m', self.m)
        # print('self.h', self.h)
        # print('self.s', self.s)
        # print('self.t_alpha', self.t_alpha)

    @staticmethod
    def laplacian_sharpness(images):
        """Return per-image Laplacian-variance sharpness scores."""
        if images is None:
            return None
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        gray = images.detach().float().mean(dim=1, keepdim=True)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        response = F.conv2d(gray, kernel, padding=1)
        return response.var(dim=(2, 3), unbiased=False, keepdim=False).view(-1, 1)

    def _quality_factor(self, norms, images=None):
        safe_norms = torch.clip(norms, min=0.001, max=100).detach()
        if images is None:
            return safe_norms

        sharpness = torch.log1p(self.laplacian_sharpness(images)).detach()
        norm_z = (safe_norms - safe_norms.mean()) / (safe_norms.std(unbiased=False) + self.eps)
        sharp_z = (sharpness - sharpness.mean()) / (sharpness.std(unbiased=False) + self.eps)
        return self.quality_alpha * norm_z + (1.0 - self.quality_alpha) * sharp_z

    def forward(self, embbedings, label, images=None):

        norms = torch.norm(embbedings, 2, 1, True)
        embbedings = torch.div(embbedings, norms)

        kernel_norm = l2_norm(self.weight,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        quality = self._quality_factor(norms, images)

        # update batchmean batchstd
        with torch.no_grad():
            mean = quality.mean().detach()
            std = quality.std(unbiased=False).detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (quality - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1).long(), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1).long(), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, cosine
