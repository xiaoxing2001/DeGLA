
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import os


def extract_and_normalize(features, frozen_features):
    normalized_features = [F.normalize(f.detach(), dim=1) for f in features]
    frozen_normalized_features = [F.normalize(f.detach(), dim=1) for f in frozen_features]
    return normalized_features, frozen_normalized_features

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features#把文本特征劈开两半,然后再拼接
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
class CLIPloss_DeGLA(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            text_local_contrast = False,
            text_local_weight = 0.0,
            image_local_contrast = False,
            image_local_weight = 0.0,
            distill_weight = 0.0,
            neg_text = 4          
            ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}
        self.text_local_contrast = text_local_contrast
        self.text_local_weight = text_local_weight
        self.image_local_contrast = image_local_contrast
        self.image_local_weight = image_local_weight
        self.distill_weight =  distill_weight
        self.neg_text = neg_text
    def forward(self, image_features, text_features, logit_scale,teacher_image_features,teacher_text_features):
        device = image_features.device
        if self.world_size > 1:
            all_image_features,all_text_features = gather_features_degla(image_features,text_features,self.local_loss,self.gather_with_grad,self.rank,self.world_size,self.use_horovod)
            all_teacher_image_features,all_teacher_text_features = gather_features_degla(teacher_image_features,teacher_text_features,self.local_loss,self.gather_with_grad,self.rank,self.world_size,self.use_horovod)
            gt_teacher_text = all_teacher_text_features[:all_teacher_image_features.shape[0]]
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_teacher_image_features = teacher_image_features
            all_teacher_text_features = teacher_text_features
            gt_teacher_text = all_teacher_text_features[:all_teacher_image_features.shape[0]]
        logits_per_image_ft_coco = logit_scale * all_image_features @ all_text_features.T
        logits_per_text_ft_coco = logit_scale * all_text_features @ all_image_features.T
        text_local_loss = torch.tensor(0.0,device=device)
        image_local_loss = torch.tensor(0.0,device=device)
        distill_loss = torch.tensor(0.0,device=device)
        if self.text_local_contrast:
            text_local_loss=self.get_text_local_loss(logit_scale,all_text_features,gt_teacher_text)
        if self.image_local_contrast:
            image_local_loss=self.get_image_local_loss(logit_scale,all_image_features,all_text_features) 
        if self.distill_weight>0.0:
            distill_loss = self.get_mse_distill_loss(all_image_features,all_text_features,all_teacher_image_features,all_teacher_text_features)
        num_logits = logits_per_image_ft_coco.shape[0] 
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss =(
            F.cross_entropy(logits_per_image_ft_coco, labels) +
            F.cross_entropy(logits_per_text_ft_coco[:len(logits_per_image_ft_coco)], labels)
            ) / 2
        if self.text_local_contrast:
            total_loss+=text_local_loss*self.text_local_weight
        if self.image_local_contrast:
            total_loss += image_local_loss*self.image_local_weight
        if self.distill_weight>0.0:
            total_loss+= distill_loss*self.distill_weight
        return total_loss,text_local_loss,image_local_loss,distill_loss
    def get_text_local_loss(self,logit_scale,text_features,frozen_texts):
        logit_scale = 10.0
        gt_text = text_features[:frozen_texts.shape[0]]
        da_text = text_features[frozen_texts.shape[0]:]
        text_embeddings = torch.cat([frozen_texts, da_text])
        contrast_matrix =  logit_scale * gt_text @ text_embeddings.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)
        mask_0 = torch.full((num_logits, num_logits), float('-inf'), device=contrast_matrix.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat(1, self.neg_text)  
        mask = torch.cat([mask_0, mask_1], dim=1) 
        contrast_matrix += mask
        text_local_loss = F.cross_entropy(contrast_matrix, labels)
        return text_local_loss
    def get_image_local_loss(self,logit_scale,image_features,text_features):

        contrast_matrix = logit_scale*image_features @ text_features.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)
        mask_0 = torch.full((num_logits, num_logits), float('-inf')).to(device=labels.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat_interleave(self.neg_text, dim=1)
        mask = torch.cat([mask_0, mask_1], dim=1)
        contrast_matrix = contrast_matrix + mask
        image_local_loss = F.cross_entropy(contrast_matrix, labels)
        return image_local_loss
    def get_mse_distill_loss(self,image_features,text_features,t_image_features,t_text_features):
        mse_distill_loss = F.mse_loss(image_features,t_image_features,reduction='sum')+F.mse_loss(text_features,t_text_features,reduction='sum')
        return mse_distill_loss
def gather_features_degla(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]

            num_samples = image_features.shape[0]
            positive_text_features = text_features[:num_samples] 
            negative_text_features = text_features[num_samples:] 
            gathered_positive_text_features = [torch.zeros_like(positive_text_features) for _ in range(world_size)]
            gathered_negative_text_features = [torch.zeros_like(negative_text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_positive_text_features, positive_text_features)
            dist.all_gather(gathered_negative_text_features, negative_text_features)

            if not local_loss:
                gathered_image_features[rank] = image_features
                gathered_positive_text_features[rank] = positive_text_features
                gathered_negative_text_features[rank] = negative_text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_positive_text_features = torch.cat(gathered_positive_text_features, dim=0)
            all_negative_text_features = torch.cat(gathered_negative_text_features, dim=0)
            all_text_features = torch.cat([all_positive_text_features, all_negative_text_features], dim=0)

    return all_image_features, all_text_features

