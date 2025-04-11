'''
fine_grained_distill_loss_v1 :kl散度对齐师生logits
'''



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

#---my_com_clip loss
class ClipLoss_pab_eva_clip(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            finetune_weight = 0.0,        
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
        self.finetue_weight = finetune_weight  

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features_pab_eva_clip(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            logits_per_image_ft = logit_scale * all_image_features @ all_text_features.T
            logits_per_text_ft = logit_scale * all_text_features @ all_image_features.T
        else:
            logits_per_image_ft = logit_scale * image_features @ text_features.T
            logits_per_text_ft = logit_scale * text_features @ image_features.T
        num_logits = logits_per_image_ft.shape[0]//2 
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        '''
        hard_text
        '''
        # finetune_loss =(
        #     F.cross_entropy(logits_per_image_ft, labels) +
        #     F.cross_entropy(logits_per_text_ft[:len(logits_per_image_ft)], labels)
        #     ) / 2
        '''
        hard_image
        '''
        # finetune_loss =(
        #     F.cross_entropy(logits_per_image_ft[:num_logits], labels) +
        #     F.cross_entropy(logits_per_text_ft, labels)
        #     ) / 2
        '''
        ft / hard_text+hard_image
        '''        
        finetune_loss =(
            F.cross_entropy(logits_per_image_ft[:num_logits], labels) +
            F.cross_entropy(logits_per_text_ft[:num_logits], labels)
            ) / 2
        return finetune_loss




#---my_com_clip loss
class ClipLoss_my_com_clip(nn.Module):
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
            finetune_weight = 0.0,        
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
        self.finetue_weight = finetune_weight  

    def forward(self, image_features, text_features, logit_scale):
        '''
        默认单卡训练
        '''
        device = image_features.device
        gt_text = text_features[:image_features.shape[0]]

        logits_per_image_ft_coco = logit_scale * image_features @ gt_text.T
        logits_per_text_ft_coco = logit_scale * gt_text @ image_features.T

        # logits_per_image = logit_scale * image_features @ text_features.T
        # logits_per_text = logit_scale * text_features @ image_features.T
        text_local_loss = torch.tensor(0.0,device=device)
        image_local_loss = torch.tensor(0.0,device=device)
        if self.text_local_contrast:
            text_local_loss=self.get_text_local_loss(logit_scale,image_features,text_features)
        if self.image_local_contrast:
            image_local_loss=self.get_image_local_loss(logit_scale,image_features,text_features) 
        
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
        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
        #     ) / 2

        # if self.text_local_contrast:
        #     total_loss+=text_local_loss*self.text_local_weight
        finetune_loss =(
            F.cross_entropy(logits_per_image_ft_coco, labels) +
            F.cross_entropy(logits_per_text_ft_coco, labels)
            ) / 2
        total_loss = finetune_loss* self.finetue_weight
        if self.text_local_contrast:
            total_loss+=text_local_loss*self.text_local_weight
        if self.image_local_contrast:
            # total_loss+=image_local_loss*self.image_local_weight
            total_loss += image_local_loss*self.image_local_weight
        return total_loss,text_local_loss,image_local_loss

    def get_text_local_loss(self,logit_scale,image_features,text_features):
        # 假设 text_features 和 image_features 是预先定义的张量
        # gt_text_embeddings 和 da_text_embeddings 的提取
        gt_text_embeddings = text_features[:image_features.shape[0]]
        da_text_embeddings = text_features[image_features.shape[0]:]

        # 合并 image_features 和 da_text_embeddings
        Im_DA_mix_embeddings = torch.concat([image_features, da_text_embeddings])

        # 计算对比矩阵
        contrast_matrix = logit_scale*gt_text_embeddings @ Im_DA_mix_embeddings.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)

        # 创建掩码
        mask_0 = torch.full((num_logits, num_logits), float('-inf')).to(device=labels.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat_interleave(4, dim=1)
        # 合并掩码
        mask = torch.cat([mask_0, mask_1], dim=1)
        contrast_matrix = contrast_matrix + mask

        # 计算交叉熵损失
        text_local_loss = F.cross_entropy(contrast_matrix, labels)   
        return text_local_loss
    
    def get_image_local_loss(self,logit_scale,image_features,text_features):

        contrast_matrix = logit_scale*image_features @ text_features.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)

        # 创建掩码
        mask_0 = torch.full((num_logits, num_logits), float('-inf')).to(device=labels.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat_interleave(5, dim=1)
        # 合并掩码
        mask = torch.cat([mask_0, mask_1], dim=1)
        contrast_matrix = contrast_matrix + mask

        # 计算交叉熵损失
        image_local_loss = F.cross_entropy(contrast_matrix, labels)
        return image_local_loss


class ClipLoss_my_com_clip_distill(nn.Module):
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
            distill_mse = False,
            distill_kl = False ,
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
        self.distill_mse = distill_mse
        self.distill_kl = distill_kl
        self.distill_weight =  distill_weight
        self.neg_text = neg_text

    def forward(self, image_features, text_features, logit_scale,teacher_image_features,teacher_text_features):
        '''
        '''
        device = image_features.device
        if self.world_size > 1:
            all_image_features,all_text_features = gather_features_negclip(image_features,text_features,self.local_loss,self.gather_with_grad,self.rank,self.world_size,self.use_horovod)
            all_teacher_image_features,all_teacher_text_features = gather_features_negclip(teacher_image_features,teacher_text_features,self.local_loss,self.gather_with_grad,self.rank,self.world_size,self.use_horovod)
            gt_teacher_text = all_teacher_text_features[:all_teacher_image_features.shape[0]]
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_teacher_image_features = teacher_image_features
            all_teacher_text_features = teacher_text_features
            gt_teacher_text = all_teacher_text_features[:all_teacher_image_features.shape[0]]
        #---negCLIP Loss-----
        logits_per_image_ft_coco = logit_scale * all_image_features @ all_text_features.T
        logits_per_text_ft_coco = logit_scale * all_text_features @ all_image_features.T

        # logits_per_image = logit_scale * image_features @ text_features.T
        # logits_per_text = logit_scale * text_features @ image_features.T
        text_local_loss = torch.tensor(0.0,device=device)
        image_local_loss = torch.tensor(0.0,device=device)
        distill_loss = torch.tensor(0.0,device=device)
        if self.text_local_contrast:
            text_local_loss=self.get_text_local_loss(logit_scale,all_text_features,gt_teacher_text)
        if self.image_local_contrast:
            image_local_loss=self.get_image_local_loss(logit_scale,all_image_features,all_text_features) 
        if self.distill_mse:
            distill_loss = self.get_mse_distill_loss(all_image_features,all_text_features,all_teacher_image_features,all_teacher_text_features)
        if self.distill_kl:
            distill_loss = self.get_kl_distill_loss(all_image_features, all_text_features, all_teacher_image_features, all_teacher_text_features)
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
        if self.distill_mse or self.distill_kl:
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

        # 创建掩码
        mask_0 = torch.full((num_logits, num_logits), float('-inf'), device=contrast_matrix.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat(1, self.neg_text)  # 检查这个维度是否正确
        mask = torch.cat([mask_0, mask_1], dim=1)  # 确保 mask 与 contrast_matrix 兼容
        # 应用掩码并计算损失
        contrast_matrix += mask
        text_local_loss = F.cross_entropy(contrast_matrix, labels)
        return text_local_loss
    
    def get_image_local_loss(self,logit_scale,image_features,text_features):

        contrast_matrix = logit_scale*image_features @ text_features.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)

        # 创建掩码
        mask_0 = torch.full((num_logits, num_logits), float('-inf')).to(device=labels.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat_interleave(self.neg_text, dim=1)
        # 合并掩码
        mask = torch.cat([mask_0, mask_1], dim=1)
        contrast_matrix = contrast_matrix + mask

        # 计算交叉熵损失
        image_local_loss = F.cross_entropy(contrast_matrix, labels)
        return image_local_loss

    def get_mse_distill_loss(self,image_features,text_features,t_image_features,t_text_features):

        mse_distill_loss = F.mse_loss(image_features,t_image_features,reduction='sum')+F.mse_loss(text_features,t_text_features,reduction='sum')
        return mse_distill_loss

    def get_kl_distill_loss(self, image_features, text_features, t_image_features, t_text_features, temperature=3.0):
        
        text_features = text_features[:image_features.shape[0]]
        t_text_features = t_text_features[:image_features.shape[0]]
        
        # 学生模型的 logits
        s_logits_per_image = image_features @ text_features.T
        s_logits_per_text = text_features @ image_features.T
        
        # 教师模型的 logits
        t_logits_per_image = t_image_features @ t_text_features.T
        t_logits_per_text = t_text_features @ t_image_features.T
        
        # 使用温度缩放 logits
        s_logits_per_image = s_logits_per_image / temperature
        s_logits_per_text = s_logits_per_text / temperature
        t_logits_per_image = t_logits_per_image / temperature
        t_logits_per_text = t_logits_per_text / temperature

        # 计算教师模型的 softmax 概率分布
        t_probs_per_image = F.softmax(t_logits_per_image, dim=1)
        t_probs_per_text = F.softmax(t_logits_per_text, dim=1)
        
        # 计算学生模型的 log softmax 概率分布
        s_log_probs_per_image = F.log_softmax(s_logits_per_image, dim=1)
        s_log_probs_per_text = F.log_softmax(s_logits_per_text, dim=1)
        
        # 计算 KL 散度损失
        kl_loss_image = F.kl_div(s_log_probs_per_image, t_probs_per_image, reduction='batchmean') * (temperature ** 2)
        kl_loss_text = F.kl_div(s_log_probs_per_text, t_probs_per_text, reduction='batchmean') * (temperature ** 2)
        
        # 总的 KL 散度蒸馏损失
        kl_distill_loss = kl_loss_image + kl_loss_text
        
        return kl_distill_loss



class ClipLoss_negclip(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features_negclip(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0] 
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
            ) / 2
        return total_loss

#
class ClipLoss_negclip_distill(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            distill_weight = 0.0,
            distill_mse = False,
            distill_kl = False,
            image_local_contrast = False,
            image_local_weight = 0.0,
            text_local_contrast = False,
            text_local_weight = 0.0,
            ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.distill_weight = distill_weight
        self.distill_mse = distill_mse
        self.distill_kl = distill_kl
        self.image_local_contrast = image_local_contrast
        self.image_local_weight = image_local_weight
        self.text_local_contrast = text_local_contrast
        self.text_local_weight = text_local_weight
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale,teacher_image_features,teacher_text_features):
        device = image_features.device
        text_local_loss = torch.tensor(0.0,device=device)
        image_local_loss = torch.tensor(0.0,device=device)
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features_negclip(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                
                # distill_logits_per_image = logit_scale * all_CLIP_image_features @ all_image_features.T
                # distill_logits_per_text = logit_scale * all_CLIP_text_features @ all_text_features.T
        else:
            all_image_features = image_features
            all_text_features = text_features
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        # ---------distill mse-------------
        if self.distill_mse:
            distill_loss = self.get_mse_distill_loss(image_features,text_features,teacher_image_features,teacher_text_features)
        elif self.distill_kl:
            distill_loss = self.get_kl_distill_loss(image_features,text_features,teacher_image_features,teacher_text_features)
        #---------distill cross entropy------------- 
        if self.image_local_contrast:
            image_local_loss = self.get_image_local_loss(logit_scale,all_image_features,all_text_features)
                
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
            ) / 2
        if self.distill_mse or self.distill_kl:
            total_loss+=distill_loss*self.distill_weight
        if self.image_local_contrast:
            total_loss+=image_local_loss*self.image_local_weight
        return total_loss,distill_loss,image_local_loss,text_local_loss

    def get_image_local_loss(self,logit_scale,image_features,text_features):

        contrast_matrix = logit_scale*image_features @ text_features.T
        num_logits = contrast_matrix.shape[0]
        labels = torch.arange(num_logits, device=contrast_matrix.device, dtype=torch.long)

        # 创建掩码
        mask_0 = torch.full((num_logits, num_logits), float('-inf')).to(device=labels.device)
        mask_0.fill_diagonal_(0)
        mask_1 = mask_0.repeat_interleave(4, dim=1)
        # 合并掩码
        mask = torch.cat([mask_0, mask_1], dim=1)
        contrast_matrix = contrast_matrix + mask

        # 计算交叉熵损失
        image_local_loss = F.cross_entropy(contrast_matrix, labels)
        return image_local_loss

    def get_mse_distill_loss(self,image_features,text_features,t_image_features,t_text_features):

        text_features = text_features
        t_text_features = t_text_features
        mse_distill_loss = F.mse_loss(image_features,t_image_features,reduction='sum')+F.mse_loss(text_features,t_text_features,reduction='sum')
        return mse_distill_loss

    def get_kl_distill_loss(self, image_features, text_features, t_image_features, t_text_features, temperature=3.0):
        
        # 学生模型的 logits
        s_logits_per_image = image_features @ text_features.T
        s_logits_per_text = text_features @ image_features.T
        
        # 教师模型的 logits
        t_logits_per_image = t_image_features @ t_text_features.T
        t_logits_per_text = t_text_features @ t_image_features.T
        
        # 使用温度缩放 logits
        s_logits_per_image = s_logits_per_image / temperature
        s_logits_per_text = s_logits_per_text / temperature
        t_logits_per_image = t_logits_per_image / temperature
        t_logits_per_text = t_logits_per_text / temperature

        # 计算教师模型的 softmax 概率分布
        t_probs_per_image = F.softmax(t_logits_per_image, dim=1)
        t_probs_per_text = F.softmax(t_logits_per_text, dim=1)
        
        # 计算学生模型的 log softmax 概率分布
        s_log_probs_per_image = F.log_softmax(s_logits_per_image, dim=1)
        s_log_probs_per_text = F.log_softmax(s_logits_per_text, dim=1)
        
        # 计算 KL 散度损失
        kl_loss_image = F.kl_div(s_log_probs_per_image, t_probs_per_image, reduction='batchmean') * (temperature ** 2)
        kl_loss_text = F.kl_div(s_log_probs_per_text, t_probs_per_text, reduction='batchmean') * (temperature ** 2)
        
        # 总的 KL 散度蒸馏损失
        kl_distill_loss = kl_loss_image + kl_loss_text
        
        return kl_distill_loss

    def distillation_loss(self,student_output, teacher_output):

        return F.mse_loss(student_output, teacher_output)


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            is_siglip=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.is_siglip = is_siglip
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        # num_logits equals to batchsize
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.is_siglip:
            logit_scale, logit_bias = logit_scale
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                if self.is_siglip:
                    # siglip 
                    logits_per_image = torch.matmul(image_features, all_text_features.t()) * logit_scale +logit_bias
                    logits_per_text = torch.matmul(text_features, all_image_features.t()) * logit_scale +logit_bias
                
                else:
                    logits_per_image = logit_scale * image_features @ all_text_features.T
                    logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.is_siglip:
                    # siglip 
                    logits_per_image = torch.matmul(all_image_features, all_text_features.t()) * logit_scale + logit_bias
                    logits_per_text = logits_per_image.T
                else:
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
        else:
            if self.is_siglip:
                # siglip 
                logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
                logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale + logit_bias
            else:
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        if self.is_siglip:
            n = logits_per_image.shape[0]
            labels = 2 * torch.eye(n, device=device) - torch.ones(n, device = device) 
            total_loss = -torch.mean(F.logsigmoid(labels * logits_per_image))
        else:    
            labels = self.get_ground_truth(device, logits_per_image.shape[0])

            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss



def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        token_mask = None,
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
            all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
        else:
            if token_mask is not None:
                gathered_token_mask = [torch.zeros_like(token_mask) for _ in range(world_size)]
                dist.all_gather(gathered_token_mask, token_mask)

            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_valid_caption_mask[rank] = valid_caption_mask
                if token_mask is not None:
                    gathered_token_mask[rank] = token_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)
            if token_mask is not None:
                all_token_mask = torch.cat(gathered_token_mask,dim=0)
    if token_mask is not None:
        return all_image_features, all_text_features, all_valid_caption_mask,all_token_mask
    return all_image_features, all_text_features, all_valid_caption_mask
# 
def gather_features_negclip(
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

            # 将文本特征拆分为正样本（前半部分）和负样本（后半部分）
            num_samples = image_features.shape[0]
            positive_text_features = text_features[:num_samples]  # 正样本
            negative_text_features = text_features[num_samples:]  # 负样本

            # 初始化正样本和负样本的列表
            gathered_positive_text_features = [torch.zeros_like(positive_text_features) for _ in range(world_size)]
            gathered_negative_text_features = [torch.zeros_like(negative_text_features) for _ in range(world_size)]

            # 使用 all_gather 分别收集正样本和负样本
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_positive_text_features, positive_text_features)
            dist.all_gather(gathered_negative_text_features, negative_text_features)

            if not local_loss:
                # 确保本地进程的特征保留梯度
                gathered_image_features[rank] = image_features
                gathered_positive_text_features[rank] = positive_text_features
                gathered_negative_text_features[rank] = negative_text_features

            # 将全局的图像特征拼接
            all_image_features = torch.cat(gathered_image_features, dim=0)

            # 将全局的正样本和负样本分别拼接，然后再组合成最终的文本特征
            all_positive_text_features = torch.cat(gathered_positive_text_features, dim=0)
            all_negative_text_features = torch.cat(gathered_negative_text_features, dim=0)

            # 最终将正样本和负样本拼接在一起，前半部分是正样本，后半部分是负样本
            all_text_features = torch.cat([all_positive_text_features, all_negative_text_features], dim=0)

    return all_image_features, all_text_features

def gather_features_pab_eva_clip(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
     # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
# 初始化图像特征的列表
        # gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]

        # 将文本特征拆分为正样本（前半部分）和负样本（后半部分）
        num_samples = image_features.shape[0] // 2  # 假设每个进程的文本特征的前半部分是正样本，后半部分是负样本
        positive_text_features = text_features[:num_samples]  # 正样本
        negative_text_features = text_features[num_samples:]  # 负样本
        positive_image_features = image_features[:num_samples]
        negative_image_features = image_features[num_samples:]

        # 初始化正样本和负样本的列表
        gathered_positive_text_features = [torch.zeros_like(positive_text_features) for _ in range(world_size)]
        gathered_negative_text_features = [torch.zeros_like(negative_text_features) for _ in range(world_size)]
        gathered_positive_image_features = [torch.zeros_like(positive_image_features) for _ in range(world_size)]
        gathered_negative_image_features = [torch.zeros_like(negative_image_features) for _ in range(world_size)]

        # 使用 all_gather 分别收集正样本和负样本
        dist.all_gather(gathered_positive_text_features, positive_text_features)
        dist.all_gather(gathered_negative_text_features, negative_text_features)
        dist.all_gather(gathered_positive_image_features, positive_image_features)
        dist.all_gather(gathered_negative_image_features, negative_image_features)

        if not local_loss:
            # 确保本地进程的特征保留梯度
            gathered_positive_text_features[rank] = positive_text_features
            gathered_negative_text_features[rank] = negative_text_features
            gathered_positive_image_features[rank] = positive_image_features
            gathered_negative_image_features[rank] = negative_image_features

        # 将全局的图像特征拼接
        all_positive_image_features = torch.cat(gathered_positive_image_features, dim=0)
        all_negative_image_features = torch.cat(gathered_negative_image_features, dim=0)

        # 将全局的正样本和负样本分别拼接，然后再组合成最终的文本特征
        all_positive_text_features = torch.cat(gathered_positive_text_features, dim=0)
        all_negative_text_features = torch.cat(gathered_negative_text_features, dim=0)

        # 最终将正样本和负样本拼接在一起，前半部分是正样本，后半部分是负样本
        all_text_features = torch.cat([all_positive_text_features, all_negative_text_features], dim=0)
        all_image_features = torch.cat([all_positive_image_features, all_negative_image_features], dim=0)

    return all_image_features, all_text_features


class Clip_DALoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            threshold_type='mean',
          
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative
        
    def forward(self, image_features, text_features,valid_caption_mask, logit_scale, thresholds):
        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        if isinstance(logit_scale, tuple):
            raise NotImplementedError("siglip not supported")
        device = image_features.device
        cmr_loss,imc_loss=0.0,0.0
        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features=all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size=all_image_features.shape[0],all_image_features.shape[-1]


            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                #extra hard negative loss
                if self.hardnegative:
                    all_text_features=torch.cat([gt_all_text_features,da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T # batch_size * 5xbatch_size       
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                # cross-modal rank loss
                if self.cmr_loss:
                    da_logits_per_image= logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,all_valid_caption_mask,thresholds)
                
                # intra-modal contrastive loss
                if self.imc_loss:
                    text_embedding_matrix=logit_scale * gt_all_text_features @ da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    imc_loss+=self.get_imc_loss(logits_per_image,text_embedding_matrix)

        else:
        # not updating very long time
            gt_len,feature_size=image_features.shape[0],image_features.shape[-1]
            gt_text_features=text_features[:image_features.shape[0]]
            da_text_features=text_features[image_features.shape[0]:]
            all_text_features=torch.cat([gt_text_features,da_text_features])
            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = logit_scale * gt_text_features @ image_features.T
            if self.cmr_loss:
                da_logits_per_image=  logit_scale * (da_text_features.reshape(gt_len,-1,feature_size)@ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,valid_caption_mask,thresholds)
            if self.imc_loss:
                text_embedding_matrix=logit_scale * gt_text_features @ da_text_features.T #(batch_size,4*batch_size)
                imc_loss=self.get_imc_loss(logits_per_image,text_embedding_matrix)         
        num_logits = logits_per_image.shape[0]
        # label_per_image_smooth = torch.zeros((num_logits, 5 * num_logits), device=device, dtype=torch.float32)

        # # 左半部分对角线设置为 0.9
        # label_per_image_smooth[torch.arange(num_logits), torch.arange(num_logits)] = 0.9

        # # 生成右半部分的对角线为 0.1 的矩阵
        # right_half = torch.zeros((num_logits, num_logits), device=device, dtype=torch.float32)
        # right_half[torch.arange(num_logits), torch.arange(num_logits)] = 0.1/4

        # # 将右半部分的每一列复制 4 次
        # right_half_expanded = right_half.repeat_interleave(4, dim=1)

        # # 将生成的矩阵填入到标签矩阵的右半部分
        # label_per_image_smooth[:, num_logits:] = right_half_expanded
       
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        if self.cmr_loss:
            total_loss+=cmr_loss*self.cmr_loss_weight
        if self.imc_loss:
            total_loss+=imc_loss*self.imc_loss_weight
            
        return total_loss,thresholds,cmr_loss,imc_loss
   
        
    def get_cmr_loss(self,gt_logits_per_image:torch.Tensor,da_logits_per_image:torch.Tensor,valid_caption_mask,thresholds:torch.Tensor) -> torch.Tensor:
        # calculating cmr loss
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
        # gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
        cmr_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask


        # updating thresholds
        if self.threshold_type=='mean':
            mask = da_logits_per_image!=0
            average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
            thresholds=(gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
            thresholds=thresholds.detach()
        elif self.threshold_type=='max':
            thresholds,max_indices=(gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
            thresholds=thresholds.expand(gt_similarity.shape)/5
            thresholds=thresholds.detach()
        return cmr_loss.mean(),thresholds

    def get_imc_loss(self,gt_logits_per_image:torch.Tensor,embedding_matrix:torch.Tensor):
        """
        gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,5xbatch_size]
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """
        
        logtis_matrix = embedding_matrix
        labels=torch.zeros(logtis_matrix.shape[0],device=logtis_matrix.device,dtype=torch.long)
        imc_loss=F.cross_entropy(logtis_matrix,labels)
        return imc_loss
        
        

class Clip_DALoss_distill(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            threshold_type='mean',
            distill = False
          
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative


        self.distill = distill
             
    def forward(self, image_features, text_features,valid_caption_mask, logit_scale, thresholds,
                s_v_patch_embedding=None,s_t_token_embedding=None,t_v_patch_embedding=None,t_t_token_embedding= None,token_mask = None,#local distill
                t_image_features = None,t_text_features = None#global distill
                ):


        if s_v_patch_embedding ==None:
            distill_which = 'global'
        else:
            distill_which = 'local' 

        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        if isinstance(logit_scale, tuple):
            raise NotImplementedError("siglip not supported")
        device = image_features.device
        cmr_loss,imc_loss=0.0,0.0

        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features=all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size=all_image_features.shape[0],all_image_features.shape[-1]
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                #extra hard negative loss
                if self.hardnegative:
                    all_text_features=torch.cat([gt_all_text_features,da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T # batch_size * 5xbatch_size 
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T
                # cross-modal rank loss
                if self.cmr_loss:
                    da_logits_per_image= logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,all_valid_caption_mask,thresholds)
                # intra-modal contrastive loss
                if self.imc_loss:
                    text_embedding_matrix=logit_scale * gt_all_text_features @ da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    imc_loss+=self.get_imc_loss(logits_per_image,text_embedding_matrix)
                if self.distill:
                    if distill_which == 'global':
                        distill_loss = self.get_distill_loss_global(image_features,text_features,t_image_features,t_text_features,valid_caption_mask)#全局蒸馏损失
                    elif distill_which =='local':
                        distill_loss = self.get_distill_loss_local(s_v_patch_embedding,s_t_token_embedding,t_v_patch_embedding,t_t_token_embedding,valid_caption_mask,token_mask = token_mask)
        else:
            gt_len,feature_size=image_features.shape[0],image_features.shape[-1]
            gt_text_features=text_features[:image_features.shape[0]]
            da_text_features=text_features[image_features.shape[0]:]
            all_text_features=torch.cat([gt_text_features,da_text_features])
            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = logit_scale * gt_text_features @ image_features.T
            if self.cmr_loss:
                da_logits_per_image=  logit_scale * (da_text_features.reshape(gt_len,-1,feature_size)@ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,valid_caption_mask,thresholds)
            if self.imc_loss:
                text_embedding_matrix=logit_scale * gt_text_features @ da_text_features.T #(batch_size,4*batch_size)
                imc_loss=self.get_imc_loss(logits_per_image,text_embedding_matrix) 
            if self.distill:
                if distill_which == 'global':
                    distill_loss = self.get_distill_loss_global(image_features,text_features,t_image_features,t_text_features,valid_caption_mask)#全局蒸馏损失
                elif distill_which =='local':
                    distill_loss = self.get_distill_loss_local(s_v_patch_embedding,s_t_token_embedding,t_v_patch_embedding,t_t_token_embedding,valid_caption_mask,token_mask = token_mask)
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        if self.cmr_loss:
            total_loss+=cmr_loss*self.cmr_loss_weight
        if self.imc_loss:
            total_loss+=imc_loss*self.imc_loss_weight
        return total_loss,thresholds,cmr_loss,imc_loss,distill_loss
   
        
    def get_cmr_loss(self,gt_logits_per_image:torch.Tensor,da_logits_per_image:torch.Tensor,valid_caption_mask,thresholds:torch.Tensor) -> torch.Tensor:
        # calculating cmr loss
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
        # gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
        cmr_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask


        # updating thresholds
        if self.threshold_type=='mean':
            mask = da_logits_per_image!=0
            average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
            thresholds=(gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
            thresholds=thresholds.detach()
        elif self.threshold_type=='max':
            thresholds,max_indices=(gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
            thresholds=thresholds.expand(gt_similarity.shape)/5
            thresholds=thresholds.detach()
        return cmr_loss.mean(),thresholds

    def get_imc_loss(self,gt_logits_per_image:torch.Tensor,embedding_matrix:torch.Tensor):
        """
        gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,5xbatch_size]
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """
        
        logtis_matrix = embedding_matrix
        labels=torch.zeros(logtis_matrix.shape[0],device=logtis_matrix.device,dtype=torch.long)
        imc_loss=F.cross_entropy(logtis_matrix,labels)
        return imc_loss
    
    def get_distill_loss_global(self,image_features,text_features,t_image_features,t_text_features,valid_caption_mask,alpha = 0.5):
        '''
        text_embedding 和 vision embedding 的全局对齐蒸馏
        '''
        logits_scale = 100
        if self.world_size >1:
            all_image_features, all_text_features, _ = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            caption_types=torch.tensor(([1]*all_image_features.shape[0]+[2]*all_image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size       

            s_logits_per_image = logits_scale * all_image_features @ gt_all_text_features.T
            s_logits_per_text = logits_scale * gt_all_text_features.T @ all_image_features


            all_t_image_features, all_t_text_features, _ = gather_features_da(
                t_image_features, t_text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            caption_types=torch.tensor(([1]*all_t_image_features.shape[0]+[2]*all_t_image_features.shape[0]*4)*self.world_size)
            gt_all_t_text_features=all_t_text_features[caption_types==1] # batch_size * word_size       

            t_logits_per_image = logits_scale * all_t_image_features @ gt_all_t_text_features.T
            t_logits_per_text = logits_scale * gt_all_t_text_features @ all_t_image_features.T

            kl_loss_image = F.kl_div(F.log_softmax(s_logits_per_image, dim=-1),
                                    F.softmax(t_logits_per_image, dim=-1),
                                    reduction='batchmean') 

            # 计算 KL 散度损失：文本-图像蒸馏
            kl_loss_text = F.kl_div(F.log_softmax(s_logits_per_text, dim=-1),
                                    F.softmax(t_logits_per_text, dim=-1),
                                    reduction='batchmean')

            # 最终的蒸馏损失
            distill_loss = alpha * kl_loss_image + (1 - alpha) * kl_loss_text
        else:
            gt_text_features = text_features[:image_features.shape[0]]
            gt_t_text_features = t_text_features[:image_features.shape[0]]

            s_logits_per_image = logits_scale * image_features @ gt_text_features.T
            s_logits_per_text = logits_scale * gt_text_features.T @ image_features
            t_logits_per_image = logits_scale * t_image_features @ gt_t_text_features.T
            t_logits_per_text = logits_scale * gt_t_text_features.T @ t_image_features
            kl_loss_image = F.kl_div(F.log_softmax(s_logits_per_image, dim=-1),
                                    F.softmax(t_logits_per_image, dim=-1),
                                    reduction='batchmean') 

            # 计算 KL 散度损失：文本-图像蒸馏
            kl_loss_text = F.kl_div(F.log_softmax(s_logits_per_text, dim=-1),
                                    F.softmax(t_logits_per_text, dim=-1),
                                    reduction='batchmean')

            # 最终的蒸馏损失
            distill_loss = alpha * kl_loss_image + (1 - alpha) * kl_loss_text

        return distill_loss

    def get_distill_loss_local(self,v_patch_embedding,t_token_embedding,v_patch_embedding_teacher,
                               t_token_embedding_teacher,valid_caption_mask,alpha = 0.5,
                               token_mask = None):
        '''
        最后一层的embedding做细粒度的对齐蒸馏
        '''
        if self.world_size > 1:
            all_v_patch_embedding, all_t_token_embedding, _,all_token_mask = gather_features_da(
                v_patch_embedding, t_token_embedding, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod,token_mask=token_mask)
            caption_types=torch.tensor(([1]*v_patch_embedding.shape[0]+[2]*v_patch_embedding.shape[0]*4)*self.world_size)
            gt_all_t_token_embedding=all_t_token_embedding[caption_types==1] # batch_size * word_size        
            all_token_mask = all_token_mask[caption_types==1]


            all_v_patch_embedding_teacher, all_t_token_embedding_teacher, _ ,_= gather_features_da(
                v_patch_embedding_teacher, t_token_embedding_teacher, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod,token_mask=token_mask)
            gt_all_t_token_embedding_teacher=all_t_token_embedding_teacher[caption_types==1] # batch_size * word_size        

            #------修订版
            logits_s_t2v,logits_s_v2t = self.get_logits_fine_grained(all_v_patch_embedding,gt_all_t_token_embedding,all_token_mask)
            logits_t_t2v,logits_t_v2t = self.get_logits_fine_grained(all_v_patch_embedding_teacher,gt_all_t_token_embedding_teacher,all_token_mask)
            
            kl_loss_t2v = F.kl_div(F.log_softmax(logits_s_t2v, dim=-1),
                                    F.softmax(logits_t_t2v, dim=-1),
                                    reduction='batchmean') 

            # 计算 KL 散度损失：文本-图像蒸馏
            kl_loss_v2t = F.kl_div(F.log_softmax(logits_s_v2t, dim=-1),
                                    F.softmax(logits_t_v2t, dim=-1),
                                    reduction='batchmean')

            # 最终的蒸馏损失
            distill_loss = alpha * kl_loss_t2v + (1 - alpha) * kl_loss_v2t
        else:
            gt_t_token_embedding=t_token_embedding[:v_patch_embedding.shape[0]]
            token_mask = token_mask[:v_patch_embedding.shape[0]]
            gt_t_token_embedding_teacher = t_token_embedding_teacher[:v_patch_embedding.shape[0]]
            #------修订版
            logits_s_t2v,logits_s_v2t = self.get_logits_fine_grained(v_patch_embedding,gt_t_token_embedding,token_mask)
            logits_t_t2v,logits_t_v2t = self.get_logits_fine_grained(v_patch_embedding_teacher,gt_t_token_embedding_teacher,token_mask)
            

            tempture = 1
            kl_loss_t2v = F.kl_div(F.log_softmax(logits_s_t2v, dim=-1),
                                    F.softmax(logits_t_t2v, dim=-1),
                                    reduction='batchmean') **tempture**2

            # 计算 KL 散度损失：文本-图像蒸馏
            kl_loss_v2t = F.kl_div(F.log_softmax(logits_s_v2t, dim=-1),
                                    F.softmax(logits_t_v2t, dim=-1),
                                    reduction='batchmean') ** tempture**2

            # 最终的蒸馏损失
            distill_loss = alpha * kl_loss_t2v + (1 - alpha) * kl_loss_v2t

        return distill_loss

        #-----第一种方式，计算两个师生网络在细粒度对齐损失的kl散度

    def get_logits_fine_grained(self,v_patch_embedding,t_token_embedding,token_mask):
        inverse_temp = 100
        #计算相似度
        similarity = torch.einsum('btd,bpd->btp',t_token_embedding,v_patch_embedding)#TODO:将text_token的梯度断掉
        #min-max归一化
        similarity_min, _ = torch.min(similarity, dim=-1, keepdim=True)
        similarity_max, _ = torch.max(similarity, dim=-1, keepdim=True)

        similarity = (similarity - similarity_min)/(similarity_max-similarity_min+1e-8)
        threshold = 1./similarity.shape[2]
        # similarity = torch.where(similarity< threshold,0.0,similarity)
        similarity[similarity<threshold] = 0.
        v_align_weights = similarity/(torch.sum(similarity,dim = -1,keepdim=True)+1e-8)
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd',v_align_weights.detach(),v_patch_embedding)
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed,p=2,dim=-1)
        t_token_embedding = F.normalize(t_token_embedding,p=2,dim=-1)
        # #------with token_mask
        t_token_embedding = t_token_embedding[token_mask]
        l_grouped_v_patch_embed = l_grouped_v_patch_embed[token_mask]
        #------

        #---without token_mask
        # t_token_embedding = t_token_embedding.reshape(-1, t_token_embedding.shape[-1])
        # l_grouped_v_patch_embed = l_grouped_v_patch_embed.reshape(-1,l_grouped_v_patch_embed.shape[-1])

        logits_t2v = inverse_temp * t_token_embedding @ l_grouped_v_patch_embed.T
        logits_v2t = inverse_temp * l_grouped_v_patch_embed @ t_token_embedding.T
        return logits_t2v,logits_v2t



        
    def write_similarity(self,args,inter_visual_features,inter_text_features,frozen_visual_features,frozen_text_features):

        visual_file = os.path.join(args.logs,args.name,'visual_similarity.txt')
        text_file = os.path.join(args.logs,args.name,'text_similarity.txt')
        if self.world_size > 1:
            all_inter_visual_features, all_inter_text_features = zip(
                *[gather_features(image_features=inter_visual_features[i].contiguous(), text_features=inter_text_features[i].contiguous(),rank=self.rank,world_size=self.world_size) for i in range(len(inter_visual_features))]
            )
            all_frozen_inter_visual_features, all_frozen_inter_text_features = zip(
                *[gather_features(image_features=frozen_visual_features[i].contiguous(), text_features=frozen_text_features[i].contiguous(),rank=self.rank,world_size=self.world_size) for i in range(len(frozen_visual_features))]
            )

            # 标准化特征
            intermediate_visual_features, frozen_intermediate_visual_features = extract_and_normalize(
                all_inter_visual_features, all_frozen_inter_visual_features
            )
            intermediate_text_features, frozen_intermediate_text_features = extract_and_normalize(
                all_inter_text_features, all_frozen_inter_text_features
            )

            # 计算相似度
            visual_similarity = []
            text_similarity = []
            visual_similarity_str = ""
            text_similarity_str = ""

            for vis_feat, frozen_vis_feat, text_feat, frozen_text_feat in zip(
                intermediate_visual_features, frozen_intermediate_visual_features,
                intermediate_text_features, frozen_intermediate_text_features
            ):
                # 计算并添加视觉和文本相似度
                visual_similarity.append(F.cosine_similarity(vis_feat, frozen_vis_feat).mean())
                text_similarity.append(F.cosine_similarity(text_feat, frozen_text_feat).mean())
                # 构建相似度字符串
                visual_similarity_str += f"{visual_similarity[-1]:.4f}   "
                text_similarity_str += f"{text_similarity[-1]:.4f}   "

            # 仅在主进程（rank 0）中进行文件写入操作
            if self.rank == 0:
                with open(visual_file, 'a') as vf, open(text_file, 'a') as tf:
                    vf.write(visual_similarity_str.strip() + "\n")
                    tf.write(text_similarity_str.strip() + "\n")
        else:
            intermediate_visual_features, frozen_intermediate_visual_features = extract_and_normalize(
            inter_visual_features, frozen_visual_features
            )
            intermediate_text_features, frozen_intermediate_text_features = extract_and_normalize(
                inter_text_features, frozen_text_features
            )

            # 计算相似度
            visual_similarity = []
            text_similarity = []
            visual_similarity_str = ""
            text_similarity_str = ""

            for vis_feat, frozen_vis_feat, text_feat, frozen_text_feat in zip(
                intermediate_visual_features, frozen_intermediate_visual_features,
                intermediate_text_features, frozen_intermediate_text_features):
                # 计算并添加视觉和文本相似度
                visual_similarity.append(F.cosine_similarity(vis_feat, frozen_vis_feat).mean())
                text_similarity.append(F.cosine_similarity(text_feat, frozen_text_feat).mean())
                # 构建相似度字符串
                visual_similarity_str += f"{visual_similarity[-1]:.4f}   "
                text_similarity_str += f"{text_similarity[-1]:.4f}   "

            # 仅在主进程（rank 0）中进行文件写入操作
            if self.rank == 0:
                with open(visual_file, 'a') as vf, open(text_file, 'a') as tf:
                    vf.write(visual_similarity_str.strip() + "\n")
                    tf.write(text_similarity_str.strip() + "\n")              
