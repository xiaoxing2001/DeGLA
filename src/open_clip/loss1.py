
'''
fine_grained_distill_loss_v2 :kl散度对齐师生logits + 文本的mse
motivation? 文本侧防止进一步坍缩
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
        

class Clip_DALoss_distill_v2(nn.Module):

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

            _, all_text_features_teacher, _ = gather_features_da(
                t_image_features, t_text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            gt_all_text_features_teacher=all_text_features_teacher[caption_types==1] # batch_size * word_size


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
                        distill_loss = self.get_distill_loss_local(s_v_patch_embedding,s_t_token_embedding,t_v_patch_embedding,t_t_token_embedding,valid_caption_mask,token_mask = token_mask,
                                                                   t_text_features=gt_all_text_features_teacher,s_text_features=gt_all_text_features)
        else:
            gt_len,feature_size=image_features.shape[0],image_features.shape[-1]
            gt_text_features=text_features[:image_features.shape[0]]
            gt_text_features_teacher = t_text_features[:image_features.shape[0]]
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
                    distill_loss = self.get_distill_loss_local(s_v_patch_embedding,s_t_token_embedding,t_v_patch_embedding,t_t_token_embedding,valid_caption_mask,token_mask = token_mask,
                                                               t_text_features=gt_text_features_teacher,s_text_features=gt_text_features)
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
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size       

            # s_logits_per_image = logits_scale * all_image_features @ gt_all_text_features.T
            # s_logits_per_text = logits_scale * gt_all_text_features.T @ all_image_features


            all_t_image_features, all_t_text_features, _ = gather_features_da(
                t_image_features, t_text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_t_text_features=all_t_text_features[caption_types==1] # batch_size * word_size       

            # t_logits_per_image = logits_scale * all_t_image_features @ gt_all_t_text_features.T
            # t_logits_per_text = logits_scale * gt_all_t_text_features @ all_t_image_features.T

            # kl_loss_image = F.kl_div(F.log_softmax(s_logits_per_image, dim=-1),
            #                         F.softmax(t_logits_per_image, dim=-1),
            #                         reduction='batchmean') 

            # # 计算 KL 散度损失：文本-图像蒸馏
            # kl_loss_text = F.kl_div(F.log_softmax(s_logits_per_text, dim=-1),
            #                         F.softmax(t_logits_per_text, dim=-1),
            #                         reduction='batchmean')

            # 最终的蒸馏损失
            # distill_loss = alpha * kl_loss_image + (1 - alpha) * kl_loss_text
            distill_loss = F.mse_loss(all_image_features,all_t_image_features)+F.mse_loss(all_text_features,all_t_text_features)
        else:
            gt_text_features = text_features[:image_features.shape[0]]
            gt_t_text_features = t_text_features[:image_features.shape[0]]

            # s_logits_per_image = logits_scale * image_features @ gt_text_features.T
            # s_logits_per_text = logits_scale * gt_text_features.T @ image_features
            # t_logits_per_image = logits_scale * t_image_features @ gt_t_text_features.T
            # t_logits_per_text = logits_scale * gt_t_text_features.T @ t_image_features
            # kl_loss_image = F.kl_div(F.log_softmax(s_logits_per_image, dim=-1),
            #                         F.softmax(t_logits_per_image, dim=-1),
            #                         reduction='batchmean') 

            # # 计算 KL 散度损失：文本-图像蒸馏
            # kl_loss_text = F.kl_div(F.log_softmax(s_logits_per_text, dim=-1),
            #                         F.softmax(t_logits_per_text, dim=-1),
            #                         reduction='batchmean')

            # # 最终的蒸馏损失
            # distill_loss = alpha * kl_loss_image + (1 - alpha) * kl_loss_text
            # distill_loss = F.mse_loss(image_features,t_image_features) +F.mse_loss()
            

        return distill_loss

    def get_distill_loss_local(self,v_patch_embedding,t_token_embedding,v_patch_embedding_teacher,
                               t_token_embedding_teacher,valid_caption_mask,alpha = 0.5,
                               token_mask = None,t_text_features = None,s_text_features = None):
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
            
            tempture = 5

            kl_loss_t2v = F.kl_div(F.log_softmax(logits_s_t2v/tempture, dim=-1),
                                    F.softmax(logits_t_t2v/tempture, dim=-1),
                                    reduction='batchmean') * tempture**2

            # 计算 KL 散度损失：文本-图像蒸馏
            kl_loss_v2t = F.kl_div(F.log_softmax(logits_s_v2t/tempture, dim=-1),
                                    F.softmax(logits_t_v2t/tempture, dim=-1),
                                    reduction='batchmean')*tempture**2

            # 最终的蒸馏损失
            distill_loss = alpha * kl_loss_t2v + (1 - alpha) * kl_loss_v2t
            # distill_loss = F.mse_loss(s_text_features,t_text_features)
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
            distill_loss = alpha * kl_loss_t2v + (1 - alpha) * kl_loss_v2t+F.mse_loss(s_text_features,t_text_features)

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