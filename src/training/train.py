import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
try:
    import wandb
except ImportError:
    wandb = None

from tqdm import tqdm
from open_clip import Clip_DALoss, get_cast_dtype,ClipLoss,ClipLoss_negclip_distill,ClipLoss_negclip,\
    Clip_DALoss_distill,Clip_DALoss_distill_v2,Clip_DALoss_distill_v3,ClipLoss_my_com_clip,ClipLoss_my_com_clip_distill
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def update_teacher_ema(student_model, teacher_model, alpha=0.996):
    """
    使用学生模型的 EMA 参数更新教师模型。
    
    Args:
        student_model (torch.nn.Module): 学生模型。
        teacher_model (torch.nn.Module): 教师模型。
        alpha (float): EMA 衰减率。
    """
    with torch.no_grad():  # 禁止梯度计算，避免更新教师模型的梯度
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            # 更新教师模型参数为 EMA
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data




def train_one_epoch_person(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss=ClipLoss(local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        is_siglip = 'siglip' in args.model,
        )

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # 初始化监控指标

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        images, texts = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            image_features, text_features, logit_scale  = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)
        backward(total_loss, scaler)
    
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch_ceclip(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.frozen_visual:
        model.module.visual.eval()
    if args.frozen_text:
        model.module.transformer.eval()
        model.module.token_embedding.eval()
        model.module.ln_final.eval()
    if args.extra_da:
        # custom loss function 
        loss = Clip_DALoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            cmr_loss=args.cmr_loss,
            imc_loss=args.imc_loss,
            imc_loss_weight=args.imc_loss_weight,
            cmr_loss_weight=args.cmr_loss_weight,
            threshold_type=args.threshold_type,
            hardnegative=args.hardnegative,
            )
    else:
        loss=ClipLoss(local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            is_siglip = 'siglip' in args.model,
            )

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    # Optimize by directly setting thresholds based on conditions
    if args.cmr_loss:
        # Use a ternary conditional operator to set thresholds based on the type of threshold
        thresholds = args.fixed_threshold_value if args.threshold_type == "fixed" else 0.0
    else:
        # Without cmr loss
        thresholds = None

    cmr_loss=None
    imc_loss=None

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        if args.extra_da:
            images, texts, valid_caption_mask = batch
            valid_caption_mask=valid_caption_mask[:,1:]
            valid_caption_mask=valid_caption_mask.to(device=device, non_blocking=True)
        else:
            images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                image_features, text_features, logit_scale  = model(images, texts)
                if args.extra_da:
                    if args.frozen_text:
                        total_loss, thresholds ,cmr_loss,imc_loss = loss(image_features, text_features.detach(),valid_caption_mask, logit_scale,thresholds)
                    else:
                        total_loss, thresholds ,cmr_loss,imc_loss = loss(image_features, text_features,valid_caption_mask, logit_scale,thresholds)
                    if args.threshold_type!="fixed" and thresholds is not None and args.upper_bound is not None:
                        thresholds=torch.clamp(thresholds,0,args.upper_bound)
                else:
                    total_loss=loss(image_features, text_features, logit_scale)
            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this malocalkes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                    total_loss = loss(image_features, text_features, logit_scale)
                backward(total_loss, scaler)
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"CMR_Loss: {cmr_loss if cmr_loss is not None else None} "
                f"IMC_Loss: {imc_loss if imc_loss is not None else None} "
            )
              
            if args.cmr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                logging.info(f"Thresholds: {thresholds}")
            elif args.cmr_loss:
                # init thresholds for mean cmr loss
                logging.info(f"Threshold_1: {thresholds.mean(0)[0].item():.3f} "
                            f"Threshold_2: {thresholds.mean(0)[1].item():.3f} "
                            f"Threshold_3: {thresholds.mean(0)[2].item():.3f} "
                            f"Threshold_4: {thresholds.mean(0)[3].item():.3f}")

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "CMR_Loss": cmr_loss if cmr_loss is not None else None,
                "IMC_Loss": imc_loss if imc_loss is not None else None
            }
             
            if args.cmr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                log_data['threshold']=thresholds
            elif args.cmr_loss:
                # init thresholds for mean cmr loss
                log_data["threshold_1"]= thresholds.mean(0)[0].item() 
                log_data["threshold_2"]= thresholds.mean(0)[1].item() 
                log_data["threshold_3"]= thresholds.mean(0)[2].item()
                log_data["threshold_4"]= thresholds.mean(0)[3].item()

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
    


def train_one_epoch_ceclip_distill(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None,teacher_model=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.frozen_visual:
        model.module.visual.eval()
    if args.frozen_text:
        model.module.transformer.eval()
    if args.extra_da:
        # custom loss function 
        loss = Clip_DALoss_distill_v3(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            cmr_loss=args.cmr_loss,
            imc_loss=args.imc_loss,
            imc_loss_weight=args.imc_loss_weight,
            cmr_loss_weight=args.cmr_loss_weight,
            threshold_type=args.threshold_type,
            hardnegative=args.hardnegative,
            distill = args.distill,
            )
    else:
        loss=ClipLoss(local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            is_siglip = 'siglip' in args.model,
            )

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
    loss_m = AverageMeter()
    loss_distill_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # Optimize by directly setting thresholds based on conditions
    if args.cmr_loss:
        # Use a ternary conditional operator to set thresholds based on the type of threshold
        thresholds = args.fixed_threshold_value if args.threshold_type == "fixed" else 0.0
    else:
        # Without cmr loss
        thresholds = None
    cmr_loss=None
    imc_loss=None
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        if not args.skip_scheduler:
            scheduler(step)
        if args.extra_da:
            images, texts, valid_caption_mask = batch
            valid_caption_mask=valid_caption_mask[:,1:]
            valid_caption_mask=valid_caption_mask.to(device=device, non_blocking=True)
        else:
            images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.accum_freq == 1:
            with autocast():
                #细粒度蒸馏
                if args.distill_which =='local':
                    s_image_features,s_text_features,logit_scale,s_v_patch_embedding,s_t_token_embedding,token_mask= model(images,texts,get_fine_grained_embedding=True)
                    with torch.no_grad():
                        t_image_features,t_text_features,_,t_v_patch_embedding,t_t_token_embedding,token_mask = teacher_model(images,texts,get_fine_grained_embedding=True)
                else:
                    s_image_features, s_text_features, logit_scale  = model(images, texts)
                    with torch.no_grad():
                        t_image_features,t_text_features,_,= teacher_model(images,texts)

                if args.extra_da:
                    # total_loss, thresholds ,cmr_loss,imc_loss,distill_loss = loss(image_features, text_features,valid_caption_mask, logit_scale,thresholds,intermediate_visual_features,
                    #                                                  intermediate_text_features,frozen_intermediate_visual_features,frozen_intermediate_text_features,args,
                    #                                                  v_patch_embedding,t_token_embedding,v_patch_embedding_teacher,t_token_embedding_teacher)
                    if args.distill_which =='local':
                        total_loss, thresholds ,cmr_loss,imc_loss,distill_loss = loss(s_image_features, s_text_features,
                                                                                      valid_caption_mask, logit_scale,thresholds,
                                                                                      s_v_patch_embedding = s_v_patch_embedding,s_t_token_embedding = s_t_token_embedding,
                                                                                      t_v_patch_embedding = t_v_patch_embedding,t_t_token_embedding = t_t_token_embedding,
                                                                                      t_text_features = t_text_features,t_image_features = t_image_features,
                                                                                      token_mask = token_mask)
                    else:
                        total_loss, thresholds ,cmr_loss,imc_loss,distill_loss = loss(s_image_features, s_text_features,
                                                                                      valid_caption_mask, logit_scale,thresholds,t_image_features = t_image_features,t_text_features = t_text_features)
                                            
                    
                    if args.threshold_type!="fixed" and thresholds is not None and args.upper_bound is not None:
                        thresholds=torch.clamp(thresholds,0,args.upper_bound)
                else:
                    total_loss=loss(image_features, text_features, logit_scale)
            backward(total_loss+args.distill_weight*distill_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)
            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this malocalkes data time logging unreliable when accumulating
                continue
            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                    total_loss = loss(image_features, text_features, logit_scale)
                backward(total_loss, scaler)
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_distill_m.update(distill_loss.item(), batch_size)
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Distill Loss: {loss_distill_m.val:#.5g} ({loss_distill_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"CMR_Loss: {cmr_loss if cmr_loss is not None else None} "
                f"IMC_Loss: {imc_loss if imc_loss is not None else None} "
            )
              
            if args.cmr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                logging.info(f"Thresholds: {thresholds}")
            elif args.cmr_loss:
                # init thresholds for mean cmr loss
                logging.info(f"Threshold_1: {thresholds.mean(0)[0].item():.3f} "
                            f"Threshold_2: {thresholds.mean(0)[1].item():.3f} "
                            f"Threshold_3: {thresholds.mean(0)[2].item():.3f} "
                            f"Threshold_4: {thresholds.mean(0)[3].item():.3f}")

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "CMR_Loss": cmr_loss if cmr_loss is not None else None,
                "IMC_Loss": imc_loss if imc_loss is not None else None
            }
             
            if args.cmr_loss and args.threshold_type=="fixed":
                # init thresholds for fixed threshold
                log_data['threshold']=thresholds
            elif args.cmr_loss:
                # init thresholds for mean cmr loss
                log_data["threshold_1"]= thresholds.mean(0)[0].item() 
                log_data["threshold_2"]= thresholds.mean(0)[1].item() 
                log_data["threshold_3"]= thresholds.mean(0)[2].item()
                log_data["threshold_4"]= thresholds.mean(0)[3].item()

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()




def train_one_epoch_my_com_clip(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()
    loss = ClipLoss_my_com_clip(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        images, texts, hard_captions, = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        hard_captions = hard_captions.to(device=device, non_blocking=True)
        texts = torch.cat([texts, hard_captions])
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)
        backward(total_loss, scaler)
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset() 
def train_one_epoch_my_com_clip_distill(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None,teacher_model = None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()
    loss = ClipLoss_my_com_clip_distill(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        text_local_contrast=args.text_local_contrast,
        text_local_weight=args.text_local_weight,
        image_local_contrast=args.image_local_contrast,
        image_local_weight=args.image_local_weight,
        distill_weight=args.distill_weight,
        distill_mse=args.distill_mse,
        distill_kl = args.distill_kl,
        neg_text=args.neg_text
        )
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    distill_loss_m = AverageMeter()
    text_local_loss_m = AverageMeter()
    image_local_loss_m = AverageMeter()
    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        images, texts, neg_text = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        neg_text = neg_text.to(device=device, non_blocking=True).view(-1, neg_text.shape[-1])
        texts = torch.cat([texts, neg_text])        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            with torch.no_grad():
                t_image_features,t_text_features,_ = teacher_model(images,texts)
            total_loss,text_local_loss,image_local_loss,distill_loss = loss(image_features, text_features, logit_scale,t_image_features,t_text_features)
        backward(total_loss, scaler)
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
        if args.ema_teacher:
            update_teacher_ema(model,teacher_model,alpha=args.ema_alpha)#TODO:ema
        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch
            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            distill_loss_m.update(distill_loss.item(),batch_size)
            text_local_loss_m.update(text_local_loss.item(), batch_size)
            image_local_loss_m.update(image_local_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Distill_Loss: {distill_loss_m.val:#.5g} ({distill_loss_m.avg:#.4g}) "
                f"Text_local_Loss: {text_local_loss_m.val:#.5g} ({text_local_loss_m.avg:#.4g}) "
                f"Image_local_Loss: {image_local_loss_m.val:#.5g} ({image_local_loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()    
def train_one_epoch_negclip(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    #-----train script for negclip
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.frozen_visual:
        model.module.visual.eval()
    if args.frozen_text:
        model.module.transformer.eval()
        model.module.token_embedding.eval()
        model.module.ln_final.eval()
    loss = ClipLoss_negclip(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        # original negclip
        # images, new_images, texts, new_texts, hard_captions, new_hard = batch
        # images = images.to(device=device, non_blocking=True)
        # new_images = new_images.to(device=device, non_blocking=True)
        # texts = texts.to(device=device, non_blocking=True)
        # new_texts = new_texts.to(device=device, non_blocking=True)
        # hard_captions = hard_captions.to(device=device, non_blocking=True)
        # new_hard = new_hard.to(device=device, non_blocking=True)
        # images = torch.cat([images, new_images])
        # texts = torch.cat([texts, new_texts])
        # texts = torch.cat([texts, hard_captions])
        # texts = torch.cat([texts, new_hard])

        # neg_clip_my data
        images, texts, neg_text = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        neg_text = neg_text.to(device=device, non_blocking=True).view(-1, neg_text.shape[-1])
        texts = torch.cat([texts, neg_text])
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)
        backward(total_loss, scaler)
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
             
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

def train_one_epoch_negclip_wo_Image_branch(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    #-----train script for negclip
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.frozen_visual:
        model.module.visual.eval()
    if args.frozen_text:
        model.module.transformer.eval()
        model.module.token_embedding.eval()
        model.module.ln_final.eval()
    loss = ClipLoss_negclip(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        images,  texts, hard_captions = batch

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        hard_captions = hard_captions.to(device=device, non_blocking=True)
        texts = torch.cat([texts, hard_captions])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)
        backward(total_loss, scaler)
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
             
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

def train_one_epoch_negclip_distill(model, data,epoch, optimizer, scaler, scheduler, args, tb_writer=None,teacher_model = None):
    #-----train script for negclip
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.frozen_text:
        model.module.frozen_text()
    loss = ClipLoss_negclip_distill(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        distill_weight=args.distill_weight,
        distill_mse=args.distill_mse,
        distill_kl=args.distill_kl,
        image_local_contrast=args.image_local_contrast,
        image_local_weight=args.image_local_weight,
        text_local_contrast=args.text_local_contrast,
        text_local_weight=args.text_local_weight)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    loss_distill_m = AverageMeter()
    loss_image_local_m = AverageMeter()
    loss_text_local_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
        # images, new_images, texts, new_texts, hard_captions, new_hard = batch
        # images = images.to(device=device, non_blocking=True)
        # new_images = new_images.to(device=device, non_blocking=True)
        # texts = texts.to(device=device, non_blocking=True)
        # new_texts = new_texts.to(device=device, non_blocking=True)
        # hard_captions = hard_captions.to(device=device, non_blocking=True)
        # new_hard = new_hard.to(device=device, non_blocking=True)
        # images = torch.cat([images, new_images])
        # texts = torch.cat([texts, new_texts])
        # texts = torch.cat([texts, hard_captions])
        # texts = torch.cat([texts, new_hard])
        images,texts,neg_text = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        neg_text = neg_text.to(device=device, non_blocking=True).view(-1, neg_text.shape[-1])
        texts = torch.cat([texts, neg_text])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            with torch.no_grad():
                teacher_image_features,teacher_text_features,_ = teacher_model(images,texts)
            total_loss,distill_loss,image_local_loss,text_local_loss = loss(image_features, text_features, logit_scale,teacher_image_features,teacher_text_features)
        backward(total_loss, scaler)
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if isinstance(logit_scale, tuple):
            # for siglip
            logit_scale, logit_bias = logit_scale
        else:
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_distill_m.update(distill_loss.item(),batch_size)
            loss_image_local_m.update(image_local_loss.item(),batch_size)
            loss_text_local_m.update(text_local_loss.item(),batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss_distill: {loss_distill_m.val:#.5g} ({loss_distill_m.avg:#.4g}) "
                f"Loss_image_local: {loss_image_local_m.val:#.5g} ({loss_image_local_m.avg:#.4g}) "
                f"Loss_text_local: {loss_text_local_m.val:#.5g} ({loss_text_local_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
             
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()





def evaluate(model, data, epoch, args, tb_writer=None, is_siglip = False):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    # zeroshot eval imagenet
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating batches with {'siglip' if is_siglip else 'clip'}")):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)

                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    batch_size = images.shape[0]
                    if is_siglip:
                        logit_scale, logit_bias = logit_scale
                        logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
                        
                        n = logits_per_image.size(0)
                        labels = 2 * torch.eye(n, device=device) - torch.ones(n, device = device) 
                        total_loss = -torch.mean(F.logsigmoid(labels * logits_per_image))
                    else:
                        logit_scale = logit_scale.mean()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()

                        labels = torch.arange(batch_size, device=device).long()
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels)
                        ) / 2

                    # gen_loss = maybe_compute_generative_loss(model_out)
                    gen_loss = None

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

            if is_siglip:
                val_metrics = get_siglip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                    logit_bias=logit_bias.cpu(),

                )
            else:
                val_metrics = get_clip_metrics(
                    image_features=torch.cat(all_image_features),
                    text_features=torch.cat(all_text_features),
                    logit_scale=logit_scale.cpu(),
                )
            metrics.update(**val_metrics)
            loss = cumulative_loss / num_samples
            metrics.update(
                {"clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def get_siglip_metrics(image_features, text_features, logit_scale, logit_bias):
    metrics = {}

    logits_per_image = (torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}

    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
