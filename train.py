import os
import math
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
import lpips
from copy import deepcopy
from time import time
import argparse
import yaml
from utils.utils import (
    novel_view_by_reprojection,
    center_crop_arr,
    setup_logging,
    get_infinite_loader,
    get_parameter_groups,
    build_scheduler,
    select_gan_losses,
    calculate_adaptive_weight,
    update_ema,
    get_recon_loss_fn,
    save_checkpoint,
)
from disc import build_discriminator
from utils.pose_enc import extri_intri_to_pose_encoding
from models.models import OVIE_models
from moge2.moge.model.v2 import MoGeModel
from data import ImageOnlyDataset, PreprocessedDataset


from utils.reprojections import ExtrinsicsSamplingRouter
from transformers import AutoModel

torch.set_float32_matmul_precision("high")
torch._functorch.config.donated_buffer = False


def build_training_targets(
    images,
    points3d,
    intrinsics_cam0,
    mask,
    normals,
    extrinsics_sampler,
    config,
    nvs,
    device,
):
    extrinsics_cam01, extrinsics_cam10, sampling_methods = extrinsics_sampler(
        points3d, nvs, normals.reshape(normals.shape[0], -1, 3), intrinsics_cam0
    )
    intrinsics_cam1 = intrinsics_cam0

    cam01_params = extri_intri_to_pose_encoding(
        extrinsics_cam01,
        intrinsics_cam1,
        image_size_hw=(config["data"]["image_size"], config["data"]["image_size"]),
    )[:, :7]

    cam10_params = extri_intri_to_pose_encoding(
        extrinsics_cam10,
        intrinsics_cam0,
        image_size_hw=(config["data"]["image_size"], config["data"]["image_size"]),
    )[:, :7]

    images_cam1_target, images_cam1_target_visibility_mask = novel_view_by_reprojection(
        points3d.to(device),
        images.to(device),
        intrinsics_cam1.to(device),
        extrinsics_cam10.to(device),
        splat_size=nvs["splat_size"],
        valid_mask=mask,
        normals_world=normals.permute(0, 3, 1, 2).to(device),
    )

    return (
        images_cam1_target.to(device),
        images_cam1_target_visibility_mask.to(device),
        cam01_params.to(device),
        cam10_params.to(device),
    )


@torch.no_grad()
def validate(
    model, depth_estimator, val_loader, extrinsics_sampler, config, nvs, device
):
    recon_loss_fn = get_recon_loss_fn(config)

    total_loss_sum = torch.tensor(0.0, device=device)
    recon_loss_sum = torch.tensor(0.0, device=device)
    total_samples_tensor = torch.tensor(0.0, device=device)
    for images, _ in val_loader:
        images_cam0 = images.to(device)
        batch_size = images.size(0)

        depth_estimation = depth_estimator.infer(images_cam0)
        points3d_cam0 = (
            depth_estimation["points"].to(device).reshape(images_cam0.shape[0], -1, 3)
        )
        intrinsics_cam0 = depth_estimation["intrinsics"].to(device)
        masks_cam0 = depth_estimation["mask"].to(device)
        normals_cam0 = depth_estimation["normal"].to(device)

        images_cam1_target, images_cam1_target_visibility_mask, cam01_params, _ = (
            build_training_targets(
                images,
                points3d_cam0,
                intrinsics_cam0,
                masks_cam0,
                normals_cam0,
                extrinsics_sampler,
                config,
                nvs,
                device,
            )
        )

        with torch.amp.autocast("cuda", enabled=config["train"]["amp"]["enabled"]):
            images_cam1_pred = model(x=images_cam0, cam_params=cam01_params)
            novel_view_recon_loss = recon_loss_fn(
                images_cam1_pred,
                images_cam1_target,
                mask=images_cam1_target_visibility_mask,
            )

        recon_loss_sum += novel_view_recon_loss * batch_size
        total_loss_sum += novel_view_recon_loss * batch_size
        total_samples_tensor += batch_size

    if dist.is_initialized():
        for t in [total_loss_sum, recon_loss_sum, total_samples_tensor]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss_avg = total_loss_sum / total_samples_tensor
    val_loss_recon = recon_loss_sum / total_samples_tensor

    images_cam1_pred = torch.clamp(images_cam1_pred, 0, 1)
    return (
        val_loss_avg.item(),
        val_loss_recon.item(),
        images,
        images_cam1_target,
        images_cam1_pred,
    )


def train(config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    cudnn.benchmark = True

    seed = config["train"]["seed"] if "seed" in config["train"] else 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose(
        [
            lambda img: center_crop_arr(img, config["data"]["image_size"]),
            transforms.ToTensor(),
        ]
    )
    train_dataset = PreprocessedDataset(
        data_dirs=config["data"]["data_path"],
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["global_batch_size"] // world_size,
        sampler=train_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = ImageOnlyDataset(
        root=config["data"]["val_data_path"], transform=transform
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["global_batch_size"] // world_size,
        sampler=val_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    total_steps = config["train"]["num_training_steps"]

    model = OVIE_models[config["model"]["model_type"]](
        image_size=256,
        vit_use_qknorm=config["model"].get("use_qknorm", False),
        vit_use_swiglu=config["model"].get("use_swiglu", True),
        vit_use_rope=config["model"].get("use_rope", False),
        vit_use_rmsnorm=config["model"].get("use_rmsnorm", True),
        vit_wo_shift=config["model"].get("wo_shift", False),
        vit_use_checkpoint=config["model"].get("use_checkpoint", False),
    ).to(device)
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    gan_config = config.get("gan", {})
    disc_config = gan_config.get("disc", {})
    disc_optimizer_config = disc_config.get("optimizer", {})
    disc_loss_config = gan_config.get("loss", {})

    discriminator, disc_aug = build_discriminator(disc_config, device)
    discriminator = discriminator.to(device)
    if disc_config.get("compile", False):
        discriminator = torch.compile(discriminator)
    discriminator = DDP(
        discriminator,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(
        disc_kind=disc_loss_config.get("disc_loss", "hinge"),
        gen_kind=disc_loss_config.get("gen_loss", "vanilla"),
    )

    depth_estimator = MoGeModel.from_pretrained(
        config["depth_estimator"]["model_name"]
    ).to(device)
    for p in depth_estimator.parameters():
        p.requires_grad_(False)
    depth_estimator.eval()
    if config["depth_estimator"]["compile"]:
        depth_estimator = torch.compile(depth_estimator)

    if config["model"].get("compile", False):
        model = torch.compile(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    model_woddp = model.module
    last_layer = model.module.final_conv.weight

    weight_decay = config["train"].get("weight_decay", 0.0)
    if weight_decay == 0:
        param_groups = model.parameters()
    else:
        param_groups = get_parameter_groups(
            model,
            weight_decay=weight_decay,
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config["train"]["lr"],
        weight_decay=weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        disc_params,
        lr=disc_optimizer_config.get("lr", 2e-4),
        betas=disc_optimizer_config.get("betas", (0.9, 0.95)),
        weight_decay=disc_optimizer_config.get("weight_decay", 0.0),
    )

    scheduler = build_scheduler(
        optimizer, config["train"].get("scheduler", {}), total_steps
    )
    disc_true_steps = int(
        total_steps - disc_loss_config.get("disc_upd_start_ratio", 0.0) * total_steps
    )
    disc_scheduler = build_scheduler(
        disc_optimizer, disc_config.get("scheduler", {}), disc_true_steps
    )

    lpips_model = None
    if config["train"].get("lpips_weight", 0.0) > 0:
        lpips_model = lpips.LPIPS(net="vgg").to(device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)
        if config["model"].get("compile", False):
            lpips_model = torch.compile(lpips_model)

    dino_model = None
    if config["train"].get("dino_perceptual_loss_weight", 0.0) > 0:
        dino_model = AutoModel.from_pretrained(
            config["train"].get(
                "dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m"
            )
        )
        dino_model.to(device)
        dino_model.eval()
        if config["model"].get("compile", False):
            dino_model = torch.compile(dino_model)

    scaler_gen = torch.amp.GradScaler("cuda", enabled=config["train"]["amp"]["enabled"])
    scaler_disc = torch.amp.GradScaler(
        "cuda", enabled=config["train"]["amp"]["enabled"]
    )

    extrinsics_sampler = ExtrinsicsSamplingRouter(config["view_sampling_weights"])

    if rank == 0:
        experiment_name = f"{config['experiment_name']}_{config['model']['model_type']}_t={int(time())}"
        config["train"]["output_dir"] = os.path.join(
            config["train"]["output_dir"], experiment_name
        )
        os.makedirs(config["train"]["output_dir"], exist_ok=True)
        logger = setup_logging(
            log_file=os.path.join(config["train"]["output_dir"], "training.log")
        )
        writer = SummaryWriter(
            log_dir=os.path.join(config["train"]["output_dir"], "tensorboard")
        )
        with open(os.path.join(config["train"]["output_dir"], "config.yaml"), "w") as f:
            yaml.dump(config, f)
    else:
        writer = None
        logger = None

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {total_params / 1e6:.2f} million trainable parameters.")
        logger.info(
            f"Image resolution: {config['data']['image_size']}x{config['data']['image_size']}."
        )
        logger.info(f"Using {world_size} GPUs for training.")
        logger.info(f"Global batch size: {config['train']['global_batch_size']}.")
        logger.info(f"Training for {total_steps} total steps.")
        logger.info(f"Learning rate: {config['train']['lr']}.")
        if scheduler:
            logger.info("Main LR Scheduler: Warmup + Cosine Decay enabled.")
        if disc_scheduler:
            logger.info("Discriminator LR Scheduler: Warmup + Cosine Decay enabled.")

    global_step = 0
    start_time = time()

    ckpt_path = config["train"].get("checkpoint_path", None)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        if rank == 0:
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        discriminator.module.load_state_dict(checkpoint["discriminator"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])

        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        if disc_scheduler is not None and "disc_scheduler" in checkpoint:
            disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])

        global_step = checkpoint["global_step"]

        if rank == 0:
            logger.info(f"Resumed at Global Step {global_step}")
    elif ckpt_path is not None and rank == 0:
        logger.warning(
            f"Checkpoint path provided but not found: {ckpt_path}. Starting from scratch."
        )

    log_accumulators = {
        "loss/total": 0.0,
        "loss/novel_view_recon": 0.0,
        "loss/gan": 0.0,
        "loss/lpips": 0.0,
        "loss/dino_perceptual": 0.0,
        "misc/gan_adaptive_weight": 0.0,
        "misc/grad_norm": 0.0,
        "disc/loss": 0.0,
        "disc/logits_real": 0.0,
        "disc/logits_fake": 0.0,
    }
    log_counts = 0

    recon_loss_fn = get_recon_loss_fn(config)
    if rank == 0:
        logger.info(
            f"Using {config['train'].get('recon_loss', 'charbonnier')} loss for reconstruction."
        )

    # define percentage-based thresholds
    disc_start_ratio = disc_loss_config.get("disc_start_ratio", 0.0)
    disc_upd_start_ratio = disc_loss_config.get("disc_upd_start_ratio", 0.0)
    disc_start_step = int(total_steps * disc_start_ratio)
    disc_upd_start_step = int(total_steps * disc_upd_start_ratio)

    data_iterator = iter(get_infinite_loader(train_loader, train_sampler))

    while global_step < total_steps + 1:
        data_batch = next(data_iterator)

        # trigger GAN logic based on step percentage milestones
        use_gan = (global_step >= disc_start_step) and (
            disc_loss_config.get("disc_weight", 0.0) > 0.0
        )
        train_disc = (global_step >= disc_upd_start_step) and (
            disc_loss_config.get("disc_weight", 0.0) > 0.0
        )

        optimizer.zero_grad(set_to_none=True)
        discriminator.eval()

        # load data and build targets
        nvs = config["view_sampling_params"]

        images_cam0 = data_batch["images"].to(device)
        points3d_cam0 = (
            data_batch["points3d"].to(device).reshape(images_cam0.shape[0], -1, 3)
        )
        intrinsics_cam0 = data_batch["intrinsics"].to(device)
        masks_cam0 = data_batch["mask"].to(device)
        normals_cam0 = data_batch["normals"].to(device)
        dino_features_cam0 = data_batch["dino_features"].to(device)
        dino_features_cam0 = dino_features_cam0.reshape(
            dino_features_cam0.shape[0], dino_features_cam0.shape[1], -1
        ).transpose(1, 2)

        (
            images_cam1_target,
            images_cam1_target_visibility_mask,
            cam01_params,
            cam10_params,
        ) = build_training_targets(
            images_cam0,
            points3d_cam0,
            intrinsics_cam0,
            masks_cam0,
            normals_cam0,
            extrinsics_sampler,
            config,
            nvs,
            device,
        )

        # compute generator losses
        with torch.amp.autocast("cuda", enabled=config["train"]["amp"]["enabled"]):
            images_cam1_pred = model(x=images_cam0, cam_params=cam01_params)
            novel_view_recon_loss = recon_loss_fn(
                images_cam1_pred,
                images_cam1_target,
                mask=images_cam1_target_visibility_mask,
            )

            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_model is not None:
                masked_pred = images_cam1_pred * images_cam1_target_visibility_mask
                lpips_loss = lpips_model(
                    masked_pred * 2 - 1, images_cam1_target * 2 - 1
                ).mean()

            dino_perceptual_loss = torch.tensor(0.0, device=device)
            if dino_model is not None:
                masked_pred = images_cam1_pred * images_cam1_target_visibility_mask
                pred_and_target = torch.cat((masked_pred, images_cam1_target), dim=0)

                norm_transform = transforms.Normalize(
                    mean=config["train"].get("imagenet_mean", (0.485, 0.456, 0.406)),
                    std=config["train"].get("imagenet_std", (0.229, 0.224, 0.225)),
                )
                pred_and_target_norm = norm_transform(pred_and_target)
                features_all = dino_model(pred_and_target_norm).last_hidden_state

                start_idx = 1 + dino_model.config.num_register_tokens
                features_all = features_all[:, start_idx:]

                feat_pred, feat_target = features_all.split(
                    images_cam1_pred.shape[0], dim=0
                )

                sim = torch.cosine_similarity(feat_pred, feat_target, dim=-1)
                per_patch_loss = 1.0 - sim

                B, C, H, W = images_cam1_target_visibility_mask.shape
                N_patches = feat_pred.shape[1]
                patch_grid_size = int(math.sqrt(N_patches))

                mask_down = torch.nn.functional.interpolate(
                    images_cam1_target_visibility_mask.float(),
                    size=(patch_grid_size, patch_grid_size),
                    mode="area",
                )

                mask_flat = mask_down.reshape(B, -1)
                masked_loss = per_patch_loss * mask_flat
                dino_perceptual_loss = masked_loss.sum() / (mask_flat.sum() + 1e-6)

            if use_gan:
                images_cam1_pred_normed = images_cam1_pred * 2.0 - 1.0
                fake_aug = disc_aug.aug(images_cam1_pred_normed)
                logits_fake, _ = discriminator(fake_aug, None)
                gan_loss = gen_loss_fn(logits_fake)
            else:
                gan_loss = torch.zeros_like(novel_view_recon_loss)

        if use_gan:
            adaptive_weight = calculate_adaptive_weight(
                novel_view_recon_loss,
                gan_loss,
                last_layer,
                max_d_weight=disc_loss_config.get("max_d_weight", 1.0),
            )
            total_loss = (
                novel_view_recon_loss
                + lpips_loss * config["train"].get("lpips_weight", 0.0)
                + dino_perceptual_loss
                * config["train"].get("dino_perceptual_loss_weight", 0.0)
                + disc_loss_config.get("disc_weight", 0.0) * adaptive_weight * gan_loss
            )
        else:
            adaptive_weight = torch.zeros_like(novel_view_recon_loss)
            total_loss = (
                novel_view_recon_loss
                + lpips_loss * config["train"].get("lpips_weight", 0.0)
                + dino_perceptual_loss
                * config["train"].get("dino_perceptual_loss_weight", 0.0)
            )

        losses_to_check = [
            novel_view_recon_loss,
            gan_loss,
            lpips_loss,
            dino_perceptual_loss,
        ]

        if use_gan:
            losses_to_check.append(adaptive_weight)

        valid_step_flag = torch.tensor(1.0, device=device)
        for l in losses_to_check:
            if not torch.isfinite(l):
                valid_step_flag = torch.tensor(0.0, device=device)
                break

        if dist.is_initialized():
            dist.all_reduce(valid_step_flag, op=dist.ReduceOp.MIN)

        if valid_step_flag.item() < 0.5:
            if rank == 0:
                logger.warning(
                    f"Warning: NaN/Inf detected in outputs or losses at step {global_step}. Skipping step."
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler_gen.scale(total_loss).backward()
        scaler_gen.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config["train"].get("max_grad_norm", 1.0)
        )
        log_accumulators["misc/grad_norm"] += grad_norm.item()

        valid_grad_flag = torch.tensor(1.0, device=device)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            valid_grad_flag = torch.tensor(0.0, device=device)

        if dist.is_initialized():
            dist.all_reduce(valid_grad_flag, op=dist.ReduceOp.MIN)

        if valid_grad_flag.item() < 0.5:
            if rank == 0:
                logger.warning(
                    f"Warning: NaN/Inf detected in Gradients at step {global_step}. Skipping step."
                )
            scaler_gen.update()
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler_gen.step(optimizer)
        scaler_gen.update()

        if scheduler is not None:
            scheduler.step()

        update_ema(ema, model.module, decay=config["train"]["ema_decay"])

        # discriminator update
        disc_metrics = None
        if train_disc:
            model.eval()
            discriminator.train()
            disc_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config["train"]["amp"]["enabled"]):
                real_normed = images_cam0 * 2.0 - 1.0
                with torch.no_grad():
                    images_cam1_pred = model_woddp(
                        x=images_cam0, cam_params=cam01_params
                    )
                    images_cam1_pred_normed = images_cam1_pred * 2.0 - 1.0

                fake_detached = images_cam1_pred_normed.clamp(-1.0, 1.0)
                fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                fake_input = disc_aug.aug(fake_detached)
                real_input = disc_aug.aug(real_normed)
                logits_fake, logits_real = discriminator(fake_input, real_input)
                d_loss = disc_loss_fn(logits_real, logits_fake)

            scaler_disc.scale(d_loss).backward()
            scaler_disc.step(disc_optimizer)
            scaler_disc.update()

            if disc_scheduler is not None:
                disc_scheduler.step()

            disc_metrics = {
                "d_loss": d_loss.item(),
                "logits_real": logits_real.mean().item(),
                "logits_fake": logits_fake.mean().item(),
            }

        discriminator.eval()
        model.train()
        dist.barrier()

        # logging
        log_accumulators["loss/total"] += total_loss.item()
        log_accumulators["loss/novel_view_recon"] += novel_view_recon_loss.item()
        log_accumulators["loss/gan"] += gan_loss.item()
        log_accumulators["loss/lpips"] += lpips_loss.item()
        log_accumulators["loss/dino_perceptual"] += dino_perceptual_loss.item()
        log_accumulators["misc/gan_adaptive_weight"] += (
            adaptive_weight.item() if use_gan else 0.0
        )

        if disc_metrics:
            log_accumulators["disc/loss"] += disc_metrics["d_loss"]
            log_accumulators["disc/logits_real"] += disc_metrics["logits_real"]
            log_accumulators["disc/logits_fake"] += disc_metrics["logits_fake"]

        log_counts += 1

        if rank == 0 and global_step % config["train"]["log_every"] == 0:
            averaged_stats = {
                key: val / log_counts for key, val in log_accumulators.items()
            }
            averaged_stats["misc/steps_sec"] = log_counts / (time() - start_time)

            if scheduler:
                averaged_stats["lr/main"] = scheduler.get_last_lr()[0]
            else:
                averaged_stats["lr/main"] = config["train"]["lr"]

            if disc_scheduler and train_disc:
                averaged_stats["lr/disc"] = disc_scheduler.get_last_lr()[0]
            elif train_disc:
                averaged_stats["lr/disc"] = disc_optimizer_config.get("lr", 2e-4)

            for key, value in averaged_stats.items():
                writer.add_scalar(key, value, global_step)

            print_msg = f"({100 * global_step / total_steps:.4f}%) Step {global_step} / {total_steps} "
            for key, value in averaged_stats.items():
                print_msg += f" | {key}: {value:.4f}"

            logger.info(print_msg)

            log_accumulators = {k: 0.0 for k in log_accumulators}
            log_counts = 0
            start_time = time()

        # save checkpoints
        if global_step % config["train"]["ckpt_every"] == 0 and global_step > 0:
            if rank == 0:
                logger.info("Saving checkpoint...")
                save_checkpoint(
                    model=model,
                    ema_model=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    output_dir=config["train"]["output_dir"],
                    logger=logger,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    disc_scheduler=disc_scheduler,
                )
            dist.barrier()

        # validate
        val_every = config["train"].get("val_every", 5000)
        if global_step % val_every == 0 and global_step > 0:
            if rank == 0:
                logger.info(f"Running validation at step {global_step}...")

            (
                val_loss_avg,
                val_novel_view_loss_avg,
                images_gt,
                images_cam1_target,
                images_sample,
            ) = validate(
                ema,
                depth_estimator,
                val_loader,
                extrinsics_sampler,
                config,
                nvs,
                device,
            )

            if rank == 0:
                writer.add_scalar("Loss/val", val_loss_avg, global_step)
                writer.add_scalar(
                    "Loss/val_novel_view_recon", val_novel_view_loss_avg, global_step
                )

                img_grid_source = utils.make_grid(images_gt, nrow=images_gt.shape[0])
                img_grid_target = utils.make_grid(
                    images_cam1_target, nrow=images_cam1_target.shape[0]
                )
                img_grid_sample = utils.make_grid(
                    images_sample, nrow=images_sample.shape[0]
                )

                writer.add_image("Images/val_source", img_grid_source, global_step)
                writer.add_image("Images/val_target", img_grid_target, global_step)
                writer.add_image("Images/val_sample", img_grid_sample, global_step)
                logger.info(f"Step {global_step} - Validation Loss: {val_loss_avg:.4f}")

            dist.barrier()

        global_step += 1

    # clean up
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        writer.close()
        logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
