import os
import argparse
import torch
import math
from torch.utils.data import Dataset,DataLoader
from utils.comm import setup_seed, create_dir
from configs.lld_default import get_cfg_defaults
from dataset.dataset_lld import LLD_Curve_Dataset, LLD_Curve_Dataset_collate_fn
from network.mbv2_mlsd import MobileV2_MLSD
from network.mbv2_mlsd_large import MobileV2_MLSD_Large
from network.lld_repvgg_large import Lld_Repvgg_Large
from lr_scheduler.lr_scheduler import WarmupMultiStepLR
from utils.logger import TxtLogger
from trainer.lld_trainer import Lld_trainer

torch.set_num_threads(4)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default= os.path.dirname(os.path.abspath(__file__))+ '/configs/lld_cfg.yaml',
                        type=str,
                        help="")
    return parser.parse_args()


def get_train_dataloader(cfg, is_train=True):
    dataset = LLD_Curve_Dataset(cfg, is_train=is_train)

    dataloader_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.sys.num_workers,
        drop_last=True,
        collate_fn=LLD_Curve_Dataset_collate_fn
    )
    return dataloader_loader


def get_val_dataloader(cfg, is_train=False):
    dataset = LLD_Curve_Dataset(cfg, is_train=is_train)

    dataloader_loader = DataLoader(
        dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.sys.num_workers,
        drop_last=False,
        collate_fn=LLD_Curve_Dataset_collate_fn
    )
    return dataloader_loader


def build_model(cfg):
    model_name = cfg.model.model_name
    if model_name == 'mobilev2_mlsd':
        m = MobileV2_MLSD(cfg)
        return m
    if model_name == 'mobilev2_mlsd_large':
        m = MobileV2_MLSD_Large(cfg)
        return m
    if model_name == 'repvgg_mlsd_large':
        m = Lld_Repvgg_Large(cfg)
        return m
    raise NotImplementedError('{} no such model!'.format(model_name))


def train(cfg):
    train_loader = get_train_dataloader(cfg, is_train=True)
    val_loader   = get_val_dataloader(cfg, is_train=False)
    model = build_model(cfg).cuda()

    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.train.load_from, map_location=device)
        include_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and v.shape==model_dict[k].shape)}
        exclude_dict = {k: v for k, v in pretrained_dict.items() if
                           (k not in model_dict or v.shape != model_dict[k].shape)}

        print("Exclude Weights from Pretrain")
        for k, v in exclude_dict.items():
            print("name:{},  shape:{}".format(k, v.shape))

        # 更新权重
        model_dict.update(include_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(torch.load(cfg.train.load_from, map_location=device), strict=False)

    if cfg.train.milestones_in_epo:
        ns = len(train_loader)
        milestones = []
        for m in cfg.train.milestones:
            milestones.append(m * ns)
        cfg.train.milestones = milestones

    optimizer = torch.optim.Adam(params=model.parameters(),lr=cfg.train.learning_rate,weight_decay=cfg.train.weight_decay)

    if cfg.train.use_step_lr_policy:
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones= cfg.train.milestones,
            gamma = cfg.train.lr_decay_gamma,
            warmup_iters=cfg.train.warmup_steps,
        )

    else: ## similiar with in the paper
        warmup_steps = 5 * len(train_loader) ## 5 epoch warmup
        min_lr_scale = 0.0001
        start_step = 70 * len(train_loader)
        end_step = 150 * len(train_loader)
        n_t = 0.5
        lr_lambda_fn = lambda step: (0.9 * step / warmup_steps + 0.1) if step < warmup_steps else \
            1.0 if step < start_step else \
                min_lr_scale if \
                    n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step))) < min_lr_scale else \
                    n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step)))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

    create_dir(cfg.train.save_dir)
    logger = TxtLogger(cfg.train.save_dir + "/train_logger.txt")
    lld_trainer = Lld_trainer(
        cfg,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        logger=logger,
        save_dir=cfg.train.save_dir,
        log_steps=cfg.train.log_steps,
        device_ids=cfg.train.device_ids,
        gradient_accum_steps=1,
        max_grad_norm=1000.0,
        batch_to_model_inputs_fn=None,
        early_stop_n=cfg.train.early_stop_n)

    #learner.val(model, val_loader)
    #learner.val(model, train_loader)
    lld_trainer.train(train_loader, val_loader, epoches=cfg.train.num_train_epochs)


if __name__ == '__main__':
    setup_seed(233)
    cfg = get_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ',args.config.strip())
    cfg.merge_from_file(args.config)
    print(cfg)

    create_dir(cfg.train.save_dir)
    cfg_str = cfg.dump()
    with open(cfg.train.save_dir + "/cfg.yaml", "w") as f:
        f.write(cfg_str)
    f.close()
    train(cfg)