import  os
import  torch
import  tqdm
from torch.utils.data import  Dataset, DataLoader
from torch.optim.optimizer import Optimizer

from utils.logger import TxtLogger
from utils.meter import AverageMeter
from loss.lld_loss import LldCurveLoss


class Lld_trainer():
    def __init__(self,
                 cfg,
                 model : torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler,
                 logger : TxtLogger,
                 save_dir : str,
                 log_steps = 100,
                 device_ids = [0,1],
                 gradient_accum_steps = 1,
                 max_grad_norm = 100.0,
                 batch_to_model_inputs_fn = None,
                 early_stop_n = 4,
                 ):
        self.cfg = cfg
        self.model  = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device_ids = device_ids
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn  = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0

        self.input_size = self.cfg.datasets.input_size
        self.loss_fn = LldCurveLoss(cfg)
        self.epo = 0

    def step(self,step_n,  batch_data : dict):
        imgs  = batch_data["xs"].cuda()

        batch_gt_confidence = batch_data["batch_gt_confidence"].cuda()
        batch_gt_offset_x = batch_data["batch_gt_offset_x"].cuda()
        batch_gt_offset_y = batch_data["batch_gt_offset_y"].cuda()
        batch_gt_line_index = batch_data["batch_gt_line_index"].cuda()
        batch_ignore_mask = batch_data["batch_ignore_mask"].cuda()
        batch_foreground_mask = batch_data["batch_foreground_mask"].cuda()
        batch_gt_line_id = batch_data["batch_gt_line_id"].cuda()
        batch_gt_line_cls = batch_data["batch_gt_line_cls"].cuda()
        batch_foreground_expand_mask = batch_data["batch_foreground_expand_mask"].cuda()

        outputs = self.model(imgs)

        loss_dict = self.loss_fn(outputs, batch_gt_confidence,
                                batch_gt_offset_x,
                                batch_gt_offset_y,
                                batch_gt_line_index,
                                batch_ignore_mask,
                                batch_foreground_mask,
                                batch_gt_line_id,
                                batch_gt_line_cls,
                                batch_foreground_expand_mask)

        loss = loss_dict['loss']
        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        return loss, loss_dict

    def val(self, model, val_dataloader : DataLoader):
        thresh = self.cfg.decode.score_thresh
        topk = self.cfg.decode.top_k
        min_len = self.cfg.decode.len_thresh

        model = model.eval()
        sap_thresh = 10
        data_iter = tqdm.tqdm(val_dataloader)
        f_scores = []
        recalls = []
        precisions = []

        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for batch_data in data_iter:
            imgs = batch_data["xs"].cuda()
            # label = batch_data["ys"].cuda()
            outputs = model(imgs)

            batch_gt_confidence = batch_data["batch_gt_confidence"].cuda()
            batch_gt_offset_x = batch_data["batch_gt_offset_x"].cuda()
            batch_gt_offset_y = batch_data["batch_gt_offset_y"].cuda()
            batch_gt_line_index = batch_data["batch_gt_line_index"].cuda()
            batch_ignore_mask = batch_data["batch_ignore_mask"].cuda()
            batch_foreground_mask = batch_data["batch_foreground_mask"].cuda()
            batch_gt_line_id = batch_data["batch_gt_line_id"].cuda()
            batch_gt_line_cls = batch_data["batch_gt_line_cls"].cuda()
            batch_foreground_expand_mask = batch_data["batch_foreground_expand_mask"].cuda()

            loss_dict = self.loss_fn(outputs, batch_gt_confidence,
                                     batch_gt_offset_x,
                                     batch_gt_offset_y,
                                     batch_gt_line_index,
                                     batch_ignore_mask,
                                     batch_foreground_mask,
                                     batch_gt_line_id,
                                     batch_gt_line_cls,
                                     batch_foreground_expand_mask)

            loss = loss_dict['loss']
        return {"loss": loss}

    def train(self, train_dataloader : DataLoader,
              val_dataloader : DataLoader,
              epoches=100):
        best_score = 0
        torch.set_num_threads(4)
        self.best_loss = -1

        for self.epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_avg_confidence_loss = AverageMeter()
            train_avg_offeset_loss = AverageMeter()
            train_avg_emb_loss = AverageMeter()
            train_avg_emb_id_loss = AverageMeter()
            train_avg_cls_loss = AverageMeter()

            data_iter = tqdm.tqdm(train_dataloader, ncols=120)  # set show length
            for batch in data_iter:
                self.model.train()
                train_loss, loss_dict = self.step(step_n, batch)
                train_avg_loss.update(train_loss.item(), 1)

                train_avg_confidence_loss.update(loss_dict['confidence_loss'].item(), 1)
                train_avg_offeset_loss.update(loss_dict['offset_loss'].item(), 1)
                train_avg_emb_loss.update(loss_dict['embedding_loss'].item(), 1)
                train_avg_emb_id_loss.update(loss_dict['embedding_id_loss'].item(), 1)
                train_avg_cls_loss.update(loss_dict['cls_loss'].item(), 1)

                status = '[{0}] lr= {1:.6f} los= {2:.3f}, ' \
                         'avg: {3:.3f}, cof: {4:.3f}, off: ' \
                         '{5:.3f}, emb: {6:.4f}, emb_id: {7:.4f}, ' \
                         'cls: {8:.4f}'.format(
                    self.epo + 1,
                    self.scheduler.get_lr()[0],
                    train_loss.item(),
                    train_avg_loss.avg,
                    train_avg_confidence_loss.avg,
                    train_avg_offeset_loss.avg,
                    train_avg_emb_loss.avg,
                    train_avg_emb_id_loss.avg,
                    train_avg_cls_loss.avg,
                    )

                data_iter.set_description(status)
                step_n +=1

            # ##self.scheduler.step() ## we update every step instead
            if self.epo > self.cfg.val.val_after_epoch:
                m = self.val(self.model, val_dataloader)
            if self.best_loss == -1 or \
                    (self.best_loss != -1 and train_loss.item() < self.best_loss):
                model_path = os.path.join(self.save_dir, 'best.pth')
                torch.save(self.model.state_dict(), model_path)
                self.best_loss = train_loss.item()

            model_path = os.path.join(self.save_dir, 'latest.pth')
            torch.save(self.model.state_dict(), model_path)
        return best_score
