
from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from torch.utils.tensorboard import SummaryWriter
import torchvision
from semantic_head import SemanticHead
from connector import Connector
#from torchsummary import summary

torch.autograd.set_detect_anomaly(True)

'''
- Parallelisation: carry out calculations simultaneously (across different devices)
- Process: execution of a program instance (there can be many instances such as opening an application multiple times)
- Threads: lightweight processes, share an address space. They have parent processes (processes that created them)
- Global interpreter lock: allows only one active thread
- Python multiprocessing module: uses subprocesses instead of threads so program can sidestep the GIL and run calculations in parallel
'''

# Guess: this setting chooses how memory is shared
torch.multiprocessing.set_sharing_strategy('file_system')

# Loads image labels depending on the dataset you use
def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


# Main class according to pytorch lightening
class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
           
            '''
            for name, param in self.net.named_parameters():
                print(param.shape)
            a = input()'''
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        
        # Model code
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        
        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)
        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False
        self.automatic_optimization = False # Setting this to false freezes backbone

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)
        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        if self.cfg.correspondence_weight > 0:
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
            )

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            loss += self.cfg.aug_alignment_weight * aug_alignment

        if self.cfg.crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            loss += self.cfg.crf_weight * crf

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            # parses batch through backbone
            feats, code = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
            # parses through head
            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)#

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)


            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}


            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return net_optim, linear_probe_optim, cluster_probe_optim


'''
- Hydra is a library for parsing arguments, has extra features
- Things can get messy quicky when you have lots of command line arguments due to multiple configurations
- Errors occur if arguments depend on each other, e.g. option dataset=image_net depends on option size=256x256
- Config files are better but still have problems; you need to record what changes you make to the settings which you can easily forget to do
- Allows for the option to have multiple config files for different situations which are grouped (e.g. load different default settings for different datasets). Does this by changing the path to the config file
https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710

- OmegaConfig: library supporting functions for YAML files (config files in this case)
- Example command: print(conf["heading"]["myvar"]) to print a variable from the yaml file
'''

'''
Sets up everyting and runs fit()
config_path: where the config files are located
config_name: what config file to use with this method
'''
@hydra.main(config_path="configs", config_name="train_config.yml") # Defines main function to run
def my_app(cfg: DictConfig) -> None: # ??
    OmegaConf.set_struct(cfg, False) # Allows you to edit config
    print(OmegaConf.to_yaml(cfg)) # Converts to yaml and prints
    pytorch_data_dir = cfg.pytorch_data_dir # Fetch path name from config
    data_dir = join(cfg.output_root, "data") # AVNS
    log_dir = join(cfg.output_root, "logs") # AVNS
    checkpoint_dir = join(cfg.pytorch_data_dir, "checkpoints") # AVNS

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name) # Gets path for particular dataset folder
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S')) # Gets name for particular dataset
    cfg.full_name = prefix # Adds a new variable to the cfg

    os.makedirs(data_dir, exist_ok=True) # Create dir for data
    os.makedirs(log_dir, exist_ok=True) # Create dir for log
    os.makedirs(checkpoint_dir, exist_ok=True) # Create dir for checkpoints

    seed_everything(seed=0) # Applies seed

    print(data_dir)
    print(cfg.output_root)

    # T is torchvision object
    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ]) # Effects to apply
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ]) # Effects to apply

    sys.stdout.flush() # Resets buffer

    # Build dataset
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    # Apply cropping depending on dataset
    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    # Build valuation datset
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )
    if cfg.subset:
        train_subset = torch.utils.data.Subset(train_dataset, list(range(0, 1000)))
        train_loader = DataLoader(train_subset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        if cfg.submitting_to_aml:
            val_batch_size = 16
        else:
            val_batch_size = cfg.batch_size
        val_subset = torch.utils.data.Subset(val_dataset, list(range(0, 1000)))
        val_loader = DataLoader(val_subset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        if cfg.submitting_to_aml:
            val_batch_size = 16
        else:
            val_batch_size = cfg.batch_size

        val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)


    # Instantiates the main pytorch lightening class
    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    print(join("/home/acb19fw/git/STEGO_v2/logs/", name))

    # Set up tensorboard
    tb_logger = TensorBoardLogger(
        join("/home/acb19fw/git/STEGO_v2/logs/", name),
        default_hp_metric=False,
        log_graph=True
    )


    # Azure ML stuff
    if cfg.submitting_to_aml:
        gpu_args = dict(gpus=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
        #gpu_args = dict(gpus=-1, distributed_backend="ddp", val_check_interval=cfg.val_freq)
        # gpu_args = dict(gpus=1, accelerator='ddp', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")


    trainer = Trainer(
        enable_progress_bar=cfg.progress_bar,
        #progress_bar_refresh_rate=0,
        #log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        # Steps take a precedence over epochs so we are removing max steps
        #max_steps=cfg.max_steps,
        max_epochs = cfg.epochs + 1,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                filename='{epoch}-{mIoU}',
                #every_n_train_steps=None,
                every_n_epochs=1,
                #train_time_interval=None,
                #period=1,
                save_top_k=-1,
                #monitor="test/cluster/mIoU",
                #mode="max",
            )
        ],
        **gpu_args
    )

    # Add graph to tensorboard
    '''
    How to run:
    1. Locate event file within the /checkpoints folder
    2. Zip folder with event file inside
    3. Upload to repo
    4. Pull into local
    5. Unzip and run tensorboard on your machine (install T.B. with conda powershell)
    '''
    '''
    writer = SummaryWriter('graph_log')
    dataiter = iter(train_loader)
    images = next(dataiter)
    batch = images["img"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    writer.add_graph(model.to(device), batch)
    writer.close()'''

    # Print model summary
    #demo = torch.tensor(np.ones(([16, 3, 224, 224]), dtype="float32"), dtype=torch.float32)
    #print(summary(model, demo))

    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    prep_args()
    my_app()

