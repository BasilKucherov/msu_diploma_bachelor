import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from models import DSCNN
from loss_functions import *
from transforms import *
from dataset import KWSDataset
from dataset import BackgroundNoiseDataset
import os
from pathlib import Path
from clustering_metrics import clustering_metric_fc, clustering_metric_hv
from sklearn.metrics import davies_bouldin_score, silhouette_score
import json
import tqdm
import argparse
import logging
import time

SPEECH_COMMANDS_SAMPLE_RATE = 16000
MSWC_SAMPLE_RATE = 16000

DEFAULT_CLASSES_PER_BATCH = 10 # Used in Triplet loss
DEFAULT_WORKERS_NUMBER = 16
DATALOADER_DROP_LAST = True

STATS_FILE_NAME = "stats.json"
CONFIG_FILE_NAME = "config.json"

TRAIN_SUFFIX = "/train"
VALID_SUFFIX = "/valid"

def create_model(model_description: dict, overrule_embedding: bool=False):
    if 'name' not in model_description:
            return '[ERROR]: corrupted model config'

    if model_description['name'] == 'DSCNN':
        n_mels = model_description['n_mels']
        in_shape = (n_mels, 32)
        in_channels = model_description['in_channels']
        ds_cnn_number = model_description['ds_cnn_number']
        ds_cnn_size = model_description['ds_cnn_size']
        is_classifier = False if overrule_embedding else model_description['is_classifier']
        classes_number = 0 if not is_classifier else model_description['classes_number']

        return DSCNN(in_channels, in_shape, ds_cnn_number, ds_cnn_size, is_classifier, classes_number)


def create_dataloader(
    loss_description: dict,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    n_classes_per_batch: int=DEFAULT_CLASSES_PER_BATCH) -> DataLoader:

    dl = None
    
    match loss_description["name"]:
        case "CrossEntropyLoss":
            dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=DATALOADER_DROP_LAST)
        case "TripletLoss":
            batch_sampler = TripletBatchSampler(dataset.get_class_indices(), batch_size, n_classes_per_batch)
            dl = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        case "LiftedStructuredLoss":
            dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=DATALOADER_DROP_LAST)
        case "NPairLoss":
            batch_sampler = NPairBatchSampler(dataset.get_class_indices(), batch_size)
            dl = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        case "SilhouetteLoss":
            batch_sampler =  SilhouetteBatchSampler(dataset.get_class_indices(), batch_size, n_classes_per_batch)
            dl = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        case "SilhouetteMarginLoss":
            batch_sampler =  SilhouetteBatchSampler(dataset.get_class_indices(), batch_size, n_classes_per_batch)
            dl = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        case _:
            print(f"Error: unknown loss function {loss_description['name']}")

    return dl

def create_loss_function(loss_description: dict):
    loss_fn = None

    match loss_description["name"]:
        case "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss()
        case "TripletLoss":
            margin = loss_description['loss_margin'] if 'loss_margin' in loss_description.keys() else 1
            loss_agr_policy = loss_description['loss_agr_policy'] if 'loss_agr_policy' in loss_description.keys() else 'mean'
                
            if loss_description['triplet_mining_strategy'] == 'batch_random':
                loss_fn = TripletLossBatchRandom(margin=margin, loss_agr_policy=loss_agr_policy)
            elif loss_description['triplet_mining_strategy'] == 'batch_hard':
                loss_fn = TripletLossBatchHard(margin=margin, loss_agr_policy=loss_agr_policy)
        case "LiftedStructuredLoss":
            margin = loss_description['loss_margin'] if 'loss_margin' in loss_description.keys() else 1
            loss_fn = LiftedStructuredLoss(margin=margin)
        case "NPairLoss":
            l2_reg = loss_description['l2_reg'] if 'l2_reg' in loss_description.keys() else 0.02
            loss_fn = NpairLoss(l2_reg=l2_reg)
        case "SilhouetteLoss":
            loss_fn = SilhouetteLoss()
        case "SilhouetteMarginLoss":
            margin = loss_description['loss_margin'] if 'loss_margin' in loss_description.keys() else 1
            loss_fn = SilhouetteMarginLoss(margin=margin)
        case _:
            print(f"Error: unknown loss function {loss_description['name']}")
        
    return loss_fn

def create_train_dataset(root: str, split_path: str, background_noise_path: str, n_mels: int=32):
    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(background_noise_path, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])

    sample_rate = SPEECH_COMMANDS_SAMPLE_RATE if "speech" in root else MSWC_SAMPLE_RATE

    dataset = KWSDataset(root, 
                          split_path, 
                          Compose([
                                LoadAudio(sample_rate=sample_rate),
                                data_aug_transform,
                                add_bg_noise,
                                train_feature_transform]))
    
    return dataset


def create_valid_dataset(root: str, split_path: str, n_mels: int=32):
    valid_feature_transform = Compose([ToSTFT(), ToMelSpectrogramFromSTFT(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])

    sample_rate = SPEECH_COMMANDS_SAMPLE_RATE if "speech" in root else MSWC_SAMPLE_RATE

    dataset = KWSDataset(root,
                                split_path,
                                Compose([LoadAudio(sample_rate=sample_rate),
                                        FixAudioLength(),
                                        valid_feature_transform]))
    
    return dataset

def train_epoch(       
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    criterions: dict
) -> dict:
    model.train()
    optimizer.zero_grad()

    epoch_loss = []

    criterions_epoch = {name: 0 for name in criterions.keys()}

    for id, batch in enumerate(dataloader):
        images = batch["input"]
        images = torch.unsqueeze(images, 1)
        targets = batch["target"]

        images = images.to(device)
        targets = targets.to(device)

        net_out = model(images)
        loss = loss_fn(net_out, targets)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
    
    avg_loss = np.mean(epoch_loss)
    criterions_epoch["loss"] = avg_loss
    
    return criterions_epoch


def valid_epoch(       
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    criterions: dict
) -> dict:
    model.eval()

    epoch_loss = []

    all_outs = []
    all_targets = []

    criterions_epoch = {name: 0 for name in criterions.keys()}
    
    with torch.no_grad():
        for id, batch in enumerate(dataloader):
            images = batch["input"]
            images = torch.unsqueeze(images, 1)
            targets = batch["target"]

            images = images.to(device)
            targets = targets.to(device)

            net_out = model(images)            
            loss = loss_fn(net_out, targets)

            if model.is_classifier == True:
                model.is_classifier = False
                net_out = model(images)    
                model.is_classifier = True        
                
            epoch_loss.append(loss.item())
            
            all_outs += net_out.tolist()
            all_targets += targets.tolist()
        
        avg_loss = np.mean(epoch_loss)
        criterions_epoch["loss"] = avg_loss

        for criterion_name, criterion in criterions.items():
            criterions_epoch[criterion_name] = criterion(all_outs, all_targets)
    
    return criterions_epoch

def save_config(config):
    save_path = os.path.join(config["dir"], config["name"], CONFIG_FILE_NAME)

    with open(save_path, "w") as fp:
        json.dump(config, fp)


def train_encoder(train_config: dict, device: torch.device, validation_frequency: int=3):
    criterions = {
        "fc": clustering_metric_fc,
        "hv": clustering_metric_hv,
        "silhouette_score": silhouette_score,
        "davies_bouldin_score": davies_bouldin_score
    }
    
    model = create_model(train_config["model"]).to(device)
    train_dataset = create_train_dataset(train_config["train_dataset"]["root"],
                                         train_config["train_dataset"]["split_path"],
                                         train_config["train_dataset"]["background_noise_path"],
                                         train_config["model"]["n_mels"])
    valid_dataset = create_valid_dataset(train_config["valid_dataset"]["root"],
                                         train_config["valid_dataset"]["split_path"],
                                         train_config["model"]["n_mels"])

    loss_fn = create_loss_function(train_config["loss"])

    train_dataloader = create_dataloader(train_config["loss"],
                                         train_dataset,
                                         train_config["batch_size"],
                                         True,
                                         DEFAULT_WORKERS_NUMBER)
    valid_dataloader = create_dataloader(train_config["loss"],
                                         valid_dataset,
                                         train_config["batch_size"],
                                         True,
                                         DEFAULT_WORKERS_NUMBER)
    
    experiment_dir = os.path.join(train_config["dir"], train_config["name"])
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    n_epochs = train_config["n_epoch"]
    learning_rate = train_config["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # new
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    stats = {}

    for epoch in tqdm.tqdm(range(n_epochs)):
        epoch_stats = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, criterions)

        for key, value in epoch_stats.items():
            stats.setdefault(key + TRAIN_SUFFIX, {})
            stats[key + TRAIN_SUFFIX][epoch] = value

        if epoch % validation_frequency == 0 or epoch == n_epochs - 1:
            epoch_stats = valid_epoch(model, valid_dataloader, loss_fn, device, criterions)

            for key, value in epoch_stats.items():
                stats.setdefault(key + VALID_SUFFIX, {})
                stats[key + VALID_SUFFIX][epoch] = value

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(checkpoints_dir, f"epoch_{epoch:04d}"))
    
    stats_path = os.path.join(experiment_dir, STATS_FILE_NAME)
    with open(stats_path, "w") as fp:
        json.dump(stats, fp)
            
    train_config_path = os.path.join(experiment_dir, CONFIG_FILE_NAME)
    with open(train_config_path, "w") as fp:
        json.dump(train_config, fp)


def validate_encoder(experiment: str, device: torch.device):
    experiment_dir = os.path.join("experiments", experiment)
    train_config_path = os.path.join(experiment_dir, CONFIG_FILE_NAME)

    with open(train_config_path, "r") as fp:
        train_config = json.load(fp)

    criterions = {
        "fc": clustering_metric_fc,
        "hv": clustering_metric_hv,
        "silhouette_score": silhouette_score,
        "davies_bouldin_score": davies_bouldin_score
    }
    
    model = create_model(train_config["model"]).to(device)
    valid_dataset = create_valid_dataset(train_config["valid_dataset"]["root"],
                                         train_config["valid_dataset"]["split_path"],
                                         train_config["model"]["n_mels"])

    loss_fn = create_loss_function(train_config["loss"])

    valid_dataloader = create_dataloader(train_config["loss"],
                                         valid_dataset,
                                         train_config["batch_size"],
                                         True,
                                         DEFAULT_WORKERS_NUMBER)
    
    experiment_dir = os.path.join(train_config["dir"], train_config["name"])
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")

    checkpoints_names = sorted(os.listdir(checkpoints_dir))
    stats = {}

    print(f"Dataset (len|classes): {len(valid_dataset)} | {len(valid_dataset.classes)}")

    for checkpoint_name in tqdm.tqdm(checkpoints_names):
        epoch = int(checkpoint_name.split("_")[1])
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)

        checkpoint = torch.load(checkpoint_path)

        state_dict = checkpoint["state_dict"]

        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        
        epoch_stats = valid_epoch(model, valid_dataloader, loss_fn, device, criterions)
        
        for key, value in epoch_stats.items():
            stats.setdefault(key + VALID_SUFFIX, {})
            stats[key + VALID_SUFFIX][epoch] = value

    stats_path = os.path.join(experiment_dir, STATS_FILE_NAME)

    with open(stats_path, "w") as fp:
        json.dump(stats, fp)

def is_valid_config(config: dict):
    if "train_dataset" in train_config.keys() and not os.path.isdir(train_config["train_dataset"]["root"]):
        print(f"ERROR: '{train_config['train_dataset']['root']}' is not dir")
        return False
    
    if "train_dataset" in train_config.keys() and not os.path.isfile(train_config["train_dataset"]["split_path"]):
        print(f"ERROR: '{train_config['train_dataset']['split_path']}' is not file")
        return False

    if "train_dataset" in train_config.keys() and "background_noise_path" in train_config["train_dataset"].keys() and not os.path.isdir(train_config["train_dataset"]["background_noise_path"]):
        print(f"ERROR: '{train_config['train_dataset']['background_noise_path']}' is not dir")
        return False

    if "valid_dataset" in train_config.keys() and not os.path.isdir(train_config["valid_dataset"]["root"]):
        print(f"ERROR: '{train_config['valid_dataset']['root']}' is not dir")
        return False
    
    if "valid_dataset" in train_config.keys() and not os.path.isfile(train_config["valid_dataset"]["split_path"]):
        print(f"ERROR: '{train_config['valid_dataset']['split_path']}' is not file")
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(filename='train.log', level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Command line arguments")
    parser.add_argument("--config", type=str, help="Path to the config file with splits description", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        train_config = json.load(fp)

    if not is_valid_config(train_config):
        logging.info("ERROR corrupted config. Exiting.")
        print("ERROR corrupted config. Exiting.")
        exit(1)

    device = torch.device('cpu')
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device('cuda', 0)
        
    print(type(device), device)

    if use_gpu:
        torch.backends.cudnn.benchmark = True
    
    start_time = time.time()
    logging.info(f"Start train {train_config['name']}")
    train_encoder(train_config, device, 1)
    end_time = time.time()
    logging.info(f"Finish train {train_config['name']} in {end_time - start_time:.2f} seconds")
