import os, torch, time
import numpy as np
import robustness.data_augmentation as da
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from config_setups.config_log_setup import save_dict_to_json
from utils.general import load_json
from torch.nn import MultiMarginLoss
from autoattack.autopgd_base import APGDAttack_targeted
from autoattack.fab_pt import FABAttack_PT
from autoattack.square import SquareAttack

# ===== Project import =====
from loss.losses import CELoss, MarginLossOrig, MarginLossTrain, NegCELoss, NegMarginLoss
from models.model import CifarResNetFeatureModel, ShallowRandmomNet, UnionRes18, DeepMindWideResNetModel,\
    ImageNetResNetFeatureModel, AlexNetFeatureModel
from dataset.transform import TRAIN_TRANSFORMS_DEFAULT, TEST_TRANSFORMS_DEFAULT
from dataset.dataset import ImageNetN, CifarCDataset
from attacks.auto_attacks import AutoL1Attack, AutoL2Attack,\
     AutoLinfAttack, APGDAttackMargin, L1Attack, L2Attack, LinfAttack, APGDAttackCE
from attacks.target_fab import FABAttackPTModified

# ==== PAT attacks ====
from percept_utils.modified_attacks import LagrangePerceptualAttack_Revised, \
    FastLagrangePerceptualAttack_Revised, LagrangePerceptualAttack_Revised_Lp, \
    PerceptualPGDAttack


def get_balanced_subset(
    full_dataset, subset_len=None, 
    num_classes=None, sample_index_list=None,
    random_select=True):
    """
        Tested against cifar10 and ImageNet datasets.
    """
    if sample_index_list is None:
        assert subset_len is not None
        assert num_classes is not None
        assert subset_len > num_classes, "Need at least 1 sample per class"
        msg = "Constrcuting the Subset Dataset with size [%d] \n" % subset_len
        time_start = time.time()
        # initialize a dict to record the (sample_idx, label) pair
        total_sample_dict = {}  
        for key in range(num_classes):
            total_sample_dict[key] = []
        # record the (sample_idx, label) pair 
        full_dataset_len = len(full_dataset)
        for idx in range(full_dataset_len):
            sample_label = full_dataset[idx][1]  # Grab the label
            total_sample_dict[sample_label].append(idx)

        # === Construct subset index list ===
        sample_per_class = subset_len // num_classes
        sample_index_list = []
        for key in total_sample_dict.keys():
            if random_select:
                sample_list = np.random.choice(
                    total_sample_dict[key],
                    sample_per_class,
                    replace=False).tolist()
            else:
                sample_list = total_sample_dict[key][0:sample_per_class]
            sample_index_list += sample_list
        time_end = time.time()
        msg += ">> Time used to construct the subset: [%.03f] \n" % (
            time_end - time_start
        )
    else:
        msg = "Continue to use the previous Subset Dataset ... \n"
    
    # === Construct Subset ===
    subset = torch.utils.data.Subset(
        full_dataset, sample_index_list
    )
    return subset, sample_index_list, msg


def build_model(
    model_config, 
    global_config, 
    device
    ):
    model_type = model_config["type"]
    model_path = model_config["weight_path"]
    use_clamp_value = model_config["use_clamp_input"]
    # use_clamp_value = False
    
    if model_path is not None:
        model_path = eval(model_path)
    if "RobustUnion" in model_type:
        msg = " >> Init a UnionRes18 Model \n"
        model = UnionRes18(use_clamp_input=use_clamp_value).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device)
            model.model.load_state_dict(state_dict)
    elif "DeepMind" in model_type:
        msg = " >> Init a DeepMind WRN70(Cifar10) Model \n"
        model = DeepMindWideResNetModel(use_clamp_input=use_clamp_value).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path, map_location=device
            )
            model.model.load_state_dict(state_dict)
    elif "PAT-ImageNet" in model_type:
        msg = " >> Init a ImageNetResNetFeatureModel (PAT version) \n"
        num_classes = global_config["dataset"]["num_classes"]
        model = ImageNetResNetFeatureModel(
            num_classes=num_classes,
            use_clamp_input=use_clamp_value,
            pretrained=False,
            lpips_feature_layer=None
        ).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device
            )["model"]
            model.model.load_state_dict(state_dict)
    elif "PAT-Cifar" in model_type:
        msg = " >> Init a Cifar-Preatrained (PAT version) \n"
        num_classes = global_config["dataset"]["num_classes"]
        model = CifarResNetFeatureModel(
            num_classes=num_classes,
            use_clamp_input=use_clamp_value,
            pretrained=False,
            lpips_feature_layer=None
        ).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device
            )["model"]
            model.model.load_state_dict(state_dict)
    elif "Self-Train-Cifar10" in model_type:
        msg = " >> Init a UnionRes18 Model (Self-Trained) \n"
        model = UnionRes18(use_clamp_input=use_clamp_value).to(device)
        if model_path is not None:
            msg += " >> Weight from path: %s" % model_path
            state_dict = torch.load(
                model_path,
                map_location=device)["model_state_dict"]
            model.load_state_dict(state_dict)
    elif "Shallow" in model_type:
        msg = " >> Init a Shallow-NN (for sanity check only) \n"
        num_classes = global_config["dataset"]["num_classes"]
        model = ShallowRandmomNet(
            num_classes=num_classes,
            use_clamp_input=use_clamp_value
        ).to(device)
    else:
        raise RuntimeError("The author is lazy and did not implement another model yet.")
    return model, msg


def build_lpips_model(lpips_model_config, device):
    lpips_name = lpips_model_config["type"]
    lpips_layer = eval(lpips_model_config["lpips_layer"])
    if lpips_name == "alexnet":
        model = AlexNetFeatureModel(
            lpips_feature_layer=lpips_layer
            ).to(device)
    else:
        raise RuntimeError("Unimplemented LPIPS Model.")
    model.eval()
    return model


def get_optimizer(optimizer_cfg, model):
    opt_name = optimizer_cfg["type"]
    lr = optimizer_cfg["lr"]
    weight_decay = optimizer_cfg["weight_decay"]

    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr, 
            weight_decay=weight_decay
            )
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise RuntimeError("The author did not implement other optimizers yet.")
    return optimizer


def get_scheduler(config, optimizer):
    n_epoch = config["trainer_settings"]["num_epoch"]
    scheduler_name = config["scheduler"]["type"]

    if scheduler_name == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epoch
        )
    elif scheduler_name == "Exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.1
        )
    elif scheduler_name == "ReduceLROnPlateau":
        mode = config["scheduler"]["mode"]
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode, 
            patience=15
        )
    else:
        raise RuntimeError("The author did not implement other scheduler yet.")
    return scheduler


def get_loss_func_eval(name, reduction, use_clip_loss):
    if name == "CE":
        loss_func = CELoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> CE Loss Function."
    elif name == "Margin":
        loss_func = MarginLossOrig(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> Margin Loss Function"
    elif name == "Multi-Margin":
        loss_func = MultiMarginLoss(
            reduction=reduction
        )
        msg = "  >> MultiMargin Loss Function"
    else:
        raise RuntimeError("Unimplemented Loss Type")
    return loss_func, msg


def get_loss_func_train(name, reduction, use_clip_loss):
    if name == "CE":
        loss_func = CELoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> CE Loss Function."
    elif name == "Margin":
        loss_func = MarginLossTrain(
            reduction=reduction, 
            use_clip_loss=True
        )
        msg = "  >> Margin Loss Function"
    elif name == "Multi-Margin":
        loss_func = MultiMarginLoss(
            p=2,
            reduction=reduction
        )
        msg = "  >> MultiMargin Loss Function"
    else:
        raise RuntimeError("Unimplemented Loss Type")
    return loss_func, msg


def get_granso_loss_func(name, reduction, use_clip_loss, dataset_name="imagenet"):
    if name == "CE":
        if use_clip_loss:
            if dataset_name in ["imagenet", "cifar100"]:
                clamp_value = 4.7
            elif dataset_name in ["cifar10"]:
                clamp_value = 2.4
            else:
                raise RuntimeError("Need to specify dataset name for clipping")
        else:
            clamp_value = None
        loss_func = NegCELoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss,
            clamp_value=clamp_value
        )
        msg = "  >> Granso Neg-CE Loss"
    elif name == "Margin":
        loss_func = NegMarginLoss(
            reduction=reduction, 
            use_clip_loss=use_clip_loss
        )
        msg = "  >> Granso Neg-Margin Loss"
    else:
        raise RuntimeError("Unimplemented Granso Objective Function. Check 'granso_attacks.py'. ")
    return loss_func, msg


def get_train_transform(config):
    dataset_name = dataset_name = config["dataset"]["name"]
    transform_type = config["dataset"]["train_transform"]
    input_size = config["dataset"]["input_size"]
    if "imagenet" in dataset_name:
        if transform_type == "default":
            train_transform = da.TRAIN_TRANSFORMS_IMAGENET
        if transform_type == "plain":
            train_transform = da.TEST_TRANSFORMS_IMAGENET
        else:
            train_transform = da.TRAIN_TRANSFORMS_IMAGENET
            # raise RuntimeError("Unsupported Data Augmentation for Imagenet Training.")
    elif "cifar" in dataset_name:
        if transform_type == "default":
            train_transform = TRAIN_TRANSFORMS_DEFAULT(input_size)
        elif transform_type == "plain":
            train_transform = TEST_TRANSFORMS_DEFAULT(32)
        else:
            raise RuntimeError("Unsupported Data Augmentation for Imagenet Training.")
    else:
        raise RuntimeError("Error dataset name. Check input json files.")
    return train_transform


def get_loader_clean(config, only_val=False, shuffle_val=True):
    dataset_name = config["dataset"]["name"]
    data_path = config["dataset"]["clean_dataset_path"]
    batch_size = config["dataset"]["batch_size"]
    num_worders = config["dataset"]["workers"]

    train_transform = get_train_transform(config)
    msg = " >> Train Transform: [%s]" % train_transform.__repr__()

    if "imagenet" in dataset_name:
        label_list = eval(config["dataset"]["label_list"])
        dataset = ImageNetN(data_path=data_path,
                            label_list=label_list,
                            train_transform=train_transform)
        train_loader, val_loader = dataset.make_loaders(
                                    workers=num_worders, 
                                    batch_size=batch_size,
                                    only_val=only_val,
                                    shuffle_val=shuffle_val
                                    )
        return train_loader, val_loader, msg
    elif "cifar100" in dataset_name:
        if only_val:
            train_loader = None
        else:
            train_dataset = CIFAR100(
                root=data_path, train=True, transform=train_transform, download=True
            )
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worders
            )
        
        val_dataset = CIFAR100(
            root=data_path, train=False, transform=TEST_TRANSFORMS_DEFAULT(32), download=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return train_loader, val_loader, msg
    elif "cifar10" in dataset_name:
        if only_val:
            train_loader = None
        else:
            train_dataset = CIFAR10(
                root=data_path, train=True, transform=train_transform, download=True
            )
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worders
            )

        val_dataset = CIFAR10(
            root=data_path, train=False, transform=TEST_TRANSFORMS_DEFAULT(32), download=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return train_loader, val_loader, msg
    else:
        raise RuntimeError("Unsupported Dataset Yet.")


def get_loader_c(config, 
                 corr_type,
                 corr_level,
                 shuffle_val=False):
    dataset_name = config["dataset"]["name"]
    data_path = config["dataset"]["dataset_path"]
    batch_size = config["dataset"]["batch_size"]
    num_worders = config["dataset"]["workers"]
    if "imagenet" in dataset_name:
        label_list = eval(config["dataset"]["label_list"])
        raise RuntimeError("Do not have imagenet-c dataset implemented")
    elif "cifar100" in dataset_name:
        dataset = CifarCDataset(
            root_dir=data_path, corr_type=corr_type,
            corr_level=corr_level, transform=TEST_TRANSFORMS_DEFAULT(32)
        )
        curr_data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders 
        )
        return curr_data_loader
    elif "cifar10" in dataset_name:
        dataset = CifarCDataset(
            root_dir=data_path, corr_type=corr_type,
            corr_level=corr_level, transform=TEST_TRANSFORMS_DEFAULT(32)
        )
        curr_data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return curr_data_loader
    else:
        raise RuntimeError("Invalid Input {} & {}".format(corr_type, corr_level))


def get_loader_clean_subset(
    config, 
    only_val=False, 
    shuffle_val=False,
    random_select_train_data=True, 
    shuffle_train=True):
    """
        This Dataloader is a response to Tiancong's Request, 
        where we need a balanced subset of the original dataset.
    """
    dataset_name = config["dataset"]["name"]
    data_path = config["dataset"]["clean_dataset_path"]
    batch_size = config["dataset"]["batch_size"]
    num_worders = config["dataset"]["workers"]

    # ==== Read Subset Related Info ====
    subset_index_path = os.path.join(
        config["checkpoint_dir"],
        "subset_index.json"
    )
    sample_index_list = None
    if os.path.exists(subset_index_path):
        sample_index_list = load_json(subset_index_path)["index_list"]
    
    train_transform = get_train_transform(config)
    msg = " >> Train Transform: [%s] \n" % train_transform.__repr__()
    if "cifar10" in dataset_name:
        if only_val:
            train_loader = None
        else:
            train_dataset = CIFAR10(
                root=data_path, train=True, transform=train_transform, download=True
            )
            if sample_index_list is None:
                subset_len = config["dataset"]["subset_size"]
                train_subset, sample_index_list, train_dataset_msg = get_balanced_subset(
                    train_dataset, subset_len, num_classes=10, 
                    random_select=random_select_train_data
                )
            else:
                train_subset, sample_index_list, train_dataset_msg = get_balanced_subset(
                    train_dataset, sample_index_list=sample_index_list
                )
            # ==== Save Subset Indices for continue training ====
            subset_index_dict = {
                "index_list": sample_index_list
            }
            save_dict_to_json(subset_index_dict, subset_index_path)

            # === Get Train DataLoader ===
            train_loader = DataLoader(
                dataset=train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_worders
            )
            msg += train_dataset_msg

        # === Get Validation Loader ===
        val_dataset = CIFAR10(
            root=data_path, train=False, transform=TEST_TRANSFORMS_DEFAULT(32), download=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_worders
        )
        return train_loader, val_loader, msg
    else:
        raise RuntimeError("Unsupported Dataset Yet.")


def get_auto_attack(model, attack_config, dataset_config):
    dataset_name = dataset_config["name"]
    if "cifar100" in dataset_name:
        name = "cifar"
    elif "imagenet" in dataset_name:
        name = "imagenet"
    else:
        raise RuntimeError("Unsupported dataset name. Check input.")

    attack_name = attack_config["attack_type"]
    attack_bound = attack_config["attack_bound"]
    if attack_name == "L2":
        # default bound = 1200/255
        attack = AutoL2Attack(
            model, name, bound=attack_bound
        )  
    elif attack_name == "Linf":
        # default bound = 4/255
        attack = AutoLinfAttack(
            model, name, bound=attack_bound
        )
    elif attack_name == "L1":
        attack = AutoL1Attack(
            model, name, bound=attack_bound
        )
    else:
        raise RuntimeError("Unsupported Attack Type.")
    msg = " >> Getting Attack: [%s]" % attack_name
    return attack, msg


def get_lp_attack(
    attack_config, 
    model, device, 
    global_config=None, 
    is_train=False):

    attack_type = attack_config["attack_type"]
    norm_type = attack_config["attack_method"]
    bound = attack_config["attack_bound"]
    n_iter = attack_config["apgd_max_iter"]
    n_restart = 1 if is_train else 5
    
    if attack_type in ["APGD", "APGD-CE"]:
        print("APGD-CE")
        attack = APGDAttackCE(
            model, n_restarts=n_restart, n_iter=n_iter, verbose=False, 
            eps=bound, norm=norm_type, eot_iter=1, rho=0.75, 
            seed=None, device=device, logger=None
        )
    elif attack_type == "APGD-Margin":
        print("APGD-Margin")
        attack = APGDAttackMargin(
            model, n_restarts=n_restart, n_iter=n_iter, verbose=False, 
            eps=bound, norm=norm_type, eot_iter=1, rho=0.75, 
            seed=None, device=device, logger=None
        )
    elif attack_type == "FAB":
        # attack = FABAttack_PT(
        #     model, n_restarts=5, n_iter=100,
        #     eps=bound, seed=0, norm=norm_type,
        #     verbose=False, device=device
        # )
        print("FAB")
        attack =  FABAttackPTModified(
            model, n_restarts=n_restart, n_iter=100,
            targeted=False,
            eps=bound, seed=0, norm=norm_type,
            verbose=False, device=device
        )
    elif attack_type == "FAB-Target":
        print("Target FAB")
        attack = FABAttackPTModified(
            model, n_restarts=3, n_iter=100, targeted=True,
            eps=bound, seed=0, norm=norm_type,
            verbose=False, device=device
        )
    elif attack_type == "Square":
        print("SQUARE")
        attack = SquareAttack(
            model, p_init=0.8, n_queries=5000,
            eps=bound, norm=norm_type, n_restarts=1,
            seed=None, verbose=False, device=device,
            resc_schedule=False
        )
    else:
        raise RuntimeError("Not Implemented Attack.")
    return attack


def generate_attack_lp(
    inputs, labels, 
    device, attack,
    target_class=None
    ):
    inputs = inputs.to(device, dtype=torch.float)
    if labels is not None:
        labels = labels.to(device, dtype=torch.long)

    if type(attack) in [APGDAttackCE, APGDAttackMargin]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels, best_loss=False
        )
    elif type(attack) in [APGDAttack_targeted]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels
        )
    elif type(attack) in [SquareAttack, FABAttack_PT]:
        attack_adv_inputs = attack.perturb(
            x=inputs, y=labels
        )  # None for square attack is random target attack
    elif type(attack) in [FABAttackPTModified]:
        if attack.targeted == True:
            assert target_class is not None, "Need to specify a target class."
            attack_adv_inputs = attack.perturb(
                inputs, labels, target_class
            )
        else:
            attack_adv_inputs = attack.perturb(
                inputs, labels
            )
    elif type(attack) in [L1Attack, L2Attack, LinfAttack]:
        attack_adv_inputs = attack(
            inputs, labels
        )
    else:
        raise RuntimeError("Undefined Attack Type.")
    return attack_adv_inputs


def get_perceptual_attack(
    config, model, lpips_model,
    loss_func):
    print("PAT attack use loss: ", loss_func.__repr__())
    print("Use Clip Loss:", loss_func.use_clip_loss)
    attack_name = config["test_attack"]["alg"]
    num_iter = config["test_attack"]["pat_iter"]
    attack_bound = config["test_attack"]["bound"]
    if attack_name == "Lagrange":
        train_attack = LagrangePerceptualAttack_Revised(
            model,
            lpips_model,
            loss_func=loss_func,
            num_iterations=num_iter,
            bound=attack_bound
            )
    elif attack_name == "FastLagrange":
        train_attack = FastLagrangePerceptualAttack_Revised(
            model,
            lpips_model,
            loss_func=loss_func,
            num_iterations=num_iter,
            bound=attack_bound
            )
    elif attack_name == "PPGD":
        train_attack = PerceptualPGDAttack(
            model,
            lpips_model,
            num_iterations=num_iter,
            bound=attack_bound
        )
    elif attack_name == "LagrangeLp":
        lpips_type = config["test_attack"]["lpips_type"]
        train_attack = LagrangePerceptualAttack_Revised_Lp(
            model,
            lpips_model,
            lpips_distance_norm=lpips_type,
            num_iterations=num_iter,
            bound=attack_bound
        )
    else:
        raise RuntimeError("Unsupported Perceptual Attack Type.")
    return train_attack


if __name__ == "__main__":
    pass

