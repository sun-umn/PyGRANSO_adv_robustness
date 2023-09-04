import sys
sys.path.append("C:\\Users\\Liang\\AdvRobustnessGranso\\code")
sys.path.append("/home/jusun/liang656/AdvRobustnessGranso/code/")
sys.path.append("E:\\AdvRobustnessGranso\\code")
import argparse, os, torch, time
# =======
from utils.build import build_model, get_loss_func_eval, get_loader_clean
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.general import load_json, print_and_log, save_image, tensor2img
from utils.train import get_samples
from attacks.target_fab import FABAttackPTModified
import numpy as np

# ======= The following functions should be synced ======
from unit_test.validate_granso_target import calc_min_dist_sample_fab


if __name__ == "__main__":
    clear_terminal_output()
    print("Validate FAB Untarget Version...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'validate_granso_min.json'),
        help="Path to the json config file."
    )
    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    assert cfg["dataset"]["batch_size"] == 1, "Only want to support batch size = 1"

    fab_dtype = torch.float
    granso_dtype = torch.double

    # Experiment Root
    save_root = os.path.join("..", "log_folder")
    root_name = cfg["log_folder"]["save_root"]
    save_root = os.path.join(save_root, root_name)
    makedir(save_root)

    # Experiment ID
    attack_config = cfg["test_attack"]
    attack_type = attack_config["attack_method"]
    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]
    # Create Experiment Log Dir 
    exp_name = "FAB-%s-Untarget-%d-%d" % (
        attack_type, start_sample, end_sample
    )
    check_point_dir = os.path.join(
        save_root, 
        exp_name
    )
    if cfg["continue"]:
        check_point_dir = cfg["checkpoint_dir"]
    else:
        cfg["checkpoint_dir"] = check_point_dir
        makedir(check_point_dir)
    
    # Create Experiment Log File and save settings
    log_file = create_log_info(check_point_dir)
    save_exp_info(check_point_dir, cfg)
    device, _ = set_default_device(cfg)

    # Create save csv dir
    fab_csv_dir = os.path.join(
        check_point_dir, "fab_best.csv"
    )

    # === Setup Classifier Model ===
    cls_model_config = cfg["classifier_model"]
    classifier_model, msg = build_model(
        model_config=cls_model_config, 
        global_config=cfg, 
        device=device
    )
    classifier_model.eval()
    print_and_log(msg, log_file, mode="w")

    # ==== Construct the orginal min loss corresponds to the max loss for after opt check ====
    attack_loss_config = attack_config["granso_loss"]
    min_loss_func_name = attack_loss_config["type"]
    min_loss_func, msg = get_loss_func_eval(
        min_loss_func_name, 
        reduction="none", 
        use_clip_loss=False
    )

    # ==== Get Clean Data Loader ====
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )
    num_classes = cfg["dataset"]["num_classes"]

    # ===============================
    fab_summary = {
        "sample_idx": [],
        "true_label": [],
        "clean_max_logit": [],
        
        "fab_init_distance":[],
        "fab_adv_loss": [],
        "fab_target": [],
        "time": []
    }

    # ====
    attack_true_label = attack_config["attack_true_label"]
    n_restart = attack_config["fab_restart"]
    fab_init_scale = attack_config["fab_init_scale"]
    if attack_type == "L1-Reform":
        fab_attack_type = "L1"
    else:
        fab_attack_type = attack_type
    
    fab_max_iter = attack_config["fab_max_iter"]
    fab_attack = FABAttackPTModified(
        classifier_model, n_restarts=n_restart, n_iter=fab_max_iter,
        targeted=False,
        eps=fab_init_scale, seed=0, norm=fab_attack_type,
        verbose=False, device=device
    )  

    # === Lists to save dataset ===
    orig_image_list = []
    adv_image_list = []

    # ==== Perform attack
    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_batch"]:
            # Option to select image to test
            # do nothing because these batches have been tested.
            pass
        else:
            if batch_idx > (cfg["end_batch_num"]-1):
                break
            print_and_log(
                "======= Testing Batch [%d] =======" % batch_idx, log_file
            )
            fab_summary["sample_idx"].append(batch_idx)
            
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )
            # ==== fab init and granso continue attack ====
            inputs = inputs.to(device, dtype=fab_dtype)
            labels = labels.to(device)
            classifier_model = classifier_model.to(device, dtype=fab_dtype)
            with torch.no_grad():
                pred_logits = classifier_model(inputs)
                pred = pred_logits.argmax(1)
            if attack_true_label:
                attack_target = labels
            else:
                attack_target = pred
            pred_correct = (pred == attack_target).sum().item()

            if pred_correct < 0.5:
                print_and_log(
                    "Batch idx [%d] predicted wrong. Skip." % batch_idx,
                    log_file
                )
                for key in fab_summary.keys():
                    if key != "sample_idx":
                        fab_summary[key].append(-10)
            else:
                fab_summary["clean_max_logit"].append(torch.amax(pred_logits).item())
                fab_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now using FAB untarget attack...",
                    log_file
                )
                
                # ===== FAB untarget attack
                fab_best_distance = float("inf")
                fab_best_adv_sample = None
                fab_time_start = time.time()
                fab_adv_output = fab_attack.perturb(
                    inputs, attack_target
                )
                fab_time_end = time.time()
                print_and_log(
                    "  -- Fab Untarget %d restart total time -- %.04f" % (
                        n_restart, fab_time_end - fab_time_start
                    ),
                    log_file
                )
                fab_final_output, fab_final_distance, fab_distance_list, _, _ = calc_min_dist_sample_fab(
                    fab_adv_output, None,
                    inputs, attack_type, log_file, classifier_model, attack_target.item()
                )
                if fab_final_distance < fab_best_distance:
                    fab_best_distance = fab_final_distance
                    fab_best_adv_sample = fab_final_output
                else:
                    fab_best_adv_sample = fab_adv_output[0]
                
                # ==== Visulize Samples =====
                # img_orig = tensor2img(inputs)
                # img_vis = tensor2img(fab_best_adv_sample)
                # vis_orig_name = os.path.join(check_point_dir, "%d_FAB_orig.png" % batch_idx)
                # vis_name = os.path.join(check_point_dir, "%d_FAB_adv_vis.png" % batch_idx)
                # save_image(img_vis, vis_name)
                # save_image(img_orig, vis_orig_name)

                # ===== record best result =====
                fab_summary["fab_init_distance"].append(fab_best_distance)
                with torch.no_grad():
                    classifier_model = classifier_model.to(device, dtype=fab_dtype)
                    final_pred = classifier_model(fab_best_adv_sample)
                    final_target = final_pred.argmax(1).item()
                    adv_loss = min_loss_func(final_pred, attack_target).item()
                fab_summary["fab_adv_loss"].append(adv_loss)
                fab_summary["fab_target"].append(final_target)
                fab_summary["time"].append(fab_time_end-fab_time_start)

                save_dict_to_csv(
                    fab_summary, fab_csv_dir
                )
                
                # === Visualize Image ===
                if cfg["save_vis"]:
                    vis_dir = os.path.join(
                        check_point_dir, "dataset_vis"
                    )
                    os.makedirs(vis_dir, exist_ok=True)

                    # adv_image_np = np.clip(
                    #     tensor2img(fab_best_adv_sample.to(device, dtype=fab_dtype)), 0, 1
                    # )
                    # orig_image_np = np.clip(
                    #     tensor2img(inputs), 0, 1
                    # )
                    adv_image_np = tensor2img(fab_best_adv_sample.to(device, dtype=fab_dtype))
                    orig_image_np = tensor2img(inputs)

                    orig_image_list.append(orig_image_np)
                    adv_image_list.append(adv_image_np)
                    orig_save_name = os.path.join(
                        vis_dir, "orig_img_list.npy"
                    )
                    adv_save_name = os.path.join(
                        vis_dir, "adv_img_list.npy"
                    )
                    print("Check Shape: ", np.asarray(orig_image_list).shape)
                    print("             ", np.asarray(adv_image_list).shape)
                    np.save(
                        orig_save_name, np.asarray(orig_image_list)
                    )
                    np.save(
                        adv_save_name, np.asarray(adv_image_list)
                    )