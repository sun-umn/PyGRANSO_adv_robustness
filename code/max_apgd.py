import sys, os
sys.path.append(os.path.abspath("."))
import argparse, torch, time
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
# ===========
from utils.general import load_json, print_and_log, get_samples
from config_setups.config_log_setup import makedir
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.build import build_model, get_loss_func_eval, get_lp_attack, get_loader_clean, \
    generate_attack_lp
from utils.general import tensor2img, rescale_array
from percept_utils.distance import calc_distance
# ==== 


def calc_restart_summary(
    adv_input_dict, 
    granso_iter_dict,
    inputs, 
    attack_type, 
    log_file, 
    classifier_model,
    loss_func, 
    label, 
    lpips_model,
    eps,
    feasibility_thres,
    specific_key=None
):  
    keys = adv_input_dict.keys()
    best_loss = -float("inf")
    best_dist = float("inf")
    best_sample = None
    best_idx = None
    need_continue = True
    has_someone_succeeded = False
    best_iter = float("inf")

    for key in keys:
        with torch.no_grad():
            adv_input = adv_input_dict[key]
            total_iter = granso_iter_dict[key]
            adv_output_logit = classifier_model(adv_input)
            adv_loss = loss_func(adv_output_logit, label).item()
        
        # === If adv sample alters the decision prediction ===
        attack_success = adv_loss > -1e-12
        
        # === If the distance is smaller than eps ===
        distance = calc_distance(adv_input, inputs, attack_type, lpips_model)
        feasable = (distance < (eps * (1 + feasibility_thres)))

        
        # how much [0, 1] box constraint is violated
        greater_than_1 = torch.sum(torch.where(adv_input > (1 + 1e-4), 1, 0)).item()
        smaller_than_0 = torch.sum(torch.where(adv_input < (0 - 1e-4), 1, 0)).item()
        num_violation = greater_than_1 + smaller_than_0

        if key == specific_key:
            print_and_log(
                ">> Restart [{}]: ".format(key),
                log_file
            )
            print_and_log(
                "  >> Attack Sucess: [{}] | Margin Loss Check : [{}]".format(attack_success, adv_loss),
                log_file
            )
            print_and_log(
                "  >> Attack feasible: [{}] | Distance: {} | Preset Atttack Threshold {}]".format(feasable, distance, eps),
                log_file
            )
            print_and_log("  >> Check Vox Violation Calculation: [%d]" % num_violation, log_file)
            print_and_log("  >> Max value: [%.04f] | Min value: [%.04f]" % (
                torch.amax(adv_input).item(), torch.amin(adv_input).item()),
                log_file
            )

        # A feasible attack w.o. violation does not need continue
        if attack_success and feasable and (num_violation < 1):
            best_sample = adv_input
            best_loss = adv_loss
            best_dist = distance
            best_idx = key
            best_iter = total_iter
            need_continue = False
            return best_sample, best_loss, best_dist, best_idx, num_violation, need_continue, best_iter
        
        # Prioritize someone has smaller loss function
        if attack_success:
            if has_someone_succeeded:
                if distance < best_dist:
                    best_dist = distance

                    best_sample = adv_input
                    best_loss = adv_loss
                    best_idx = key
                    best_iter = total_iter
            else:
                has_someone_succeeded = True
                best_dist = distance

                best_sample = adv_input
                best_loss = adv_loss
                best_idx = key
                best_iter = total_iter
        # if no one has succeeded, record the one has the smallest adv loss achieved
        else:
            if has_someone_succeeded:
                pass
            else:
                if adv_loss > best_loss:
                    best_dist = distance

                    best_sample = adv_input
                    best_loss = adv_loss
                    best_idx = key
                    best_iter = total_iter
        
    if specific_key is None:
        print_and_log(
            "  >> Final Margin Loss Check : [{}]".format(best_loss),
            log_file
        )
        print_and_log(
            "  >> Final Distance: {} | Preset Atttack Threshold {}]".format(best_dist, eps),
            log_file
        )
        print_and_log(
            ">>>>>>>>> The best ES point to continue is Restart [%d] with distance [%.04f] <<<<<<<<<<<<<<<<<<" % (best_idx, best_dist),
            log_file
        )
    return best_sample, best_loss, best_dist, best_idx, num_violation, need_continue, best_iter


if __name__ == "__main__":
    clear_terminal_output()
    print("APGD Max Formulation Result...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'max_apgd_example.json'),
        help="Path to the json config file."
    )

    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    assert cfg["dataset"]["batch_size"] == 1, "Only want to support batch size = 1"

    default_dtype = torch.float
    granso_dtype = torch.double

    # Experiment Root
    save_root = os.path.join("..", "log_folder")
    root_name = cfg["log_folder"]["save_root"]
    save_root = os.path.join(save_root, root_name)
    makedir(save_root)

    # Experiment ID
    attack_config = cfg["test_attack"]
    attack_alg = attack_config["attack_type"]
    attack_type = attack_config["attack_method"]

    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]
    # Create Experiment Log Dir 
    exp_name = "%s-%s-%d-%d" % (
        attack_alg, attack_type,
        start_sample, end_sample,
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
    final_res_csv_dir = os.path.join(
        check_point_dir, "attack_result_log.csv"
    )
    final_summary = {
        "sample_id": [],
        "true_label": [],

        "adv_distance": [],
        "eps": [],
        "granso_adv_loss": [],
        "box_violation": [],
        "time": []
    }

    # === Setup Classifier Model ===
    cls_model_config = cfg["classifier_model"]
    classifier_model, msg = build_model(
        model_config=cls_model_config, 
        global_config=cfg, 
        device=device
    )
    classifier_model.eval()
    print_and_log(msg, log_file, mode="w")

    # ==== Construct the max loss for generating attack ====
    attack = get_lp_attack(
        attack_config, 
        classifier_model, 
        device, 
        global_config=cfg,
        is_train=False
    )
    # ==== Construct the original unclipped loss for evaluation ====
    # This marigin loss is only used to check if attack is successful or not
    # and is not used in performing attacks
    eval_loss_func, msg = get_loss_func_eval(
        "Margin", 
        reduction="none", 
        use_clip_loss=False
    )

    # ==== Get Clean Data Loader ====
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )

    # ==== Perform Attack ====
    attack_bound = attack_config["attack_bound"]

    # === Lists to save dataset ===
    orig_image_list = []
    adv_image_list = []
    err_image_list = []

    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_batch"]:
            # Option to select image to test
            # do nothing because these batches have been tested.
            pass
        else:
            if batch_idx > (cfg["end_batch_num"]-1):
                break
            print_and_log(
                "===== Testing Batch [%d] ======" % batch_idx, log_file
            )
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )
            # ==== fab init and granso continue attack ====
            inputs = inputs.to(device, dtype=default_dtype)
            labels = labels.to(device)

            with torch.no_grad():
                pred_logits = classifier_model(inputs)
                pred = pred_logits.argmax(1)
            
            attack_target = labels

            pred_correct = (pred == attack_target).sum().item()

            final_summary["sample_id"].append(batch_idx)
            
            if pred_correct < 0.5:
                print_and_log(
                    "Batch idx [%d] predicted wrong. Skip." % batch_idx,
                    log_file
                )
                for key in final_summary.keys():
                    # log something to indicate the pred is not correct
                    if key not in ["sample_id"]:
                        final_summary[key].append(-1e12)
            else:
                final_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now APT opt...",
                    log_file
                )

                t_start_granso = time.time()
                adv_output_dict = {}
                iter_log = {}
                if "FAB" in attack_type:
                    attacked_adv_input = generate_attack_lp(
                        inputs, labels, device, attack
                    )[0]
                else:
                    attacked_adv_input = generate_attack_lp(
                        inputs, labels, device, attack
                    )
                adv_output_dict[0] = attacked_adv_input
                iter_log[0] = 0
                time_end = time.time()
                feasibility_thres = attack_config["granso_early_feasibility_thres"]
                final_attack_sample, best_loss, best_distance, _, box_violations, _, _ = calc_restart_summary(
                    adv_output_dict,
                    iter_log,
                    inputs,
                    attack_type,
                    log_file,
                    classifier_model,
                    eval_loss_func,
                    labels,
                    None,
                    attack_bound,
                    feasibility_thres
                )                

                # Record Result
                final_summary["adv_distance"].append(best_distance)
                final_summary["eps"].append(attack_bound)
                final_summary["granso_adv_loss"].append(best_loss)
                final_summary["box_violation"].append(box_violations)
                final_summary["time"].append(time_end-t_start_granso)

                
                print_and_log(
                    "  -- OPT total time -- %.04f \n" % (
                        time_end-t_start_granso
                    ),
                    log_file
                )

                # ==== if true, save the attack samples to compute the patterns =====
                if cfg["save_vis"]:
                    vis_dir = os.path.join(
                        check_point_dir, "dataset_vis"
                    )
                    os.makedirs(vis_dir, exist_ok=True)

                    adv_image_np = tensor2img(final_attack_sample)
                    orig_image_np = tensor2img(inputs)

                    orig_image_list.append(orig_image_np)
                    adv_image_list.append(adv_image_np)

                    orig_save_name = os.path.join(
                        vis_dir, "orig_img_list.npy"
                    )
                    adv_save_name = os.path.join(
                        vis_dir, "adv_img_list.npy"
                    )   

                    np.save(
                        orig_save_name, np.asarray(orig_image_list)
                    )
                    np.save(
                        adv_save_name, np.asarray(adv_image_list)
                    )


        save_dict_to_csv(
            final_summary, final_res_csv_dir
        )