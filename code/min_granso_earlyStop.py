import sys
sys.path.append("C:\\Users\\Liang\\AdvRobustnessGranso\\code")
sys.path.append("/home/jusun/liang656/AdvRobustnessGranso/code/")
sys.path.append("E:\\AdvRobustnessGranso\\code")
import argparse, os, torch, time
# =======
from utils.build import build_model, get_loss_func_eval, get_loader_clean
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.general import load_json, print_and_log, tensor2img, save_image
from utils.train import get_samples
import numpy as np
# ======= The following functions should be synced ========
from unit_test.validate_granso_target import execute_granso_min_target, get_granso_adv_output, \
    calc_min_dist_sample_fab


def get_granso_resume_info(sol):
    if sol is None:
        warm_H0 = None
    else:
        warm_H0 = sol.H_final
    return warm_H0


def calc_min_dist_sample(
    fab_adv_output, 
    granso_iter_dict,
    orig_input, 
    attack_type, log_file, 
    model, target_label, 
    attack_label=None,
    feasibility_thres=0.04
):
    keys = fab_adv_output.keys()
    best_dist = float("inf")
    best_sample = None
    best_idx = None
    best_iter = float("inf")

    for key in keys:
        fab_output = fab_adv_output[key]
        total_iter = granso_iter_dict[key]
        # Check Logits
        if feasibility_thres is not None:
            is_close_to_boundary = False
            with torch.no_grad():
                pred = model(fab_output)
                true_label_logit = pred[0, target_label].item()
            if attack_label is not None:
                attacked_logit = pred[0, attack_label].item()
            else:
                pred_clone = pred.clone()
                pred_clone[0, target_label] = -1e12
                attacked_logit = torch.amax(pred_clone).item()
                attack_label = pred_clone.argmax(1).item()
            if attacked_logit > (true_label_logit - feasibility_thres):
                is_close_to_boundary = True
        else:
            is_close_to_boundary = True
        
        fab_output = fab_output.clone().reshape(1, -1)
        orig_input = orig_input.clone().reshape(1, -1)

        # Check [0, 1] box constraint
        greater_than_1 = torch.sum(torch.where(fab_output > (1 + 1e-4), 1, 0)).item()
        smaller_than_0 = torch.sum(torch.where(fab_output < (0 - 1e-4), 1, 0)).item()
        num_violation = greater_than_1 + smaller_than_0
        print_and_log("        >>>> Check Vox Violation Calculation: [%d]" % num_violation, log_file)
        print_and_log("        >>>> Max value: [%.04f] | Min value: [%.04f]" % (
            torch.amax(fab_output).item(), torch.amin(fab_output).item()),
            log_file
        )
        
        err_vec = torch.abs(fab_output - orig_input)
        assert "L" in attack_type, "Norm character partition incorrect"

        if attack_type == "L1-Reform":
            attack_key_word = "L1"
        else:
            attack_key_word = attack_type
        p_norm = attack_key_word.split("L")[-1]
        if p_norm == "inf":
            p_distance = torch.amax(err_vec, dim=1).cpu().item()
        else:
            p_norm = float(p_norm)
            p_distance = (torch.sum(err_vec**p_norm, dim=1)**(1/p_norm)).cpu().item()
          
        
        if (p_distance < best_dist) :
            if is_close_to_boundary:  # Do not want failed sample to get recorded
                best_dist = p_distance
                best_sample = fab_adv_output[key].clone()
                best_idx = key
                best_iter = total_iter
                print_and_log(
                    "  >>>> Restart %d ===> [%.04f]" % (key, p_distance),
                    log_file
                ) 
            else:
                msg =  "  >>>> Restart %d ===> " % key
                msg += "A sample with Distance [%.04f] - Logits Diff [%.04f] not recorded" % (
                    p_distance, (attacked_logit - true_label_logit)
                )
                print_and_log(
                    msg, log_file
                )
        else:
            print_and_log(
                "  >>>> Restart %d ===> [%.04f]" % (key, p_distance),
                log_file
            )
    if best_idx is not None:
        print_and_log(
            "  >>>>>>>>>>>>>> This target best distance is Restart [%d] with distance [%.04f] <<<<<<<<<<<<<<<<<<" % (best_idx, best_dist),
            log_file
        )
    return best_sample, best_dist, num_violation, attack_label, best_idx, best_iter


if __name__ == "__main__":
    clear_terminal_output()
    print("Validate Granso Untarget version.")
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
    granso_init_scale = attack_config["granso_init_scale"]
    input_constraint_type = attack_config["granso_input_constraint_type"]
    constraint_folding_type = attack_config["granso_constraint_folding_type"]
    granso_slack_variable = attack_config["granso_slack_variable"]
    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]
    # Create Experiment Log Dir 
    exp_name = "G-Untarget-%s-%.03f-%s-%s-%s-%d-%d" % (
        attack_type, granso_init_scale, 
        input_constraint_type, constraint_folding_type,
        granso_slack_variable,
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
    granso_opt_traj_log_file = create_log_info(
        check_point_dir, "Granso_OPT_traj.txt"
    )

    # Create save csv dir
    granso_init_csv_dir = os.path.join(
        check_point_dir, "granso_init.csv"
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
    granso_init_summary = {
        "sample_idx": [],
        "true_label": [],
        "clean_max_logit": [],

        "granso_best_target": [],
        attack_type: [],
        "granso_adv_loss": [],

        "nat_img_pixel_violation": [],
        "iters": [],
        "time": []
    }

    # === Lists to save dataset ===
    orig_image_list = []
    adv_image_list = []

    # ==== Perform Granso Untarget Attack ====
    attack_true_label = attack_config["attack_true_label"]
    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_batch"]:
            # Option to select image to test
            # do nothing because these batches have been tested.
            pass
        else:
            if batch_idx > (cfg["end_batch_num"]-1):
                break
            print_and_log(
                "<== Testing Batch [%d] ===============================" % batch_idx, log_file
            )
            granso_init_summary["sample_idx"].append(batch_idx)
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
                for key in granso_init_summary.keys():
                    if key != "sample_idx":
                        granso_init_summary[key].append(-10)
            else:
                granso_init_summary["clean_max_logit"].append(torch.amax(pred).item())
                granso_init_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now Granso opt...",
                    log_file
                )

                # granso start with fab-like init
                input_to_granso = inputs.clone().to(device, dtype=granso_dtype)
                label_to_granso = attack_target.clone()
                classifier_model = classifier_model.to(device, dtype=granso_dtype)

                granso_restarts = attack_config["granso_restarts"]
                mu0 = attack_config["granso_mu0"]
                print_opt = True 

                # ===== to log the best info ====
                granso_best_distance, granso_best_target_label = float("inf"), None
                granso_best_adv_sample, granso_best_adv_violation = None, -1

                # ==== Granso Opt ====
                t_start_granso = time.time()
                granso_output_dict = {}
                granso_iter_dict = {}
                # granso_x_init = {}
                granso_x_interm_dict = {}
                granso_H_dict = {}
                granso_mu_dict = {}

                granso_final_output_dict = {}
                granso_final_iter_dict = {}
                # ===== Early Stopped Granso, pick the best one and then continue =====
                for restart_num in range(granso_restarts):
                    applied_perturbation = granso_init_scale * (2 * torch.rand_like(input_to_granso).to(device) - 1)
                    if input_constraint_type == "Sigmoid":
                        raise RuntimeError("Currently we dont want Sigmoid input. Try Sine_Transform")
                    else:
                        x_init = (input_to_granso + applied_perturbation).to(device, dtype=granso_dtype)
                    
                    # granso_x_init[restart_num] = x_init.clone()
                    try:
                        warmup_max_iter = attack_config["granso_early_max_iter"]
                        sol = execute_granso_min_target(
                            input_to_granso, label_to_granso,
                            None,
                            x_init,
                            classifier_model, device,
                            print_opt, attack_config, mu0=mu0,
                            granso_opt_log_file=granso_opt_traj_log_file,
                            max_iter=warmup_max_iter
                        )
                    except:
                        print_and_log(
                            "Granso OPT Failure ... ",
                            log_file
                        )
                        sol = None
                    granso_adv_output = get_granso_adv_output(
                        sol, attack_type, input_constraint_type, input_to_granso
                    )
                    granso_output_dict[restart_num] = granso_adv_output
                    if sol is not None:
                        granso_iter_dict[restart_num] = sol.iters
                        granso_H_dict[restart_num] = sol.H_final
                        granso_mu_dict[restart_num] = sol.final.mu
                        granso_x_interm_dict[restart_num] = sol.final.x
                    else:
                        granso_iter_dict[restart_num] = 0
                        granso_H_dict[restart_num] = None
                        granso_mu_dict[restart_num] = 1

                        applied_perturbation = granso_init_scale * (2 * torch.rand_like(input_to_granso).to(device) - 1)
                        x_init = (input_to_granso + applied_perturbation).to(device, dtype=granso_dtype)
                        granso_x_interm_dict[restart_num] = x_init.clone()
                    
                # Indent to calculate this every res is returned 
                feasibility_thres = attack_config["granso_early_feasibility_thres"]
                granso_final_output, granso_final_distance, granso_final_violation, target_idx, best_key, best_iter = calc_min_dist_sample(
                    granso_output_dict, granso_iter_dict,
                    inputs, attack_type, log_file, classifier_model, label_to_granso.item(),
                    feasibility_thres=feasibility_thres 
                )
                
                print_and_log("   ============================================================ ", log_file)
                if best_key is not None:
                    print_and_log("   Warm Restart with %d restart" % best_key, log_file)
                else:
                    best_key = 0
                    print_and_log("   Warm Restart with %d restart" % best_key, log_file)
                
                H_continue = granso_H_dict[best_key]
                x_continue = granso_x_interm_dict[best_key]
                mu_continue = granso_mu_dict[best_key]

                if H_continue["S"].shape[1] == H_continue["rho"].shape[1]:
                    try:
                        sol = execute_granso_min_target(
                            input_to_granso, label_to_granso,
                            None,
                            x_continue,
                            classifier_model, device,
                            print_opt, attack_config, mu0=mu_continue,
                            H0_init=H_continue,
                            is_continue=True,
                            granso_opt_log_file=None
                        )
                    except:
                        sol = None
                    granso_adv_output = get_granso_adv_output(
                        sol, attack_type, input_constraint_type, input_to_granso
                    )
                else:
                    granso_adv_output = granso_output_dict[best_key]
                t_end = time.time()

                granso_final_output_dict[0] = granso_adv_output
                if sol is not None:
                    granso_final_iter_dict[0] = sol.iters
                else:
                    granso_final_iter_dict[0] = float("inf")
                # Calculate the summary for this result
                granso_final_output, granso_final_distance, granso_final_violation, target_idx, best_iter = calc_min_dist_sample_fab(
                    granso_final_output_dict, granso_final_iter_dict,
                    inputs, attack_type, log_file, classifier_model, label_to_granso.item()
                )
                granso_best_distance = granso_final_distance
                granso_best_target_label = target_idx
                granso_best_adv_sample = granso_final_output
                granso_best_adv_violation = granso_final_violation
                granso_best_iter = best_iter

                granso_init_summary[attack_type].append(granso_best_distance)
                
                print_and_log(
                    "  -- Granso %d restart (early stop with 1 in depth OPT) total time -- %.04f" % (
                        granso_restarts, t_end-t_start_granso
                    ),
                    log_file
                )

                if granso_best_adv_sample is not None:
                    with torch.no_grad():
                        classifier_model = classifier_model.to(device, dtype=fab_dtype)
                        granso_best_adv_sample = granso_best_adv_sample.to(device, dtype=fab_dtype)
                        final_pred = classifier_model(granso_best_adv_sample)
                        # pred_target = final_pred.argmax(1).item()
                        adv_loss = min_loss_func(final_pred, label_to_granso).item()
                else:
                    adv_loss = -1e12
                    granso_best_target_label = -10
                    granso_best_adv_violation = -10
                    granso_best_adv_sample = granso_final_output_dict[0].to(device, dtype=fab_dtype)

                granso_init_summary["granso_adv_loss"].append(adv_loss)
                granso_init_summary["granso_best_target"].append(granso_best_target_label)
                granso_init_summary["nat_img_pixel_violation"].append(granso_best_adv_violation)
                granso_init_summary["iters"].append(granso_best_iter)
                granso_init_summary["time"].append(t_end - t_start_granso)

                # if granso_best_adv_sample is not None:
                #     img_orig = tensor2img(inputs)
                #     img_vis = tensor2img(granso_best_adv_sample)
                #     vis_orig_name = os.path.join(check_point_dir, "%d_orig.png" % batch_idx)
                #     vis_name = os.path.join(
                #         check_point_dir, "%d_adv_vis.png" % batch_idx
                #     )
                #     save_image(img_vis, vis_name)
                #     save_image(img_orig, vis_orig_name)
            # === Visualize Image ===
            if cfg["save_vis"]:
                vis_dir = os.path.join(
                    check_point_dir, "dataset_vis"
                )
                os.makedirs(vis_dir, exist_ok=True)

                # adv_image_np = np.clip(
                #     tensor2img(granso_best_adv_sample.to(device, dtype=fab_dtype)), 0, 1
                # )
                # orig_image_np = np.clip(
                #     tensor2img(inputs), 0, 1
                # )
                adv_image_np = tensor2img(granso_best_adv_sample.to(device, dtype=fab_dtype))
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
        save_dict_to_csv(
            granso_init_summary, granso_init_csv_dir
        )

        