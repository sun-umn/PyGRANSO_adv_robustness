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
from attacks.transform_functions import sine_transfor_box as sine_transform
from models.model import AlexNetFeatureModel
from utils.distance import LPIPSDistance
from attacks.granso_min_percept import granso_min_percept
# ======= The following functions should be synced ======


def execute_granso_min_percept_target(
    input_to_granso, label_to_granso,
    target_label,
    x0,
    classifier_model, device,
    lpips_distance,
    print_opt, attack_config, mu0=1,
    H0_init=None,
    is_continue=False,
    granso_opt_log_file=None,
    max_iter=None
):

    if max_iter is None:
        max_iter = attack_config["granso_max_iter"]
    mem_size = attack_config["granso_mem"]
    input_constraint_type = attack_config["granso_input_constraint_type"]
    constraint_folding_type = attack_config["granso_constraint_folding_type"]

    # ==== how total violation and stationarity is determined ====
    stat_l2 = attack_config["granso_stat_l2"]
    steering_l1 = attack_config["granso_steering_l1"]
    granso_slack_variable = attack_config["granso_slack_variable"]

    sol = granso_min_percept(
        input_to_granso, label_to_granso,
        x0=x0,
        target_label=target_label,
        model=classifier_model, attack_type=None,
        device=device,
        lpips_distance=lpips_distance,
        stat_l2=stat_l2,
        steering_l1=steering_l1,
        max_iter=max_iter,
        mu0=mu0,
        input_constraint_type=input_constraint_type,
        constraint_folding_type=constraint_folding_type,
        granso_slack_variable=granso_slack_variable,
        mem_size_param=mem_size,
        print_log=print_opt,
        H0_init=H0_init,
        is_continue=is_continue,
        granso_opt_log_file=granso_opt_log_file
    )
    return sol


def get_granso_adv_output_percept(
    sol, input_constraint_type, input_to_granso
):
    if sol is None:
        granso_adv_output = -1 * torch.ones_like(input_to_granso)
    else:
        granso_adv_output = torch.reshape(
            sol.final.x,
            input_to_granso.shape
        )
    return granso_adv_output


def calc_min_dist_percept(
    adv_output, orig_input, 
    lpips_distance, log_file, 
    model, target_label, 
    attack_label=None,
    feasibility_thres=0.05
):
    keys = adv_output.keys()
    best_dist = float("inf")
    best_sample = None
    best_idx = None

    for key in keys:
        output = adv_output[key]

        # Check Logits
        is_close_to_boundary = False
        with torch.no_grad():
            pred = model(output)
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
        

        # =======
        distance = lpips_distance(
            output, orig_input
        ).item()

        # Check [0, 1] box constraint
        greater_than_1 = torch.sum(torch.where(output > (1 + 1e-4), 1, 0)).item()
        smaller_than_0 = torch.sum(torch.where(output < (0 - 1e-4), 1, 0)).item()
        num_violation = greater_than_1 + smaller_than_0
        print_and_log("        >>>> Check Vox Violation Calculation: [%d]" % num_violation, log_file)
        print_and_log("        >>>> Max value: [%.04f] | Min value: [%.04f]" % (
            torch.amax(output).item(), torch.amin(output).item()),
            log_file
        )

        if (distance < best_dist) :
            if is_close_to_boundary:  # Do not want failed sample to get recorded
                best_dist = distance
                best_sample = adv_output[key].clone()
                best_idx = key
                print_and_log(
                    "  >>>> Restart %d ===> [%.04f]" % (key, distance),
                    log_file
                ) 
            else:
                msg =  "  >>>> Restart %d ===> " % key
                msg += "A sample with Distance [%.04f] - Logits Diff [%.04f] not recorded" % (
                    distance, (attacked_logit - true_label_logit)
                )
                print_and_log(
                    msg, log_file
                )
        else:
            print_and_log(
                "  >>>> Restart %d ===> [%.04f]" % (key, distance),
                log_file
            )
    print_and_log(
        "  >>>>>>>>>>>>>> This target best distance is: [%.04f] <<<<<<<<<<<<<<<<<<" % best_dist,
        log_file
    )
    return best_sample, best_dist, num_violation, attack_label, best_idx


if __name__ == "__main__":
    clear_terminal_output()
    print("Validate Granso Untarget version of perceptual min.")
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
    exp_name = "G-Percept-Init-%.04f-%d-%d" % (
        granso_init_scale, 
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

    # ==== Init LPIPS Distance ====
    lpips_model = AlexNetFeatureModel(
        lpips_feature_layer=False, 
        use_clamp_input=False
    ).to(device=device, dtype=granso_dtype)
    lpips_model.eval()
    lpips_distance = LPIPSDistance(
        lpips_model, activation_distance="L2"
    ).to(device, dtype=granso_dtype)


    # ==== Get Clean Data Loader ====
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )
    num_classes = cfg["dataset"]["num_classes"]
    granso_init_summary = {
        "true_label": [],
        "clean_max_logit": [],

        "granso_best_target": [],
        "perceptual": [],
        "granso_adv_loss": [],

        "nat_img_pixel_violation": []
    }

    # ==== Perform Granso Untarget Attack ====
    attack_true_label = attack_config["attack_true_label"]
    for batch_idx, data in enumerate(val_loader):
        if batch_idx < cfg["curr_batch"]:
            pass
        else:
            if batch_idx > (cfg["end_batch_num"]-1):
                break
            print_and_log(
                "<== Testing Batch [%d] ===============================" % batch_idx, log_file
            )
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )

            # ======= Check if classified correctly =====
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
                    granso_init_summary[key].append(-10)
            else:
                granso_init_summary["clean_max_logit"].append(torch.amax(pred).item())
                granso_init_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now Granso opt...",
                    log_file
                )

                # Start Granso, Init Sample
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
                granso_x_init = {}
                granso_final_output_dict = {}

                for restart_num in range(granso_restarts):
                    applied_perturbation = granso_init_scale * (2 * torch.rand(input_to_granso.shape).to(device) - 1)
                    if input_constraint_type == "Sigmoid":
                        raise RuntimeError("Currently we dont want Sigmoid input. Try Sine_Transform")
                    else:
                        x_init = (input_to_granso + applied_perturbation).to(device, dtype=granso_dtype)
                    
                    granso_x_init[restart_num] = x_init.clone()
                    try:
                        warmup_max_iter = attack_config["granso_early_max_iter"]
                        sol = execute_granso_min_percept_target(
                            input_to_granso, label_to_granso,
                            None,
                            x_init,
                            classifier_model, device,
                            lpips_distance,
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
                    
                    granso_adv_output = get_granso_adv_output_percept(
                        sol, input_constraint_type, input_to_granso
                    )
                    granso_output_dict[restart_num] = granso_adv_output
                
                # Indent to calculate this every res is returned 
                feasibility_thres = attack_config["granso_early_feasibility_thres"]
                granso_final_output, granso_final_distance, granso_final_violation, target_idx, best_key = calc_min_dist_percept(
                    granso_output_dict, input_to_granso, lpips_distance, log_file, classifier_model, label_to_granso.item(),
                    feasibility_thres=feasibility_thres
                )
                print_and_log("   ============================================================ ", log_file)
                print_and_log("   Warm Restart with %d restart" % best_key, log_file)

                restart_x_init = granso_x_init[best_key]
                sol = execute_granso_min_percept_target(
                    input_to_granso, label_to_granso,
                    None,
                    restart_x_init,
                    classifier_model, device,
                    lpips_distance, 
                    print_opt, attack_config, 
                    mu0=mu0,
                    granso_opt_log_file=granso_opt_traj_log_file,
                )
                granso_adv_output = get_granso_adv_output_percept(
                    sol, input_constraint_type, input_to_granso
                )
                granso_final_output_dict[0] = granso_adv_output
                # Calculate the summary for this result
                granso_final_output, granso_final_distance, granso_final_violation, target_idx, _ = calc_min_dist_percept(
                    granso_final_output_dict, input_to_granso, lpips_distance, log_file, classifier_model, label_to_granso.item()
                )

                granso_best_distance = granso_final_distance
                granso_best_target_label = target_idx
                granso_best_adv_sample = granso_final_output
                granso_best_adv_violation = granso_final_violation

                granso_init_summary["perceptual"].append(granso_best_distance)
                t_end = time.time()
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
                
                granso_init_summary["granso_adv_loss"].append(adv_loss)
                granso_init_summary["granso_best_target"].append(granso_best_target_label)
                granso_init_summary["nat_img_pixel_violation"].append(granso_best_adv_violation)

        save_dict_to_csv(
            granso_init_summary, granso_init_csv_dir
        )
