
import sys
sys.path.append("C:\\Users\\Liang\\AdvRobustnessGranso\\code")
sys.path.append("/home/jusun/liang656/AdvRobustnessGranso/code/")
sys.path.append("E:\\AdvRobustnessGranso\\code")
import argparse, os, torch, time
import numpy as np
import pandas as pd
# ===========
from utils.general import tensor2img, rescale_array
from utils.general import load_json, print_and_log
from config_setups.config_log_setup import makedir
from config_setups.config_log_setup import clear_terminal_output, create_log_info,\
    makedir, save_dict_to_csv, save_exp_info, set_default_device
from utils.build import build_model, get_loss_func_eval, get_granso_loss_func, get_loader_clean
from utils.train import get_samples
from attacks.granso_max_attack import granso_max_attack
from models.model import AlexNetFeatureModel
from percept_utils.distance import LPIPSDistance


def get_granso_adv_output_maxform(sol, input_to_granso):
    if sol is None:
        granso_adv_output = -1 * torch.ones_like(input_to_granso)
    else:
        granso_adv_output = torch.reshape(
            sol.final.x,
            input_to_granso.shape
        )
    return granso_adv_output


def execute_granso_max_attack(
    input_to_granso, label_to_granso,
    x0,
    classifier_model,
    max_loss_function,
    device,
    print_opt, attack_config, mu0=1,
    lpips_model=None,
    H0_init=None,
    is_continue=False,
    max_iter=None,
    dtype=torch.double
):
    attack_type = attack_config["attack_method"]
    input_constraint_type = attack_config["granso_input_constraint_type"]
    eps = attack_config["attack_bound"]
    if max_iter is None:
        max_iter = attack_config["granso_max_iter"]
    mem_size = attack_config["granso_mem"]
    # ==== how total violation and stationarity is determined ====
    stat_l2 = attack_config["granso_stat_l2"]
    steering_l1 = attack_config["granso_steering_l1"]
    granso_walltime = attack_config["granso_walltime"]
    eq_tol = attack_config["granso_eq_tol"]
    ieq_tol = attack_config["granso_ieq_tol"]
    opt_tol = attack_config["granso_opt_tol"]

    time_start = time.time()
    sol = granso_max_attack(
        inputs=input_to_granso,
        labels=label_to_granso,
        x0=x0,
        model=classifier_model,
        attack_type=attack_type,
        device=device,
        loss_func=max_loss_function,
        eps=eps,
        lpips_distance=lpips_model,
        max_iter=max_iter,
        eq_tol=eq_tol,
        ieq_tol=ieq_tol,
        opt_tol=opt_tol, 
        stat_l2=stat_l2,
        steering_l1=steering_l1,
        mu0=mu0,
        H0_init=H0_init,
        print_log=print_opt,
        mem_size_param=mem_size,
        maxclocktime=granso_walltime,
        input_constraint_type=input_constraint_type,
        dtype=dtype
    )
    time_end = time.time()
    print(">> PyGranso Execution Time: [%.05f]" % (time_end-time_start))
    return sol


def calc_distance(sample_1, sample_2, attack_type, lpips_model=None):
    if attack_type == "PAT":
        assert lpips_model is not None, "Need lpips model input"
        distance = lpips_model(sample_1, sample_2)
        return distance.item()

    delta_vec = (sample_1 - sample_2).reshape(-1)
    if attack_type == "Linf":
        distance = torch.linalg.vector_norm(delta_vec, ord=float("inf"))
    elif attack_type == "L1":
        distance = torch.linalg.vector_norm(delta_vec, ord=1)
    elif attack_type == "L2":
        distance = torch.linalg.vector_norm(delta_vec, ord=2)
    elif "L" in attack_type:
        norm_p = float(attack_type.split("L")[-1])
        distance = torch.sum(torch.abs(delta_vec)**norm_p) ** (1/norm_p)
    else:
        raise RuntimeError("Error in calculating norm")
    
    distance = distance.item()
    return distance


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
    print("Eval PyGranso Max Formulation Attack...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'generate_attack_visualization.json'),
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
    attack_type = attack_config["attack_method"]

    start_sample = cfg["curr_batch"]
    end_sample = cfg["end_batch_num"]
    # Create Experiment Log Dir 
    exp_name = "G-%s-%d-%d" % (
        attack_type,
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
    if attack_type == "PAT":
        lpips_backbone = AlexNetFeatureModel(
            lpips_feature_layer=False, 
            use_clamp_input=False,
            dtype=granso_dtype
        ).to(device=device, dtype=granso_dtype)
        lpips_backbone.eval()
        lpips_model = LPIPSDistance(lpips_backbone).to(device=device, dtype=granso_dtype)
        lpips_model.eval()
    else:
        lpips_model=None
    
    granso_walltime = attack_config["granso_walltime"]
    print_and_log(
        "Granso Wall Time Settings {}".format(granso_walltime),
        log_file
    )
    
    # Create save csv dir
    final_res_csv_dir = os.path.join(
        check_point_dir, "attack_result_log.csv"
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

    # ==== Construct the max loss for generating attack ====
    attack_loss_config = attack_config["granso_loss"]
    granso_attack_loss_func_name = attack_loss_config["type"]
    reduction = attack_loss_config["reduction"]
    use_clip_loss = attack_loss_config["use_clip_loss"]
    dataset_name = cfg["dataset"]["name"]
    granso_attack_loss_func, msg = get_granso_loss_func(
        granso_attack_loss_func_name, 
        reduction, 
        use_clip_loss,
        dataset_name=dataset_name
    )
    # ==== Construct the original unclipped loss for evaluation ====
    eval_loss_func, msg = get_loss_func_eval(
        "Margin", 
        reduction="none", 
        use_clip_loss=False
    )

    # ==== Get Clean Data Loader ====
    _, val_loader, _ = get_loader_clean(
        cfg, only_val=True, shuffle_val=False
    )

    final_summary = {
        "sample_id": [],
        "true_label": [],

        "adv_distance": [],
        "eps": [],
        "granso_adv_loss": [],
        "box_violation": [],

        "iters": [],
        "time": [],
    }

    # ==== Perform Max Attack ====
    attack_true_label = attack_config["attack_true_label"]
    granso_init_scale = attack_config["granso_init_scale"]

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
                "<== Testing Batch [%d] ===============================" % batch_idx, log_file
            )
            inputs, labels = get_samples(
                cfg,
                data_from_loader=data
            )

            # ==== Get samples and prepare OPT ====
            inputs = inputs.to(device, dtype=default_dtype)
            labels = labels.to(device)
            classifier_model = classifier_model.to(device, dtype=default_dtype)
            with torch.no_grad():
                pred_logits = classifier_model(inputs)
                pred = pred_logits.argmax(1)
            
            if attack_true_label:
                attack_target = labels
            else:
                attack_target = pred
            pred_correct = (pred == attack_target).sum().item()

            final_summary["sample_id"].append(batch_idx)
            if pred_correct < 0.5:
                print_and_log(
                    "Batch idx [%d] predicted wrong. Skip." % batch_idx,
                    log_file
                )
                for key in final_summary.keys():
                    if key not in ["sample_id"]:
                        final_summary[key].append(-1e12)
            else:
                final_summary["true_label"].append(pred.item())
                print_and_log(
                    "Prediction Correct, now Granso opt...",
                    log_file
                )

                input_to_granso = inputs.clone().to(device, dtype=granso_dtype)
                label_to_granso = attack_target.clone()
                classifier_model = classifier_model.to(device, dtype=granso_dtype)
                
                granso_restarts = attack_config["granso_restarts"]
                mu0 = attack_config["granso_mu0"]
                print_opt = True

                # ===== to log the best info ====
                granso_best_distance, granso_best_target_label = float("inf"), None
                granso_best_adv_sample, granso_best_adv_violation = None, -1

                # ===== Granso Opt =====
                granso_opt_time = 0
                
                granso_output_dict = {}
                granso_iter_dict = {}
                # granso_x_init = {}
                granso_x_interm_dict = {}
                granso_H_dict = {}
                granso_mu_dict = {}
                granso_has_converged_dict = {}

                granso_final_output_dict = {}
                granso_final_iter_dict = {}
                # ===== Early Stopped Granso, pick the best one and then continue =====
                for restart_num in range(granso_restarts):
                    t_1 = time.time()
                    applied_perturbation = granso_init_scale * (2 * torch.rand_like(input_to_granso).to(device) - 1)
                    x_init = (input_to_granso + applied_perturbation).to(device, dtype=granso_dtype)
                    # granso_x_init[restart_num] = x_init.clone()
                    # try:
                    warmup_max_iter = attack_config["granso_early_max_iter"]
                    sol = execute_granso_max_attack(
                        input_to_granso, label_to_granso,
                        x_init,
                        classifier_model, 
                        granso_attack_loss_func,
                        device,
                        print_opt, attack_config, mu0=mu0,
                        lpips_model=lpips_model,
                        max_iter=warmup_max_iter,
                        dtype=granso_dtype
                    )
                    # except:
                    #     print_and_log(
                    #         "Granso OPT Failure ... ",
                    #         log_file
                    #     )
                    #     sol = None
                    granso_adv_output = get_granso_adv_output_maxform(
                        sol, input_to_granso
                    )
                    granso_output_dict[restart_num] = granso_adv_output
                    if sol is not None:
                        granso_iter_dict[restart_num] = sol.iters
                        granso_H_dict[restart_num] = sol.H_final
                        granso_mu_dict[restart_num] = sol.final.mu
                        granso_x_interm_dict[restart_num] = sol.final.x
                        print(sol.H_final["S"].shape, sol.H_final["Y"].shape, sol.H_final["rho"].shape)
                        # === If code 0 is reached before es_max_it, then continue is unnecessary
                        print(sol.H_final["S"].shape, sol.H_final["Y"].shape, sol.H_final["rho"].shape)
                        has_converged = (sol.H_final["S"].shape[1] != sol.H_final["rho"].shape[1])
                        granso_has_converged_dict[restart_num] = has_converged
                    else:
                        granso_iter_dict[restart_num] = float("inf")
                        granso_H_dict[restart_num] = None
                        granso_mu_dict[restart_num] = None
                        granso_x_interm_dict[restart_num] = x_init.clone()
                        granso_has_converged_dict[restart_num] = False
                    
                    t_2 = time.time()
                    granso_opt_time += (t_2) - (t_1)

                    # Indent to calculate this every res is returned
                    attack_bound = attack_config["attack_bound"]
                    feasibility_thres = attack_config["granso_early_feasibility_thres"]
                    granso_best_es_sample, best_loss, best_distance, best_idx, box_violations, need_continue, best_iter = calc_restart_summary(
                        granso_output_dict,
                        granso_iter_dict,
                        input_to_granso,
                        attack_type,
                        log_file,
                        classifier_model,
                        eval_loss_func,
                        label_to_granso,
                        lpips_model,
                        attack_bound,
                        feasibility_thres,
                        specific_key=restart_num
                    )

                    if need_continue:
                        pass
                    else:
                        break

                if need_continue:
                    # Indent to calculate this every res is returned
                    attack_bound = attack_config["attack_bound"]
                    feasibility_thres = attack_config["granso_early_feasibility_thres"]
                    granso_best_es_sample, best_loss, best_distance, best_idx, box_violations, need_continue, best_iter = calc_restart_summary(
                        granso_output_dict,
                        granso_iter_dict,
                        input_to_granso,
                        attack_type,
                        log_file,
                        classifier_model,
                        eval_loss_func,
                        label_to_granso,
                        lpips_model,
                        attack_bound,
                        feasibility_thres,
                    )

                # If not find a successful attack within the es phase, warm continue with the best one
                if need_continue:
                    print_and_log(">>>>>>> >>>>>>> Warm Restart with Restart [%d]" % best_idx, log_file)
                    # restart_x_init = granso_x_init[best_idx]

                    # sol = execute_granso_max_attack(
                    #     input_to_granso, label_to_granso,
                    #     restart_x_init,
                    #     classifier_model,
                    #     granso_attack_loss_func,
                    #     device,
                    #     print_opt, attack_config, mu0=mu0,
                    #     lpips_model=lpips_model,
                    #     H0_init=None,
                    #     is_continue=False,
                    #     max_iter=None,
                    #     dtype=granso_dtype
                    # )
                    t_3 = time.time()
                    H_continue = granso_H_dict[best_idx]
                    x_continue = granso_x_interm_dict[best_idx]
                    mu_continue = granso_mu_dict[best_idx]
                    has_converged = granso_has_converged_dict[best_idx]
                    if not has_converged:
                        sol = execute_granso_max_attack(
                            input_to_granso, label_to_granso,
                            x_continue,
                            classifier_model,
                            granso_attack_loss_func,
                            device,
                            print_opt, attack_config, mu0=mu_continue,
                            lpips_model=lpips_model,
                            H0_init=H_continue,
                            is_continue=True,
                            max_iter=None,
                            dtype=granso_dtype
                        )
                    else:
                        sol = None
                    t_4 = time.time()
                    granso_opt_time += (t_4 - t_1)

                    granso_adv_output = get_granso_adv_output_maxform(
                        sol, input_to_granso
                    )
                    granso_final_output_dict[0] = granso_adv_output
                    if sol is not None:
                        granso_final_iter_dict[0] = sol.iters
                    else:
                        granso_final_iter_dict[0] = float("inf")
                    granso_best_es_sample, best_loss, best_distance, _, box_violations, _, best_iter = calc_restart_summary(
                        granso_final_output_dict,
                        granso_final_iter_dict,
                        input_to_granso,
                        attack_type,
                        log_file,
                        classifier_model,
                        eval_loss_func,
                        label_to_granso,
                        lpips_model,
                        attack_bound,
                        feasibility_thres
                    )
                    

                # Record Result
                final_summary["adv_distance"].append(best_distance)
                final_summary["eps"].append(attack_bound)
                final_summary["granso_adv_loss"].append(best_loss)
                final_summary["box_violation"].append(box_violations)
                final_summary["iters"].append(best_iter)
                final_summary["time"].append(granso_opt_time)

                print_and_log(
                    "  -- Granso %d restart (early stop with 1 in depth OPT) total time -- %.04f \n" % (
                        granso_restarts, granso_opt_time
                    ),
                    log_file
                )

                # ==== Save Visualization =====
                if cfg["save_vis"]:
                    vis_dir = os.path.join(
                        check_point_dir, "dataset_vis"
                    )
                    os.makedirs(vis_dir, exist_ok=True)

                    # err_image_tensor = (granso_best_es_sample - inputs)
                    # err_image_np = tensor2img(err_image_tensor)

                    # adv_image_np = np.clip(
                    #     tensor2img(granso_best_es_sample.to(device, dtype=torch.float)), 0, 1
                    # )
                    # orig_image_np = np.clip(
                    #     tensor2img(inputs), 0, 1
                    # )
                    adv_image_np = tensor2img(granso_best_es_sample.to(device, dtype=torch.float))
                    orig_image_np = tensor2img(inputs)

                    orig_image_list.append(orig_image_np)
                    adv_image_list.append(adv_image_np)
                    # err_image_list.append(err_image_np)

                    # err_save_name = os.path.join(
                    #     vis_dir, "err_img_list.npy"
                    # )
                    orig_save_name = os.path.join(
                        vis_dir, "orig_img_list.npy"
                    )
                    adv_save_name = os.path.join(
                        vis_dir, "adv_img_list.npy"
                    )   
                    # np.save(
                    #     err_save_name, np.asarray(err_image_list)
                    # )
                    np.save(
                        orig_save_name, np.asarray(orig_image_list)
                    )
                    np.save(
                        adv_save_name, np.asarray(adv_image_list)
                    )


                    print()
                
        save_dict_to_csv(
            final_summary, final_res_csv_dir
        )
    
