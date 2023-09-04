# This file realizes the min robust radius opt. by granso
import gc, torch
import numpy as np

# ==== Project Import ====
from models.blocks import _straightThroughClamp
from utils.general import print_and_log
from attacks.transform_functions import sine_transfor_box as sine_transform
# ==== Granso Import ====
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


# ==== Target Min =====
def granso_min_percept(
    inputs, labels,
    x0,
    target_label,
    model, attack_type,
    device,
    lpips_distance=None,
    max_iter=1000,
    ineq_tol=1e-8,
    eq_tol=1e-8, 
    opt_tol=1e-8, 
    steering_c_viol=0.1,
    steering_c_mu=0.9,
    stat_l2=True,
    steering_l1=True,
    mu0=1,
    mem_size_param=100,
    linesearch_maxit=None,
    print_log=True,
    dtype="double",
    input_constraint_type=None,
    constraint_folding_type="L2",
    granso_slack_variable=None,
    H0_init=None,
    is_continue=False,
    granso_opt_log_file=None
    ):
    assert lpips_distance is not None, "This function need"

    opts = pygransoStruct()
    if dtype == "double":
        opts.double_precision = True  # Do not use double precision
        dtype = torch.double
    elif dtype == "float":
        opts.double_precision = False
        dtype = torch.float
    else:
        raise RuntimeError("Specify the torch default dtype please.")
    
    attack_type = attack_type  # Reserved in case future need
    var_in = {"z": list(inputs.shape)}

    comb_fn = lambda X_struct : user_fn_min_separate_constraint(
        X_struct=X_struct, 
        inputs=inputs, 
        labels=labels,
        model=model,
        attack_type=attack_type,
        lpips_distance=lpips_distance,
        target_label=target_label,
        input_constraint_type=input_constraint_type,
        constraint_folding_type=constraint_folding_type,
        granso_slack_variable=granso_slack_variable,
        granso_opt_log_file=granso_opt_log_file
    )

    opts.torch_device = device
    opts.maxit = max_iter
    opts.opt_tol = opt_tol
    opts.viol_ineq_tol = ineq_tol
    opts.viol_eq_tol = eq_tol
    opts.steering_c_viol = steering_c_viol
    opts.steering_c_mu = steering_c_mu
    opts.stat_l2_model = stat_l2
    opts.steering_l1_model = steering_l1
    # === Add Line Search Params ===
    if linesearch_maxit is not None:
        opts.linesearch_maxit = linesearch_maxit
    opts.print_frequency = 1
    if not print_log:
        opts.print_level = 0
    opts.limited_mem_size = mem_size_param
    opts.mu0 = mu0
    # ==== Assigned Initialization ====
    opts.x0 = torch.reshape(
        x0,
        (-1, 1)
    )
    if H0_init is not None:
        opts.limited_mem_warm_start = H0_init
    # ==== Start Optimization ====
    soln = pygranso(
        var_spec=var_in,
        combined_fn=comb_fn,
        user_opts=opts
    )
    # ==== Calculation Done ====
    # Collect Garbage
    gc.collect()
    return soln


def user_fn_min_separate_constraint(
    X_struct, 
    inputs, labels,
    model,
    attack_type,
    lpips_distance,
    target_label=None,
    input_constraint_type=None,
    constraint_folding_type="L2",
    granso_slack_variable=None,
    granso_opt_log_file=None
    ):
    assert input_constraint_type in [
        None, "Clamp", "Box"
    ], "unidentified input constraint type for PyGranso."

    assert lpips_distance is not None, "Need lpips distance for this function."

    z = X_struct.z
    labels = labels.item()
    if target_label is not None:
        target_label = target_label.item() if type(target_label) == torch.Tensor else target_label
    
    adv_inputs = z
    # objective
    f = lpips_distance(adv_inputs, inputs).reshape(-1)
    
    # normalizing factor 
    num_pixels = torch.as_tensor(np.prod(adv_inputs.shape))
    normalization_factor = num_pixels**0.5

    # Equality constraint
    ce = None
    # Inequality constraint
    ci = pygransoStruct()
    # ci = None

    logits_outputs = model(adv_inputs)
    fc = logits_outputs[:, labels] # true class output
    # k = logits_outputs.shape[1]
    if target_label is not None:
        fl = logits_outputs[:, target_label] # attack target
        ci.c1 = (fc - fl)
    else:
        k = logits_outputs.shape[1]
        fl = torch.hstack(
            (
                logits_outputs[:, 0:labels],
                logits_outputs[:, labels+1:k]
            )
        )
        ci.c1 = fc - torch.amax(fl)
    
    if input_constraint_type == "Box":
        box_constr = torch.hstack(
            (adv_inputs.reshape(inputs.numel()) - 1,
            -adv_inputs.reshape(inputs.numel()))
        )
        box_constr = torch.clamp(box_constr, min=0)
        if constraint_folding_type == "L2":
            folded_constr = torch.linalg.vector_norm(box_constr.reshape(-1), ord=2)
        elif constraint_folding_type == "SumSquare":
            box_constr = box_constr.reshape(-1)
            folded_constr = torch.sum(box_constr ** 2)
        else:
            raise RuntimeError("Unimplemented Constraint Folding methods.")
        ci.c2 = folded_constr
        
    return [f,ci,ce]

