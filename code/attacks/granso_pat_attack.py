# This file realizes the pat attack by granso
import gc, torch, random
import numpy as np
# ==== Granso Import ====
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def pat_attack_granso(
    inputs, labels,
    model, attack_type,
    device, loss_func, eps,
    lpips_distance=None,
    max_iter=1000,
    max_clock_time=1000,
    ineq_tol=1e-5, 
    opt_tol=1e-5, 
    init_scale=0.005, 
    steering_c_viol=0.1,
    steering_c_mu=0.9,
    mu0=1,
    mem_size_param=100,
    input_constraint_type=None,
    print_log=True,
    dtype="double",
    rescale_ieq=False,
    random_first_step=False):

    assert input_constraint_type is not None, \
        "Need to specify how to deal with the input constraints. Optioins: [Box, Clamp, Sigmoid]"
    opts = pygransoStruct()
    if dtype == "double":
        opts.double_precision = True  # Do not use double precision
        dtype = torch.double
    elif dtype == "float":
        opts.double_precision = False
        dtype = torch.float
    else:
        raise RuntimeError("Specify the torch default dtype please.")

    # attack_type = attack_type  Not usefull for pat attack
    var_in = {"x_tilde": list(inputs.shape)}

    inputs = inputs.to(device, dtype=dtype)
    labels = labels.to(device)
    model = model.to(device, dtype=dtype)

    assert lpips_distance is not None, "This function is specified for this"
    lpips_distance = lpips_distance.to(device, dtype=dtype)

    if input_constraint_type == "Box":
        # print(">> Using Granso Box")
        comb_fn = lambda X_struct : user_fn_input_box_constraint(
            X_struct=X_struct, 
            inputs=inputs, 
            labels=labels,
            model=model,
            loss_func=loss_func,
            attack_type=attack_type,
            lpips_distance=lpips_distance,
            eps=eps, 
            rescale_ieq=rescale_ieq
        )
    # elif input_constraint_type == "Clamp":
    #     # print(">> Using Granso Clamp")
    #     comb_fn = lambda X_struct : user_fn_input_clamp(
    #         X_struct=X_struct, 
    #         inputs=inputs, 
    #         labels=labels,
    #         model=model,
    #         loss_func=loss_func,
    #         attack_type=attack_type,
    #         lpips_distance=lpips_distance,
    #         eps=eps,
    #         rescale_ieq=rescale_ieq
    #     )
    # elif input_constraint_type == "Sigmoid":
    #     # print(">> Using Granso Sigmoid")
    #     comb_fn = lambda X_struct : user_fn_input_sigmoid(
    #         X_struct=X_struct, 
    #         inputs=inputs, 
    #         labels=labels,
    #         model=model,
    #         loss_func=loss_func,
    #         attack_type=attack_type,
    #         lpips_distance=lpips_distance,
    #         eps=eps,
    #         rescale_ieq=rescale_ieq
    #     )
    elif input_constraint_type == "Fold-All":
        # print(">> Using L2-Max-Folding for all constraitns")
        comb_fn = lambda X_struct : user_fn_fold_all(
            X_struct=X_struct, 
            inputs=inputs, 
            labels=labels,
            model=model,
            loss_func=loss_func,
            attack_type=attack_type,
            lpips_distance=lpips_distance,
            eps=eps,
            rescale_ieq=rescale_ieq
        )
    else:
        raise RuntimeError("Undefined specification to deal with the input constraint.")
    
    
    opts.torch_device = device
    opts.maxit = max_iter
    opts.opt_tol = opt_tol
    opts.viol_ineq_tol = ineq_tol
    opts.steering_c_viol = steering_c_viol
    opts.steering_c_mu = steering_c_mu
    
    if rescale_ieq:  # if ieq is rescaled, the tol level should be decreased too
        num_pixels = np.prod(inputs.shape)
        opts.viol_ineq_tol = ineq_tol / np.sqrt(num_pixels)
    opts.maxclocktime = max_clock_time
    opts.mu0 = mu0
    if random_first_step:
        opts.init_step_size = 1 + 0.2 * random.random()

    opts.print_frequency = 1
    if not print_log:
        opts.print_level = 0
    opts.limited_mem_size = mem_size_param 

    # === Init Starting Position ===
    x0 = torch.reshape(
        inputs,
        (torch.numel(inputs), 1)
    )
    opts.x0 = (1-init_scale) * x0 + init_scale * torch.rand_like(x0)
    # if input_constraint_type == "Sigmoid":
    #     opts.x0 = inverse_sigmoid(opts.x0)

    # ==== Start Optimization ====
    try:
        soln = pygranso(
            var_spec=var_in,
            combined_fn=comb_fn,
            user_opts=opts)
    except:
        print("Skip Granso")
        soln = None
        # raise RuntimeError("Temporary Stop")
    
    # ==== Calculation Done ====
    # Collect Garbage
    gc.collect()
    return soln


# ==== Granso Attack Struct ====
def user_fn_input_box_constraint(
    X_struct, 
    inputs, labels,
    model, loss_func,
    attack_type,
    lpips_distance=None, 
    eps=0.5,
    rescale_ieq=False
    ):
    adv_inputs = X_struct.x_tilde
    epsilon = eps

    logits_outputs = model(adv_inputs)
    f = loss_func(logits_outputs, labels)

    # inequality constraint
    ci = pygransoStruct()

    err_vec = (inputs - adv_inputs).reshape(inputs.size()[0], -1)

    assert lpips_distance is not None, "must input lpips model."
    lpips_dists = lpips_distance(adv_inputs, inputs)
    ci.c1 = lpips_dists - epsilon

    
    # ===== Input Constraint for natural Image: I \in [0, 1] =====
    box_constr = torch.hstack(
        (adv_inputs.reshape(inputs.numel())-1,
        -adv_inputs.reshape(inputs.numel()))
    )
    box_constr = torch.clamp(box_constr, min=0)
    # === L2 folding normalization ====
    folded_constr =  torch.linalg.vector_norm(box_constr.reshape(-1), ord=2)
    ci.c2 = folded_constr

    # equality constraint
    ce = None
    return [f,ci,ce]


# ==== Fold all constraints into one ====
def user_fn_fold_all(
    X_struct, 
    inputs, labels,
    model, loss_func,
    attack_type,
    lpips_distance=None, 
    eps=0.5,
    rescale_ieq=False
):
    adv_inputs = X_struct.x_tilde
    epsilon = eps

    logits_outputs = model(adv_inputs)
    f = loss_func(logits_outputs, labels)

    # inequality constraint
    ci = pygransoStruct()

    err_vec = (inputs - adv_inputs).reshape(inputs.size()[0], -1)

    assert lpips_distance is not None, "must input lpips model."
    lpips_dists = lpips_distance(adv_inputs, inputs)
    constr1_vec = lpips_dists - epsilon

    # ===== Input Constraint for natural Image: I \in [0, 1] =====
    constr2_vec = torch.hstack(
        (adv_inputs.reshape(inputs.numel())-1,
        -adv_inputs.reshape(inputs.numel()))
    )

    # ==== Fold all constraint ====
    constr2_vec = torch.clamp(constr2_vec, min=0).reshape((1,-1))
    constr_vec = torch.cat((constr1_vec, constr2_vec), dim=1)
    ci.c1 = (torch.sum(constr_vec**2)**0.5)
    if rescale_ieq:
        num_pixels = torch.as_tensor(np.prod(constr_vec.shape))
        normalization_factor = (num_pixels**0.5 + 1e-12)
        ci.c1 = ci.c1 / normalization_factor

    # equality constraint
    ce = None
    return [f,ci,ce]