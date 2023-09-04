# This file realizes the max attack by granso
import gc, torch, random
import numpy as np
# ==== Granso Import ====
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def granso_attack(
    inputs, labels,
    model, attack_type,
    device, loss_func, eps,
    lpips_distance=None,
    max_iter=1000,
    max_clock_time=1e6,
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

    attack_type = attack_type
    var_in = {"x_tilde": list(inputs.shape)}

    inputs = inputs.to(device)
    labels = labels.to(device)
    model = model.to(device)

    if lpips_distance is not None:
        lpips_distance = lpips_distance.to(device, dtype=torch.float)

    # if input_constraint_type == "Box":
    #     # print(">> Using Granso Box")
    #     comb_fn = lambda X_struct : user_fn_input_box_constraint(
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
    if input_constraint_type == "Clamp":
        # print(">> Using Granso Clamp")
        comb_fn = lambda X_struct : user_fn_input_clamp(
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
    elif input_constraint_type == "Sigmoid":
        # print(">> Using Granso Sigmoid")
        comb_fn = lambda X_struct : user_fn_input_sigmoid(
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
    # elif input_constraint_type == "Fold-All":
    #     # print(">> Using L2-Max-Folding for all constraitns")
    #     comb_fn = lambda X_struct : user_fn_fold_all(
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
    else:
        raise RuntimeError("Undefined specification to deal with the input constraint.")

    # comb_fn = lambda X_struct : user_fn_plain(
    #     X_struct=X_struct, 
    #     inputs=inputs, 
    #     labels=labels,
    #     model=model,
    #     loss_func=loss_func,
    #     attack_type=attack_type,
    #     lpips_distance=lpips_distance,
    #     eps=eps,
    #     rescale_ieq=rescale_ieq
    # )
    
    opts = pygransoStruct()
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

    if dtype == "double":
        opts.double_precision = True  # Do not use double precision
    elif dtype == "float":
        opts.double_precision = False
    else:
        raise RuntimeError("Specify the torch default dtype please.")

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
    if input_constraint_type == "Sigmoid":
        opts.x0 = inverse_sigmoid(opts.x0)

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


def single_gpu_granso_attack(
    input_param, return_dict
):  
    sample_idx = input_param["idx"]
    device_list = input_param["gpu_device_list"]
    config = input_param["config"]
    attack_config = config["train_attack"]

    max_iter = attack_config["granso_max_iter"]
    bound = attack_config["bound"]
    mu0 = attack_config["granso_mu0"]
    mem_size = attack_config["granso_mem"]
    attack_type = attack_config["attack_method"]
    init_scale = attack_config["granso_init_scale"]
    input_constraint_type = attack_config["granso_input_constraint_type"]  # [Box, Clamp, Sigmoid]
    rescale_ieq_constraint = attack_config["granso_rescale_ieq"]
    random_first_step = attack_config["granso_randomize_line_search"]

    print_log = False
    default_dtype = input_param["dtype"]

    idx = sample_idx % len(device_list)
    device = device_list[idx]
    device = torch.device(device)
    cpu_device = torch.device("cpu")
    # print(device)

    inputs = input_param["inputs"].to(device, dtype=default_dtype)
    labels = input_param["labels"].to(device, dtype=torch.long)
    base_model = input_param["base_model"].to(device, dtype=default_dtype)
    lpips_distance = input_param["lpips_distance"]
    if lpips_distance is not None:
        lpips_distance = lpips_distance.to(device, dtype=default_dtype)
    loss_func = input_param["max_loss_func"]

    sol = granso_attack(
        inputs=inputs,
        labels=labels,
        model=base_model,
        attack_type=attack_type,
        device=device,
        loss_func=loss_func,
        eps=bound,
        lpips_distance=lpips_distance,
        max_iter=max_iter,
        init_scale=init_scale, 
        mu0=mu0,
        mem_size_param=mem_size,
        input_constraint_type=input_constraint_type,
        print_log=print_log,
        dtype=config["torch_dtype"],
        rescale_ieq=rescale_ieq_constraint,
        random_first_step=random_first_step
    )
    if sol is not None:
        final_adv_input = torch.reshape(
            sol.final.x,
            inputs.shape
        ).detach().to(cpu_device, dtype=torch.float)
    else:
        final_adv_input = inputs.detach().to(cpu_device, dtype=torch.float)
    return_dict[sample_idx] = final_adv_input

    del cpu_device, device
    del inputs, labels, base_model, lpips_distance, loss_func, 


# ==== Granso Wrapped Functions ====
def inverse_sigmoid(p):
    y_inv = torch.log(p/(1-p))
    return y_inv


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
    if attack_type == 'Linf':
        # # ==== V1 =====
        constr1_vec = torch.clamp((torch.abs(err_vec) - epsilon), min=0)
    elif attack_type == "Linf-v2":
       #  ==== V2 =====
        constr1_vec = torch.hstack(
            (err_vec.reshape(inputs.numel())-epsilon,
            -err_vec.reshape(inputs.numel())-epsilon)
        )
        constr1_vec = torch.clamp(constr1_vec, min=0) 
    elif attack_type == "L1":
        l1_distance = torch.sum(torch.abs(err_vec), dim=1, keepdim=True)
        constr1_vec = torch.clamp((l1_distance - epsilon), min=0)
    elif "L" in attack_type:
        # ==== Get p norm number ====
        norm_p = float(attack_type.split("L")[-1])
        # ==== Compute err image ====
        lp_distance = torch.sum(torch.abs(err_vec)**norm_p, dim=1, keepdim=True)**(1/norm_p)
        constr1_vec = lp_distance - epsilon
    elif attack_type == "Percept":
        assert lpips_distance is not None, "must input lpips model."
        lpips_dists = lpips_distance(adv_inputs, inputs)
        constr1_vec = lpips_dists - epsilon
    else:
        raise RuntimeError("Undefined Granso Max Structure for this project. Please Check.")

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


# ==== Clamp the input to formulate natural input images ====
def user_fn_input_clamp(
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

    # ==== clamp value to [0, 1] as natural image input ====
    adv_inputs = torch.clamp(adv_inputs, 0, 1)

    logits_outputs = model(adv_inputs)
    f = loss_func(logits_outputs, labels)

    # inequality constraint
    ci = pygransoStruct()
    err_vec = (inputs - adv_inputs).reshape(inputs.shape[0], -1)
    if attack_type == 'Linf':        
        tmp = torch.clamp((torch.abs(err_vec) - epsilon), min=0)
        ci.c1 = (torch.sum(tmp**2) ** 0.5)
        # ci.c1 = torch.sum(tmp**2)
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    elif attack_type == "L1":
        l1_distance = torch.sum(torch.abs(err_vec), dim=1)
        ci.c1 = (l1_distance - epsilon)
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    elif "L" in attack_type:
        # ==== Get p norm number ====
        norm_p = float(attack_type.split("L")[-1])
        # ==== Compute err image ====
        lp_distance = torch.sum(torch.abs(err_vec)**norm_p, dim=1)**(1/norm_p)
        ci.c1 = lp_distance - epsilon
    elif attack_type == "Percept":
        assert lpips_distance is not None, "must input lpips model."
        lpips_dists = lpips_distance(adv_inputs, inputs)
        ci.c1 = lpips_dists - epsilon
    else:
        raise RuntimeError("Undefined Granso Max Structure for this project. Please Check.")

    # equality constraint
    ce = None
    return [f,ci,ce]


# ==== Use sigmoid for soft input constraint as natural images ====
def user_fn_input_sigmoid(
    X_struct, 
    inputs, labels,
    model, loss_func,
    attack_type,
    lpips_distance=None, 
    eps=0.5,
    rescale_ieq=False
    ):
    w = X_struct.x_tilde
    epsilon = eps

    # ==== Soft Clamp to [0, 1] as natural image input with sigmoind functions ====
    adv_inputs = torch.sigmoid(w)

    logits_outputs = model(adv_inputs)
    f = loss_func(logits_outputs, labels)

    # inequality constraint
    ci = pygransoStruct()

    err_vec = (inputs - adv_inputs).reshape(inputs.size()[0], -1)
    if attack_type == 'Linf':        
        # ==== L2 Folding of Linf Constraint ====
        tmp = torch.clamp((torch.abs(err_vec) - epsilon), min=0)
        ci.c1 = (torch.sum(tmp**2) ** 0.5)
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    elif attack_type == "L1":
        l1_distance = torch.sum(torch.abs(err_vec), dim=1)
        ci.c1 = (l1_distance - epsilon)
    elif "L" in attack_type:
        # ==== Get p norm number ====
        norm_p = float(attack_type.split("L")[-1])
        # ==== Compute err image ====
        lp_distance = torch.sum(torch.abs(err_vec)**norm_p, dim=1)**(1/norm_p)
        ci.c1 = lp_distance - epsilon

    elif attack_type == "Percept":
        assert lpips_distance is not None, "must input lpips model."
        lpips_dists = lpips_distance(adv_inputs, inputs)
        ci.c1 = lpips_dists - epsilon
    else:
        raise RuntimeError("Undefined Granso Max Structure for this project. Please Check.")

    # equality constraint
    ce = None
    return [f,ci,ce]


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
    if attack_type == 'Linf':
        # # ==== V1 =====
        tmp = torch.clamp((torch.abs(err_vec) - epsilon), min=0)
        ci.c1 = torch.sum(tmp**2) ** 0.5
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    elif attack_type == "Linf-v2":
       #  ==== V2: oriinal linf =====
        ci.c1 = torch.linalg.norm(err_vec, ord=float("inf"))
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / (2 * normalization_factor)
    elif attack_type == "L1":
        l1_distance = torch.sum(torch.abs(err_vec), dim=1)
        ci.c1 = (l1_distance - epsilon)
    elif "L" in attack_type:
        # ==== Get p norm number ====
        norm_p = float(attack_type.split("L")[-1])
        # ==== Compute err image ====
        lp_distance = torch.sum(torch.abs(err_vec)**norm_p, dim=1)**(1/norm_p)
        ci.c1 = lp_distance - epsilon
    elif attack_type == "Percept":
        assert lpips_distance is not None, "must input lpips model."
        lpips_dists = lpips_distance(adv_inputs, inputs)
        ci.c1 = lpips_dists - epsilon
    else:
        raise RuntimeError("Undefined Granso Max Structure for this project. Please Check.")
    
    # ===== Input Constraint for natural Image: I \in [0, 1] =====
    ci.c2 = torch.hstack(
        (adv_inputs.reshape(inputs.numel())-1,
        -adv_inputs.reshape(inputs.numel()))
    )
    # === L2 folding normalization ====
    ci.c2 = (torch.sum(torch.clamp(ci.c2, min=0)**2)**0.5) 
    if rescale_ieq:
        num_pixels = torch.as_tensor(np.prod(inputs.shape))
        normalization_factor = (num_pixels**0.5 + 1e-12)
        ci.c2 = ci.c2 / normalization_factor

    # equality constraint
    ce = None
    return [f,ci,ce]


def user_fn_plain(
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
    if attack_type == 'Linf':
        # # ==== V1 =====
        tmp = torch.clamp((torch.abs(err_vec) - epsilon), min=0)
        ci.c1 = torch.sum(tmp**2) ** 0.5
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    # elif attack_type == "Linf-v2":
       #  ==== V2: oriinal linf =====
        # ci.c1 = torch.linalg.norm(err_vec, ord=float("inf"), dim=1)
        # if rescale_ieq:
        #     num_pixels = torch.as_tensor(np.prod(inputs.shape))
        #     normalization_factor =np.sqrt(num_pixels**0.5 + 1e-12)
        #     ci.c1 = ci.c1 * normalization_factor
    elif attack_type == "L1":
        l1_distance = torch.sum(torch.abs(err_vec), dim=1)
        ci.c1 = (l1_distance - epsilon)
        if rescale_ieq:
            num_pixels = torch.as_tensor(np.prod(inputs.shape))
            normalization_factor = (num_pixels**0.5 + 1e-12)
            ci.c1 = ci.c1 / normalization_factor
    elif "L" in attack_type:
        # ==== Get p norm number ====
        norm_p = float(attack_type.split("L")[-1])
        # ==== Compute err image ====
        lp_distance = torch.sum(torch.abs(err_vec)**norm_p, dim=1)**(1/norm_p)
        ci.c1 = lp_distance - epsilon
    elif attack_type == "Percept":
        assert lpips_distance is not None, "must input lpips model."
        lpips_dists = lpips_distance(adv_inputs, inputs)
        ci.c1 = lpips_dists - epsilon
    else:
        raise RuntimeError("Undefined Granso Max Structure for this project. Please Check.")
    
    # ===== Input Constraint for natural Image: I \in [0, 1] =====
    # ===== Input Constraint for natural Image: I \in [0, 1] =====
    constr2_vec = torch.hstack(
        (adv_inputs.reshape(inputs.numel())-1,
        -adv_inputs.reshape(inputs.numel()))
    )
    ci.c2 = constr2_vec

    # ==== Fold all constraint ====
    constr2_vec = torch.clamp(constr2_vec, min=0).reshape((1,-1))
    # ci.c2 = (torch.sum(constr2_vec**2)**0.5)
    # if rescale_ieq:
    #     num_pixels = torch.as_tensor(np.prod(constr2_vec.shape))
    #     normalization_factor = (num_pixels**0.5 + 1e-12)
    #     ci.c2 = ci.c2 / normalization_factor
    # equality constraint
    ce = None
    return [f,ci,ce]


if __name__ == "__main__":
    # ==== Inverse Sigmoid ====
    p = 1
    y_inv = torch.log(p/(1-p))

    print()

