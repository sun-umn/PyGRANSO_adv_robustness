#  ==== This file documents a few Normal PGD based Lp Attacks
#  ==== For benchmarking convenience
import functools, torch
from torch import nn
from operator import mul
from torch import optim
from advex_uar.common.pyt_common import get_attack as get_uar_attack
from advex_uar.attacks.attacks import InverseImagenetTransform
# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap 
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st
# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs
# ====
from attacks.pat_attacks import run_attack_with_random_targets
# ====== Something New =====
import time, math
import torch.nn.functional as F
from autoattack.autopgd_base import L1_projection, L0_norm, check_zero_gradients
from loss.losses import MarginLossOrig


PGD_ITERS = 20
DATASET_NUM_CLASSES = {
    'cifar': 10,
    'imagenet100': 100,
    'imagenet': 1000,
    'bird_or_bicycle': 2,
}


# ==== Do Not Attack in Perceptual Adversarial Robustness Paper ====
class NoAttack(nn.Module):
    """
    Attack that does nothing.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return inputs


# ==== Below are UAR attacks from paper: https://github.com/ddkang/advex-uar
class MisterEdAttack(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, randomize=False,
                 perturbation_norm_loss=False, lr=0.001, random_targets=False,
                 num_classes=None, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.randomize = randomize
        self.perturbation_norm_loss = perturbation_norm_loss
        self.attack_kwargs = kwargs
        self.lr = lr
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.attack = None

    def _setup_attack(self):
        cw_loss = lf.CWLossF6(self.model, self.normalizer, kappa=float('inf'))
        if self.random_targets:
            cw_loss.forward = functools.partial(cw_loss.forward, targeted=True)
        perturbation_loss = lf.PerturbationNormLoss(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = lf.RegularizedLoss({
            'cw': cw_loss,
            'pert': perturbation_loss,
        }, {
            'cw': 1.0,
            'pert': pert_factor,
        }, negate=True)

        self.pgd_attack = aa.PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'signed': False,
            'verbose': False,
            'num_iterations': 0 if self.randomize else PGD_ITERS,
            'random_init': self.randomize,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = advtrain.AdversarialAttackParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        if self.random_targets:
            return run_attack_with_random_targets(
                lambda inputs, labels: self.attack.attack(inputs, labels)[0],
                self.model,
                inputs,
                labels,
                num_classes=self.num_classes,
            )
        else:
            return self.attack.attack(inputs, labels)[0]


class UARModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        inverse_transform = InverseImagenetTransform(x.size()[-1])
        return self.model(inverse_transform(x) / 255)


class UARAttack(nn.Module):
    """
    One of the attacks from the paper "Testing Robustness Against Unforeseen
    Adversaries".
    """

    def __init__(self, model, dataset_name, attack_name, bound,
                 num_iterations=PGD_ITERS, step=None, random_targets=False,
                 randomize=False):
        super().__init__()

        assert randomize is False

        if step is None:
            step = bound / (num_iterations ** 0.5)

        self.random_targets = random_targets
        self.num_classes = DATASET_NUM_CLASSES[dataset_name]
        if (
           dataset_name.startswith('imagenet')
           or dataset_name == 'bird_or_bicycle'
        ):
            dataset_name = 'imagenet'
        elif dataset_name == 'cifar':
            dataset_name = 'cifar-10'

        self.model = model
        self.uar_model = UARModel(model)
        self.attack_name = attack_name
        self.bound = bound
        self.attack_fn = get_uar_attack(dataset_name, attack_name, eps=bound,
                                        n_iters=num_iterations,
                                        step_size=step, scale_each=1)
        self.attack = None

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        if self.attack_name == 'pgd_linf':
            dist = (inputs - adv_inputs).reshape(inputs.size()[0], -1) \
                .abs().max(1)[0] * 255
        elif self.attack_name == 'pgd_l2':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                ** 2
            ).sum(1).sqrt() * 255
        elif self.attack_name == 'fw_l1':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                .abs().sum(1)
                * 255 / functools.reduce(mul, inputs.size()[1:])
            )
        else:
            raise NotImplementedError()

        return dist <= self.bound

    def forward(self, inputs, labels):
        self.uar_model.training = self.model.training

        if self.attack is None:
            self.attack = self.attack_fn()
            self.attack.transform = LambdaLayer(lambda x: x / 255)
            self.attack.inverse_transform = LambdaLayer(lambda x: x * 255)

        if self.random_targets:
            attack = lambda inputs, targets: self.attack(
                self.uar_model,
                inputs,
                targets,
                avoid_target=False,
                scale_eps=False,
            )
            adv_examples = run_attack_with_random_targets(
                attack, self.model, inputs, labels, self.num_classes,
            )
        else:
            adv_examples = self.attack(self.uar_model, inputs, labels,
                                       scale_eps=False, avoid_target=True)

        # Some UAR attacks produce NaNs, so try to get rid of them here.
        perturbations = adv_examples - inputs
        perturbations[torch.isnan(perturbations)] = 0
        return (inputs + perturbations).detach()


class LinfAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8,
                'imagenet100': 8,
                'imagenet': 8,
                'bird_or_bicycle': 16,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_linf',
            bound=bound,
            **kwargs,
        )


class L2Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 255,
                'imagenet100': 3 * 255,
                'imagenet': 3 * 255,
                'bird_or_bicycle': 10 * 255,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_l2',
            bound=bound,
            **kwargs,
        )


class L1Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 0.5078125,
                'imagenet100': 1.016422,
                'imagenet': 1.016422,
                'bird_or_bicycle': 1.016422,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='fw_l1',
            bound=bound,
            **kwargs,
        )


class JPEGLinfAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 0.25,
                'imagenet100': 0.5,
                'imagenet': 0.5,
                'bird_or_bicycle': 0.5,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='jpeg_linf',
            bound=bound,
            **kwargs,
        )


class FogAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=512, **kwargs):
        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='fog',
            bound=bound,
            **kwargs,
        )


class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )


class ReColorAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.06, **kwargs):
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(pt.ReColorAdv, {
                'xform_class': ct.FullSpatial,
                'cspace': cs.CIELUVColorSpace(),
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_params': {
                  'resolution_x': 16,
                  'resolution_y': 32,
                  'resolution_z': 32,
                },
                'use_smooth_loss': True,
            }),
            perturbation_norm_loss=0.0036 / bound,
            **kwargs,
        )


# ==== Helper Function to get easy access to AutoAttack ====
class AutoAttack(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        self.model = model
        self.kwargs = kwargs
        self.attack = None

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            try:
                import autoattack
            except ImportError:
                raise RuntimeError(
                    'Error: unable to import autoattack. Please install the '
                    'package by running '
                    '"pip install git+git://github.com/fra31/auto-attack#egg=autoattack".'
                )
            self.attack = autoattack.AutoAttack(
                self.model, device=inputs.device, **self.kwargs)

        return self.attack.run_standard_evaluation(inputs, labels)


class AutoLinfAttack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8/255,
                'imagenet100': 8/255,
                'imagenet': 8/255,
                'bird_or_bicycle': 16/255,
            }[dataset_name]

        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            **kwargs,
        )


class AutoL2Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 1,
                'imagenet100': 3,
                'imagenet': 3,
                'bird_or_bicycle': 10,
            }[dataset_name]

        super().__init__(
            model,
            norm='L2',
            eps=bound,
            **kwargs,
        )


class AutoL1Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 1,
                'imagenet100': 3,
                'imagenet': 3,
                'bird_or_bicycle': 10,
            }[dataset_name]

        super().__init__(
            model,
            norm='L1',
            eps=bound,
            **kwargs,
        )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# ===== Adapted AutoAttack with Margin Loss for benchmarking =====
class APGDAttackMargin():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='Margin',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
        
        print("Loss Type:", self.loss)
    
    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def attack_single_run(self, x, y, x_init=None):
        best_loss_log, best_iter_log = -float("inf"), 0
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            # t = 0.2 * torch.ones(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            elif "Margin" in self.loss:
                if "Clip" in self.loss:
                    criterion_indiv = MarginLossOrig(reduction="none", use_clip_loss=True)
                else:
                    criterion_indiv = MarginLossOrig(reduction='none', use_clip_loss=False)
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            elif "Margin" in self.loss:
                if "Clip" in self.loss:
                    criterion_indiv = MarginLossOrig(reduction="none", use_clip_loss=True)
                else:
                    criterion_indiv = MarginLossOrig(reduction='none', use_clip_loss=False)
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                        -1, *[1]*(len(x.shape) - 1)) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.

            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            if loss_best.sum() > best_loss_log:
                best_iter_log = i+1
                best_loss_log = loss_best.sum()

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()

                # ==== 
                # print(loss_best.shape, y1.shape)
                
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    if self.norm in ['Linf', 'L2']:
                        fl_oscillation = self.check_oscillation(loss_steps, i, k,
                            loss_best, k3=self.thr_decr)
                        fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                        fl_oscillation = torch.max(fl_oscillation,
                            fl_reduce_no_impr)
                        reduced_last_check = fl_oscillation.clone()
                        loss_best_last_check = loss_best.clone()

                        if fl_oscillation.sum() > 0:
                            ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                            step_size[ind_fl_osc] /= 2.0
                            n_reduced = fl_oscillation.sum()

                            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                        k = max(k - self.size_decr, self.n_iter_min)
                    
                    elif self.norm == 'L1':
                        sp_curr = L0_norm(x_best - x)
                        fl_redtopk = (sp_curr / sp_old) < .95
                        topk = sp_curr / n_fts / 1.5
                        step_size[fl_redtopk] = alpha * self.eps
                        step_size[~fl_redtopk] /= adasp_redstep
                        step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                        sp_old = sp_curr.clone()
                    
                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                    
                    counter3 = 0
                    #k = max(k - self.size_decr, self.n_iter_min)
            print("***** Iter [%03d] | X1 norm: %.06f (Fea)" % (
                i, y1.sum())
            )
        print("APGD Maring finds the best loss at iter [{}]".format(best_iter_log))
        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr', 'Margin'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)
    

class APGDAttackCE():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
    
    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        
        
        
        
        
        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    
    def attack_single_run(self, x, y, x_init=None):
        best_loss_log, best_iter_log = -float("inf"), 0

        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
            
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):    
            # # ===== Check x_adv lies inside the ball Constraint =====
            # print(" ==== APGD Lp Constraint Sanity Check ====")
            # num_images = x.shape[0]
            # x_test = x_adv.clone().reshape(num_images, -1)
            # # x_test = x_best_adv.clone().reshape(num_images, -1)
            # x_orig = x.clone().reshape(num_images, -1)
            # if self.norm == "L1":
            #     norm_check = torch.sum(torch.abs(x_test-x_orig), axis=1)
            # elif self.norm == "L2":
            #     norm_check = torch.sqrt(torch.sum((x_test-x_orig)**2, dim=1))
            # elif self.norm == "Linf":
            #     norm_check = torch.amax(torch.abs(x_test-x_orig), dim=1)
            # else:
            #     raise RuntimeError("Wrong Norm specified")
            # print(" >> Iter [%d]" % (i))
            # print("Norm: ", norm_check.item())
            # # ==== Sanity Check Code ====


            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                        -1, *[1]*(len(x.shape) - 1)) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.

            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            if loss_best.sum() > best_loss_log:
                best_loss_log = loss_best.sum()
                best_iter_log = (i+1)
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)
            
            print("***** Iter [%03d] | X1 norm: %.06f (Fea)" % (
                i, y1.sum())
            )
        print("APGD finds the smallest loss at iter [{}]".format(best_iter_log))
        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)


