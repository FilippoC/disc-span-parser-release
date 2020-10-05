from torch.optim.optimizer import Optimizer
import torch
import transformers

class MetaOptimizer:
    def __init__(self, args, params, filter_freezed=True, steps_per_epoch=-1, n_epochs=-1):
        if filter_freezed:
            # remove freezed parts of the model (e.g. Bert)
            params = filter(lambda p: p.requires_grad, params)

        # create optimizer
        if args.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=args.optim_lr,
                momentum=args.optim_momentum,
                weight_decay=args.optim_weight_decay
            )
        elif args.optim == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=args.optim_lr,
                betas=(args.optim_adam_b1, args.optim_adam_b2),
                weight_decay=args.optim_weight_decay,
                eps=args.optim_adam_eps
            )
        else:
            raise RuntimeError("Unknown optimizer: %s" % args.optim)

        # create scheduler
        self.update_scheduler = None
        self.eval_scheduler = None
        self.reduce_on_plateau = False
        self.lr_scheduler = False
        self.epoch = 0
        if args.optim_lr_scheduler == "plateau":
            self.reduce_on_plateau = True
            self.best_so_far = -float("inf")
            self.best_so_far_epoch = 0
            self.patience = args.optim_lr_scheduler_patience
            self.plateau_factor = args.optim_lr_scheduler_decay
            """
            self.eval_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=args.optim_lr_scheduler_decay,
                patience=args.optim_lr_scheduler_patience
            )
            """
        elif args.optim_lr_scheduler == "linear":
            num_training_steps = steps_per_epoch * n_epochs
            if num_training_steps <= 0:
                raise RuntimeError("For the linear scheduler, you must provide the number of steps and number of epochs")
            self.update_scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                self.optimizer,
                args.optim_lr_warmup,
                num_training_steps
            )
            self.optim_lr_warmup = 0
        elif args.optim_lr_scheduler == "exponential":
            if args.optim_lr_warmup > 0:
                self.lr_scheduler = True
                self.lr_scheduler_gamma = args.optim_lr_scheduler_decay ** (1. / args.optim_lr_scheduler_step)
            else:
                self.update_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    args.optim_lr_scheduler_decay ** (1. / args.optim_lr_scheduler_step)
                )
        #elif args.optim_lr_scheduler == "" and args.optim_lr_warmup > 0:
        #    self.update_scheduler = transformers.optimization.get_constant_schedule_with_warmup(
        #        self.optimizer,
        #        args.optim_lr_warmup
        #    )
        elif args.optim_lr_scheduler != "":
            raise RuntimeError("Unknown scheduler: %s" % args.optim_lr_scheduler)

        self.optim_lr_warmup = 0
        self.n_updates = 0
        if args.optim_lr_warmup > 0:
            self.optim_lr_warmup = args.optim_lr_warmup
            self.warmup_coeff = args.optim_lr / args.optim_lr_warmup
            self._update_lr(self.warmup_coeff)

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument("--optim", type=str, default="adam")
        cmd.add_argument("--optim-weight-decay", type=float, default=0.0)
        cmd.add_argument("--optim-momentum", type=float, default=0.0)
        cmd.add_argument("--optim-adam-b1", type=float, default=0.9)
        cmd.add_argument("--optim-adam-b2", type=float, default=0.999)
        cmd.add_argument("--optim-adam-eps", type=float, default=1e-8)
        cmd.add_argument("--optim-lr", type=float, default=1e-3)
        cmd.add_argument("--optim-lr-scheduler", type=str, default="")
        cmd.add_argument("--optim-lr-scheduler-step", type=int, default=10)
        cmd.add_argument("--optim-lr-scheduler-decay", type=float, default=0.1)
        cmd.add_argument("--optim-lr-scheduler-patience", type=float, default=2)
        cmd.add_argument("--optim-lr-warmup", type=int, default=0)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def step(self):
        self.optimizer.step()
        if self.update_scheduler is not None:
            self.update_scheduler.step()

        self.n_updates += 1
        if self.n_updates <= self.optim_lr_warmup:
            if self.update_scheduler is not None:
                raise RuntimeError("Ok this cannot work!")
            self._update_lr(self.n_updates * self.warmup_coeff)

            if True == self.lr_scheduler and self.n_updates == self.optim_lr_warmup:
                self.update_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    self.lr_scheduler_gamma
                )


    def eval_step(self, score):
        self.epoch += 1
        if self.eval_scheduler is not None:
            self.eval_scheduler.step(score)
        elif self.reduce_on_plateau:
            if score < self.best_so_far:
                if self.n_updates > self.optim_lr_warmup and self.best_so_far_epoch + self.patience >= self.epoch:
                    self._update_lr(self.optimizer.param_groups[0]["lr"] * self.plateau_factor)
            else:
                self.best_so_far = score
                self.best_so_far_epoch = self.epoch


# this code comes from here: https://gitlab.com/mcoavoux/discoparset/blob/master/src/Asgd.py
class AverageOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.n_steps = 1

        self.mem = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                d = {'cumulative_moving_average': torch.zeros_like(p.data)}
                d['cumulative_moving_average'].copy_(p.data)
                self.mem.append(d)

    def step(self):
        it = iter(self.mem)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                d = next(it)
                if p.grad is None:
                    continue
                d['cumulative_moving_average'].mul_(self.n_steps / (self.n_steps + 1.))
                d['cumulative_moving_average'].add_(1. / self.n_steps, p.data)

        self.n_steps += 1

    def average(self):
        it = iter(self.mem)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                d = next(it)
                if 'cumulative_moving_average_saved' not in d:
                    d['cumulative_moving_average_saved'] = torch.zeros_like(p.data)
                d['cumulative_moving_average_saved'].copy_(p.data)
                p.data.copy_(d['cumulative_moving_average'])

    def cancel_average(self):
        it = iter(self.mem)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                d = next(it)
                p.data.copy_(d['cumulative_moving_average_saved'])

