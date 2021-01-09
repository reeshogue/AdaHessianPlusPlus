import math
import torch
from torch.optim import Optimizer

'''Adapted from https://github.com/amirgholami/adahessian/blob/master/image_classification/optim_adahessian.py for learning purposes.'''

def centralization(x):
    x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class AdaHessian(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-4,
                weight_decay=0, hessian_power=1, k_val=2, alpha=0.4):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, 
                        hessian_power=hessian_power, k_val=k_val, alpha=alpha)
        super(AdaHessian, self).__init__(params, defaults)
    def get_trace(self, params, grads):
        hutch_rand = [2 * torch.randint_like(p, high=2) - 1 for p in params]
        hutch_est = torch.autograd.grad(
            grads,
            params,
            grad_outputs=hutch_rand,
            only_inputs=True,
            retain_graph=True,
            create_graph=True,
        )

        hutch_trace = []
        for h in hutch_est:
            param_size = h.size()
            if len(param_size) <= 2:
                tmp_outp = h.abs()
            elif len(param_size) == 4:
                tmp_outp = torch.mean(h.abs(), dim=[2,3], keepdim=True)
            hutch_trace.append(tmp_outp)
        return hutch_trace
    def step(self):
        params = []
        groups = []
        grads = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        hutch_traces = self.get_trace(params, grads)

        for (p, group, grad, hutch_trace) in zip(params, groups, grads, hutch_traces):
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['ema'] = torch.ones_like(p.data)
                state['ema_sq'] = torch.ones_like(p.data)
                state['prev_grad'] = torch.zeros_like(p.data)
                state['prev_hess'] = torch.zeros_like(p.data)
                state['slow_buffer'] = p.data

            ema, ema_sq, prev_grad, prev_hess = state['ema'], state['ema_sq'], state['prev_grad'], state['prev_hess']

            diff = torch.abs(prev_hess - hutch_trace)
            dfc = torch.sigmoid(diff)
            state['prev_hess'] = hutch_trace.clone()
            hutch_trace = hutch_trace * dfc

            diff = torch.abs(prev_grad - grad)
            dfc = torch.sigmoid(diff)
            state['prev_grad'] = grad
            grad = grad * dfc

            lr = group['lr']
            beta1, beta2 = group['betas']

 
            ema.mul_(beta1).add_(grad.detach_(), alpha=1-beta1)
            ema_sq.mul_(beta2).addcmul_(hutch_trace, hutch_trace, value=(1-beta2))
            
            state['step'] += 1
            # buffered = group['buffer'][state['step'] % 10]
            # if state['step'] == buffered[0]:
            #     N_sma, step_size = buffered[1], buffered[2]
            # else:
            #     buffered[0] = state['step']
            #     beta2_t = beta2 ** state['step']
            #     N_sma_max = 2 / (1 - beta2) - 1
            #     N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            #     buffered[1] = N_sma
            #     if N_sma >= 5:
            #         step_size = (
            #                     group['lr']
            #                     * math.sqrt(
            #                     1 - beta2_t
            #                     * (N_sma - 4)
            #                     / (N_sma_max - 4)
            #                     * (N_sma - 2)
            #                     / N_sma
            #                     * N_sma_max
            #                     / (N_sma_max - 2)
            #                 )
            #                 / (1 - beta1 ** state['step'])
            #         )
            #     else:
            #         step_size = lr / (1 - beta1 ** state['step'])
            #     buffered[2] = step_size
            # if N_sma >= 5:
            #     denom = ema_sq.sqrt().add_(1e-8)
            #     p.data.addcdiv_(-ema, denom, value=step_size)
            # else:
            #     p.data.add_(-ema, alpha=step_size)

            


            
            bias_corr_one = 1 - beta1 ** state['step']
            bias_corr_two = 1 - beta2 ** state['step']
            
            k = group['hessian_power']
            denom = (
                (ema_sq.sqrt() ** k) /
                math.sqrt(bias_corr_two) **  k).add_(
                    group['eps']
                )

            hess = ema / bias_corr_one / denom + group['weight_decay'] * p.data

            p.data = p.data - \
                group['lr'] * (hess)
            
            if state['step'] % group['k_val'] == 0:
                slow_p = state['slow_buffer']
                slow_p.add_(p.data - slow_p, alpha=group['alpha'])
                p.data.copy_(slow_p)
