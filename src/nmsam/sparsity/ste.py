import torch


class MaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, mask):
        return master * mask

    @staticmethod
    def backward(ctx, upstream):
        return upstream, None


def effective_weight(master, mask, ste: bool = True):
    return MaskSTE.apply(master, mask) if ste else master * mask
