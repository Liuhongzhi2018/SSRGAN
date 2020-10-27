import torch


def create_model(opt):
    from .SSRGAN import SSRGAN
    model = SSRGAN()
    model.initialize(opt)

    if len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
