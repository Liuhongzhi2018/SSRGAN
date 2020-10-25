import torch


def create_model(opt):
    from .pix2pixHD_model import Pix2PixHDModel
    model = Pix2PixHDModel()
    model.initialize(opt)

    if len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
