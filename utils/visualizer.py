import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
# try:
#     from StringIO import StringIO  # Python 2.7
# except ImportError:
#     from io import BytesIO         # Python 3.x
from io import BytesIO
# from .SpectralUtils import savePNG, projectToRGB, toPNG
from .SpectralUtils import projectToRGB, toPNG
import numpy as np
# from PIL import Image
import cv2


class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.savepath = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
        self.val_savepath = os.path.join(opt.checkpoints_dir, opt.name, 'validation')
        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(self.val_savepath, exist_ok=True)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log:
            # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        # save images to a html file
        if self.use_html:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, iter, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iter, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    # save image and hsi sample
    def display_samples(self, rgb, hyper, generated, epoch, total_steps, i, mode):
        BIT_8 = 256
        filtersPath = "./cie_1964_w_gain.npz"

        # Load HS image and filters
        filters = np.load(filtersPath)['filters']

        # Project HSI to RGB
        hyper_numpy = hyper.cpu().float().numpy().squeeze(0)
        hyper_numpy = np.transpose(hyper_numpy, (1, 2, 0))
        hyper_numpy = (hyper_numpy - hyper_numpy.min())/(hyper_numpy.max()-hyper_numpy.min())
        hyper_RGB = np.true_divide(projectToRGB(hyper_numpy, filters), BIT_8)
        hyper_RGB = toPNG(hyper_RGB)

        # Project generated to RGB
        generated_numpy = generated.detach().cpu().float().numpy().squeeze(0)
        generated_numpy = np.transpose(generated_numpy, (1, 2, 0))
        generated_numpy = (generated_numpy - generated_numpy.min())/(generated_numpy.max()-generated_numpy.min())
        generated_RGB = np.true_divide(projectToRGB(generated_numpy, filters), BIT_8)
        # generated_RGB = (generated_RGB - generated_RGB.min())/(generated_RGB.max()-generated_RGB.min())
        generated_RGB = toPNG(generated_RGB)

        # Save image file
        image_numpy = rgb.cpu().float().numpy().squeeze(0)
        # print("out image_numpy max {} min {}".format(image_numpy.max(), image_numpy.min()))
        image_numpy = (image_numpy - image_numpy.min())/(image_numpy.max()-image_numpy.min())
        # print("norm image_numpy max {} min {}".format(image_numpy.max(), image_numpy.min()))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        # print("255 image_numpy max {} min {}".format(image_numpy.max(), image_numpy.min()))
        img = np.uint8(image_numpy)

        sample = self.joint_img_horizontal(img, hyper_RGB, generated_RGB)
        if mode:
            cv2.imwrite(os.path.join(self.savepath, "RGB2HSI_%d_%d.png" % (epoch, total_steps)), sample)
        else:
            cv2.imwrite(os.path.join(self.val_savepath, "val_RGB2HSI_%d_%d.png" % (epoch, i)), sample)

    def joint_img_horizontal(self, rgb, real_hyper, hyper):
        img = np.concatenate((rgb, real_hyper, hyper), axis=1)
        return img
