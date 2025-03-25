import numpy as np


def tensorboard_log_image(tracker, phase_name, prompt, epoch, images):
    np_images = np.stack([np.asarray(img) for img in images.values()])
    tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
