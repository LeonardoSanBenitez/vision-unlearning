import wandb


def wandb_log_image(tracker, phase_name, prompt, epoch, images):
    tracker.log(
        {
            phase_name: [
                wandb.Image(image, caption=f"{name}: {prompt}") for name, image in images.items()
            ]
        }
    )
