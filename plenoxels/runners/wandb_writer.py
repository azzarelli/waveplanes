import wandb

def set_config():
    pass

def log_plane(plane, name):
    """
    Args:
        plane: grid of features for a given resolution size (1,B,C,W) where B is feature size and C and W are axis resolutions
    """
    assert plane.size(0) == 1, 'Plane incorrectly shaped for logging with WandB'
    assert len(plane.shape) == 4, 'Incorrectly sizes input'

    plane = plane.squeeze(0).mean(0)
    image = wandb.Image(plane)

    wandb.log({name:image})

class Writer:
    def __init__(self, config, name, proj) -> None:
        wandb.login()
        self.name = name
        self.config = config
        self.proj = proj

    def run(self, trainer):
        with wandb.init(project=self.proj, config=self.config, name=self.name):
            trainer.train()