import wandb
import ssald
import no_ac
import nn

def runner(config = None):
    print("Inside runnner")
    wandb.init(config = config)
    config = wandb.config

    if config.mode == "ssald":
        ssald.ssald(config)
    elif config.mode == "Nearest_Neighbour":
        nn.Nearest_Neighbour(config)
    elif config.mode == "no_active":
        no_ac.no_active(config)
    else:
        print("Illegal")