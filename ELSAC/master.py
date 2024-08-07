import wandb
import sys
import os

def exclude_bias_and_norm(p): #Do ask me why but was too desperate !!!
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = 'osphvjo0'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(key="547a606eb7b917120783943b144a6c90b722dd25")
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = "dc250601", project="ELSAC")
    
    
