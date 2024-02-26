import wandb
import sys
import os

def exclude_bias_and_norm(p): #Do ask me why but was too desperate !!!
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = 'uuicd1yj'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(key="f22c1c61304ec17828f5ee7a282755ee2c02d593")
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = "dc250601", project="ssald_hyper")
    
    
