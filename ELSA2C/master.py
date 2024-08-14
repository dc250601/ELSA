import wandb
import sys
import os

def exclude_bias_and_norm(p): #Do ask me why but was too desperate !!!
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = '9910u9cw'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(key="372c22d52e2518af87c606a4f27fa4c010355a9c")
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = "dc250601", project="ELSA2C")
    
    
