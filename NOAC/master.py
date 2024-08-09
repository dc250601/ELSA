import wandb
import sys
import os

def exclude_bias_and_norm(p): #Do ask me why but was too desperate !!!
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = 'tlkr5276'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(key="93654a1a20a2034b2648c373e951236dce7071e3")
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = "dc250601", project="ELSA_no_ac")
    
    
