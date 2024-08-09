import wandb
import sys
import os

def exclude_bias_and_norm(p): #Do ask me why but was too desperate !!!
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = '66pz1fxp'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(key="d8c87e3cc39d151c366aba4bc35b92b6b024c3c2")
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = "dc250601", project="ELSA_base_noise")
    
    
