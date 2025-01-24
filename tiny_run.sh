CONFIG="tinynn" 
WANDB_NAME="h7h28_v4" 

#python3 train.py --config-name=$CONFIG "wandb.name=$WANDB_NAME"  
python3 test.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" 
python3 eval.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" 