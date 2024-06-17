WANDB_NAME="retrain_v2"
CONFIG="tinynn_v1"

python3 train.py --config-name=$CONFIG "wandb.name=$WANDB_NAME" 
python3 test.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME"
python3 eval.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" 
