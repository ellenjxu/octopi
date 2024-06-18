CONFIG="resnet" 
WANDB_NAME="retrain_v5" 
MODEL_NAME="resnet18"

python3 train.py --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME" 
python3 test.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME"
python3 eval.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME"