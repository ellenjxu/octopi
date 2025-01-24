CONFIG="resnet" 
WANDB_NAME="retrain_v2" 
MODEL_NAME="resnet34"

#python3 train.py --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME" "train.lr=0.0001" "train.epochs=20"
# python3 test.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME" "test.whole=False" "test.cp_name=best.pt"
python3 eval.py  --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME"
#python3 infer.py --config-name=$CONFIG "wandb.name=$WANDB_NAME" "model.model=$MODEL_NAME" "test.whole=True" +"npy_file=/mnt/disks/whole/init-train/pos/parasite_cleaned.npy"  