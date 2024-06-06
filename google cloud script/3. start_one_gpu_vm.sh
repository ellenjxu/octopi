export INSTANCE_NAME="test" DISK_NAME="malaria-whole" ZONE="us-west1-a" PROJECT="octopi-416908" DEVICE_NAME="google-persistent-disk-1"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=928063271722-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --accelerator=count=1,type=nvidia-tesla-t4 \
    --tags=http-server,https-server,lb-health-check \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ml-images/global/images/c0-deeplearning-common-gpu-v20240128-debian-11-py310,mode=rw,size=100,type=projects/$PROJECT/zones/$ZONE/diskTypes/pd-balanced \
    --disk=boot=no,device-name=$DEVICE_NAME,mode=rw,name=$DISK_NAME \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
# updated needed, set the deletion rule of the boot disk to keep-disk

# ssh to the vm
# set up conda environment


