INSTANCE_NAME="test"
DISK_NAME="malaria-training"
ZONE="us-west1-a"
PROJECT="octopi-416908"

# start a new instance and append the existing disk to it
# here use cpu instance n1-standard-1, with debian-12 image
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=n1-standard-1 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=928063271722-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --tags=http-server,https-server,lb-health-check \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20240603-171351,image=projects/debian-cloud/global/images/debian-12-bookworm-v20240515,mode=rw,size=10,type=projects/$PROJECT/zones/us-west1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \

# append the existing disk to the new instance
gcloud compute instances attach-disk $INSTANCE_NAME --disk=$DISK_NAME --zone=$ZONE --project=$PROJECT

# mount the disk (this is for the first time writing to the disk)
sudo mkfs.ext4 -F /dev/disk/by-id/google-persistent-disk-1 
sudo mkdir -p /mnt/disks/train
sudo mount -o discard,defaults /dev/disk/by-id/google-persistent-disk-1 /mnt/disks/train
sudo chmod a+w /mnt/disks/train

# copy and past training data
gsutil cp gs://$DISK_NAME/init-train/negative.npy "negative.npy"
gsutil cp gs://$DISK_NAME/init-train/parasite_cleaned.npy "parasite_cleaned.npy"

# detach the disk from the old instance
gcloud compute instances detach-disk $INSTANCE_NAME --disk $DISK_NAME --zone $ZONE

# delete the old instance
gcloud compute instances delete $INSTANCE_NAME --zone $ZONE --project=$PROJECT