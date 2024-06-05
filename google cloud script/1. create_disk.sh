DISK_NAME="malaria-training"
ZONE="us-west1-a"
PROJECT="octopi-416908"

gcloud compute disks create $DISK_NAME \
    --project=$PROJECT \
    --type=pd-ssd \
    --description=The\ npy\ files\ for\ training\ the\ malaria\ models \
    --size=20GB \
    --resource-policies=projects/octopi-416908/regions/southamerica-west1/resourcePolicies/default-schedule-1 \
    --zone=$ZONE
