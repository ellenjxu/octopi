# start a GPU instance with a T4 GPU on an existing disk


# start a CPU instance (n1) on existing disk, put the local ssh key to the instance under authorized_keys
gcloud compute instances create mt-test \
    --project=octopi-416908 \
    --zone=us-west1-a \
    --machine-type=n1-standard-1 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=928063271722-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server,lb-health-check \
    --disk=boot=yes,device-name=malaria-training,mode=rw,name=malaria-training \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --metadata-from-file=ssh-keys=/home/heguanglin/.ssh/google_compute_engine.pub

# ssh to the vm (https://cloud.google.com/compute/docs/connect/standard-ssh#openssh-client)
gcloud compute ssh mt-test --zone=us-west1-a --project=octopi-416908
# or in other ssh client
ssh -i PATH_TO_PRIVATE_KEY USERNAME@EXTERNAL_IP

# delete an instance (be careful that thise will delete the instance and the attached disk)
gcloud compute instances delete instance-20240603-160554 --zone=us-west1-a --project=octopi-416908
