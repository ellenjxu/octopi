sudo fsck.ext4 -n /dev/sdc
# sudo mkdir -p /mnt/disks/whole
sudo mount /dev/sdc /mnt/disks/images
sudo df -h
ls -lh /mnt/disks/images