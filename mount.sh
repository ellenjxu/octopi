# mount the disk (this is for the first time writing to the disk)
sudo lsblk
sudo fsck.ext4 /dev/your_100_gb_disk_name
sudo mount /dev/your_100_gb_disk_name /mnt/your_mount_point
sudo df -h
ls -lh /mnt/your_mount_point

# unmount
# sync
# sudo umount /mnt/disks/train