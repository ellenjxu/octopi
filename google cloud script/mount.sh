# mount the disk 
sudo lsblk
sudo fsck.ext4 /dev/sdb
sudo mkdir -p /mnt/disks/whole
sudo mount /dev/sdb /mnt/disks/whole
sudo df -h
ls -lh /mnt/disks/whole

# unmount
# sudo umount /mnt/disks/train