# Loop through numbers 01-20
for i in $(seq -f "%02g" 1 20); do
    # Loop through suffixes A, B, C
    for s in A B C; do
        # Create directory name
        DIR="SBC_20240725_${i}${s}"
        
        echo "Processing ${DIR}..."
        
        # Create directory if it doesn't exist
        mkdir -p "/mnt/disks/whole/new_sbcs/${DIR}"
        
        # Copy filtered spots and scores files from Google Cloud Storage
        gsutil -m cp \
            "gs://sbc07252024-reprocess/Reprocessed_SBC20240725/${DIR}/*_cropped.npy" \
            "/mnt/disks/whole/new_sbcs/${DIR}/"
    done
done