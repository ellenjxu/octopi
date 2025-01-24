# download cropped images from google bucket
# Loop through numbers 01-20
for i in $(seq -f "%02g" 1 20); do
    # Loop through sections A, B, C
    for s in A B C; do
        # Create directory name
        DIR="SBC_20240725_${i}${s}"
        
        echo "Processing ${DIR}..."
        
        # Create directory if it doesn't exist
        mkdir -p "downloaded_data/${DIR}"
        
        # Download filtered spots and scores files from Google Cloud Storage
        gsutil -m cp \
            "gs://sbc07252024-reprocess/Reprocessed_SBC20240725/${DIR}/*_cropped.npy" \
            "gs://sbc07252024-reprocess/Reprocessed_SBC20240725/${DIR}/*_scores.npy" \
            "downloaded_data/${DIR}/"
    done
done