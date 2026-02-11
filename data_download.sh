#!/bin/bash

# Set NITRC credentials
export NITRC_USER="your_username"
export NITRC_PASS="your_password"

# Fetch OASIS-3 experiment metadata
curl -u "${NITRC_USER}:${NITRC_PASS}" \
  "https://nitrc.org/ir/data/archive/projects/OASIS3/experiments?format=csv" \
  -o oasis3_experiments_full.csv

# Extract experiment IDs (MRI sessions only)
tail -n +2 oasis3_experiments_full.csv | cut -d, -f6 | sort -u > oasis_experiments.csv
sed -i '1iexperiment_id' oasis_experiments.csv
grep '_MR_' oasis_experiments.csv > oasis_mri_experiments.csv
sed -i '1iexperiment_id' oasis_mri_experiments.csv

# Create VM for parallel download
gcloud compute instances create oasis3-ultrafast \
  --machine-type=c2-standard-8 \
  --boot-disk-size=3000GB \
  --zone=us-central1-a \

# Copy files to VM
gcloud compute scp \
  download_oasis_scans.sh oasis_mri_experiments.csv ${USER}@oasis3-ultrafast:~ \
  --zone=us-central1-a

# SSH to VM
gcloud compute ssh oasis3-ultrafast --zone=us-central1-a

sudo apt-get update
sudo apt-get install -y tmux unzip zip curl

# Create data directory
sudo mkdir -p /data/OASIS3
sudo chown $USER:$USER /data/OASIS3
cd ~

# Split CSV for parallel processing
split -n l/8 oasis_mri_experiments.csv subset_

# Launch parallel download sessions
for f in subset_*; do
  session=$(basename "$f")
  echo "Launching $session ..."
  tmux new -d -s "$session" "bash ./download_oasis_scans.sh $f /data/OASIS3 ${NITRC_USER} ALL | tee /data/OASIS3/${session}.log"
done

# Upload to Google Cloud Storage after downloads complete
gsutil -m rsync -r /data/OASIS3 gs://clinimcl-data/OASIS3/raw/
