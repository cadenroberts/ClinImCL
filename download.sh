#!/bin/bash
# =============================================
# OASIS-3 automated parallel mri downloader script
# =============================================

# Set NITRC credentials (used for curl + download script)
export NITRC_USER="your_username"
export NITRC_PASS="your_password"

unset module

# Authenticates credentials against NITRC and returns the cookie jar file name. USERNAME and
# PASSWORD must be set before calling this function.
startSession() {
    local COOKIE_JAR=.cookies-$(date +%Y%M%d%s).txt
    if ! curl -f -k -s -u ${USERNAME}:${PASSWORD} --cookie-jar ${COOKIE_JAR} "https://www.nitrc.org/ir/data/JSESSION" > /dev/null; then
        return 1
    fi
    echo ${COOKIE_JAR}
    return 0
}

escape_chars_for_URL() {
    local input=${1}
    output=`echo "${input}" | sed -e 's/%/%25/g;' | sed -e 's/ /%20/g; s/</%3C/g; s/>/%3E/g; s/#/%23/g; s/+/%2B/g; s/{/%7B/g; s/}/%7D/g; s/|/%7C/g; s/\\\/%5C/g; s/\^/%5E/g; s/~/%7E/g; s/\[/%5B/g; s/\]/%5D/g; s/\`/%60/g; s/;/%3B/g; s/\//%2F/g; s/?/%3F/g; s/:/%3A/g; s/@/%40/g; s/=/%3D/g; s/&/%26/g; s/\\$/%24/g'`
    echo ${output}
}

download() {
    local OUTPUT=${1}
    local URL=${2}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie ${COOKIE_JAR} -o ${OUTPUT} ${URL}
}

continueDownload() {
    local OUTPUT=${1}
    local URL=${2}
    curl -H 'Expect:' --keepalive-time 2 -k --continue - --cookie ${COOKIE_JAR} -o ${OUTPUT} ${URL}
}

# Gets a resource from a URL.
get() {
    local URL=${1}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie ${COOKIE_JAR} ${URL}
}

# Ends the user session.
endSession() {
    curl -i -k --cookie ${COOKIE_JAR} -X DELETE "https://www.nitrc.org/ir/data/JSESSION"
    rm -f ${COOKIE_JAR}
}

download_scans() {
    local INFILE=$1
    local DIRNAME=$2
    local USERNAME=$3
    local SCANTYPE=${4:-ALL}
    local AV1451_PROJ_ID=$5

    if [ ! -d $DIRNAME ]; then
        mkdir -p $DIRNAME
    fi

    USERNAME=`escape_chars_for_URL "${USERNAME}"`
    PASSWORD=`escape_chars_for_URL "${PASSWORD}"`
    
    if ! COOKIE_JAR=$(startSession); then
        echo "Error starting session. Maybe a bad username/password?"
        exit 1
    fi

    sed 1d $INFILE | while IFS=, read -r EXPERIMENT_ID; do
        SUBJECT_ID=`echo $EXPERIMENT_ID | cut -d_ -f1`
        
        if ! [ $SCANTYPE = "ALL" ]; then
            echo "Checking for a ${SCANTYPE} scan for ${EXPERIMENT_ID}."
        else
            echo "Downloading all scans for ${EXPERIMENT_ID}."
        fi

        PROJECT_ID=OASIS3
        if [[ "${EXPERIMENT_ID}" == "OAS4"* ]]; then
            PROJECT_ID=OASIS4
        fi
        if [[ "${EXPERIMENT_ID}" == "OAS3"*"_AV1451"* ]]; then
            if [[ "${AV1451_PROJ_ID}" == "OASIS3_AV1451" ]] || [[ "${AV1451_PROJ_ID}" == "OASIS3_AV1451L" ]]; then
                echo "Tau project ID ${AV1451_PROJ_ID} was specified. Downloading from ${AV1451_PROJ_ID}."
                PROJECT_ID=${AV1451_PROJ_ID}
            else
                PROJECT_ID=OASIS3_AV1451
            fi
        fi

        download_url=https://www.nitrc.org/ir/data/archive/projects/${PROJECT_ID}/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_ID}/scans/${SCANTYPE}/files?format=zip
        echo $download_url
        download $DIRNAME/$EXPERIMENT_ID.zip $download_url

        if zip -Tq $DIRNAME/$EXPERIMENT_ID.zip > /dev/null; then
            if ! [ $SCANTYPE = "ALL" ]; then
                echo "Found a ${SCANTYPE} scan for ${EXPERIMENT_ID}."
            else
                echo "Downloaded all scans for ${EXPERIMENT_ID}."
            fi
            echo "Unzipping scan(s) and rearranging files."
            unzip $DIRNAME/$EXPERIMENT_ID.zip -d $DIRNAME
            
            for single_scan in $DIRNAME/$EXPERIMENT_ID/scans/*/ ; do
                if [ -d ${single_scan} ]; then
                    scan_name_all=`echo $single_scan | rev | cut -d/ -f2 | rev`
                    scan_name=`echo $scan_name_all | cut -d- -f1`
                    mkdir $DIRNAME/$EXPERIMENT_ID/$scan_name
                    mv $DIRNAME/$EXPERIMENT_ID/scans/$scan_name_all/resources/*/files/* $DIRNAME/$EXPERIMENT_ID/$scan_name/.
                    chmod -R u=rwX,g=rwX $DIRNAME/$EXPERIMENT_ID/$scan_name/*
                fi
            done
            rm -r $DIRNAME/$EXPERIMENT_ID/scans
        else
            if ! [ $SCANTYPE = "ALL" ]; then
                echo "Did not find a ${SCANTYPE} scan for ${EXPERIMENT_ID}."
            else
                echo "Could not download all scans for ${EXPERIMENT_ID}."
            fi
        fi
        rm $DIRNAME/$EXPERIMENT_ID.zip
        echo "Done with ${EXPERIMENT_ID}."
    done < $INFILE
    endSession
}

# Fetch all OASIS-3 experiment metadata
curl -u "${NITRC_USER}:${NITRC_PASS}" "https://nitrc.org/ir/data/archive/projects/OASIS3/experiments?format=csv" -o oasis3_experiments_full.csv

# Extract experiment IDs (keep only MRI for our project)
tail -n +2 oasis3_experiments_full.csv | cut -d, -f6 | sort -u > oasis_experiments.csv
sed -i '1iexperiment_id' oasis_experiments.csv
grep '_MR_' oasis_experiments.csv > oasis_mri_experiments.csv
sed -i '1iexperiment_id' oasis_mri_experiments.csv

# Create a VM with big disk for download
gcloud compute instances create oasis3-ultrafast --machine-type=c2-standard-8 --boot-disk-size=3000GB --zone=us-central1-a

# Copy files into the VM
gcloud compute scp download.sh oasis_mri_experiments.csv ${USER}@oasis3-ultrafast:~ --zone=us-central1-a

# SSH into the VM
gcloud compute ssh oasis3-ultrafast --zone=us-central1-a

sudo apt-get update
sudo apt-get install -y tmux unzip zip curl

# create workspace
sudo mkdir -p /data/OASIS3
sudo chown $USER:$USER /data/OASIS3
cd ~

# split CSV into 8 parts
split -n l/8 oasis_mri_experiments.csv subset_

# launch parallel downloads
for f in subset_*; do
  session=$(basename "$f")
  echo "Launching $session ..."
  tmux new -d -s "$session" "/bin/bash -c 'source download.sh && download_scans $f /data/OASIS3 ${NITRC_USER} ALL' | tee /data/OASIS3/${session}.log"
done

# When tmux sessions have finished downloads
gsutil -m rsync -r /data/OASIS3 gs://clinimcl-data/OASIS3/raw/