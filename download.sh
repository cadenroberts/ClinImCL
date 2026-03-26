#!/bin/bash
# =============================================
# OASIS-3 automated parallel mri downloader script
# =============================================

unset module

startSession() {
    local _user="${1:?startSession requires username as \$1}"
    local _pass="${2:?startSession requires password as \$2}"
    local COOKIE_JAR=".cookies-$(date +%s)-$$.txt"
    if ! curl -f -s -u "${_user}:${_pass}" --cookie-jar "${COOKIE_JAR}" "https://www.nitrc.org/ir/data/JSESSION" > /dev/null; then
        rm -f "$COOKIE_JAR"
        return 1
    fi
    echo "${COOKIE_JAR}"
    return 0
}

download() {
    local _jar="${1}"
    local OUTPUT="${2}"
    local URL="${3}"
    curl -f -H 'Expect:' --keepalive-time 2 --cookie "${_jar}" -o "${OUTPUT}" "${URL}"
}

endSession() {
    local _jar="${1}"
    curl -i --cookie "${_jar}" -X DELETE "https://www.nitrc.org/ir/data/JSESSION"
    rm -f "${_jar}"
}

download_scans() {
    local INFILE="$1"  # headerless CSV of experiment IDs (one per line)
    local DIRNAME="$2"
    local USERNAME="$3"
    local PASSWORD="$4"
    local SCANTYPE="${5:-ALL}"
    local AV1451_PROJ_ID="$6"

    if [ ! -d "$DIRNAME" ]; then
        mkdir -p "$DIRNAME"
    fi

    if ! COOKIE_JAR=$(startSession "$USERNAME" "$PASSWORD"); then
        echo "Error starting session. Maybe a bad username/password?"
        exit 1
    fi
    trap 'endSession "$COOKIE_JAR"' INT TERM EXIT

    while IFS=, read -r EXPERIMENT_ID; do
        SUBJECT_ID=$(echo "$EXPERIMENT_ID" | cut -d_ -f1)

        if ! [ "$SCANTYPE" = "ALL" ]; then
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
                PROJECT_ID="${AV1451_PROJ_ID}"
            else
                PROJECT_ID=OASIS3_AV1451
            fi
        fi

        download_url="https://www.nitrc.org/ir/data/archive/projects/${PROJECT_ID}/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_ID}/scans/${SCANTYPE}/files?format=zip"
        echo "$download_url"
        download "$COOKIE_JAR" "$DIRNAME/$EXPERIMENT_ID.zip" "$download_url"

        if zip -Tq "$DIRNAME/$EXPERIMENT_ID.zip" > /dev/null; then
            if ! [ "$SCANTYPE" = "ALL" ]; then
                echo "Found a ${SCANTYPE} scan for ${EXPERIMENT_ID}."
            else
                echo "Downloaded all scans for ${EXPERIMENT_ID}."
            fi
            echo "Unzipping scan(s) and rearranging files."
            unzip -o "$DIRNAME/$EXPERIMENT_ID.zip" -d "$DIRNAME"

            for single_scan in "$DIRNAME/$EXPERIMENT_ID/scans/"*/ ; do
                if [ -d "${single_scan}" ]; then
                    scan_name_all=$(echo "$single_scan" | rev | cut -d/ -f2 | rev)
                    scan_name=$(echo "$scan_name_all" | cut -d- -f1)
                    mkdir -p "$DIRNAME/$EXPERIMENT_ID/$scan_name"
                    mv "$DIRNAME/$EXPERIMENT_ID/scans/$scan_name_all/resources/"*/files/* "$DIRNAME/$EXPERIMENT_ID/$scan_name/."
                    chmod -R u=rwX,g=rwX "$DIRNAME/$EXPERIMENT_ID/$scan_name"
                fi
            done
            rm -r "$DIRNAME/$EXPERIMENT_ID/scans"
            rm -f "$DIRNAME/$EXPERIMENT_ID.zip"
        else
            if ! [ "$SCANTYPE" = "ALL" ]; then
                echo "Did not find a ${SCANTYPE} scan for ${EXPERIMENT_ID}."
            else
                echo "Could not download all scans for ${EXPERIMENT_ID}."
            fi
            rm -f "$DIRNAME/$EXPERIMENT_ID.zip"
        fi
        echo "Done with ${EXPERIMENT_ID}."
    done < "$INFILE"
    trap - INT TERM EXIT
    endSession "$COOKIE_JAR"
}

# ── Top-level commands (only run when script is executed directly) ──
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then

    : "${NITRC_USER:?Set NITRC_USER before running}"
    : "${NITRC_PASS:?Set NITRC_PASS before running}"

    # Fetch all OASIS-3 experiment metadata
    curl -f -u "${NITRC_USER}:${NITRC_PASS}" \
        "https://www.nitrc.org/ir/data/archive/projects/OASIS3/experiments?format=csv" \
        -o oasis3_experiments_full.csv \
        || { echo "Failed to fetch experiment metadata"; exit 1; }

    # Extract experiment IDs (keep only MRI for our project)
    tail -n +2 oasis3_experiments_full.csv | cut -d, -f6 | sort -u > oasis_experiments.csv
    { echo 'experiment_id'; cat oasis_experiments.csv; } > _tmp.csv && mv _tmp.csv oasis_experiments.csv
    grep '_MR_' oasis_experiments.csv > oasis_mri_experiments.csv
    { echo 'experiment_id'; cat oasis_mri_experiments.csv; } > _tmp.csv && mv _tmp.csv oasis_mri_experiments.csv

    # Create a VM with big disk for download
    read -rp "Create GCP VM oasis3-ultrafast (c2-standard-8, 3TB disk)? [y/N] " _confirm
    if [[ "${_confirm}" != [yY] ]]; then echo "Aborted."; exit 0; fi
    gcloud compute instances create oasis3-ultrafast \
        --machine-type=c2-standard-8 --boot-disk-size=3000GB --zone=us-central1-a

    # Copy files into the VM
    gcloud compute scp download.sh oasis_mri_experiments.csv \
        "${USER}@oasis3-ultrafast:~" --zone=us-central1-a

    # ── Commands below must be run manually inside the VM after SSH ──
    # gcloud compute ssh oasis3-ultrafast --zone=us-central1-a
    #
    # sudo apt-get update
    # sudo apt-get install -y tmux unzip zip curl
    #
    # sudo mkdir -p /data/OASIS3
    # sudo chown $USER:$USER /data/OASIS3
    # cd ~
    #
    # tail -n +2 oasis_mri_experiments.csv | split -n l/8 - subset_
    #
    # for f in subset_*; do
    #   session=$(basename "$f")
    #   echo "Launching $session ..."
    #   tmux new -d -s "$session" \
    #     "source download.sh && download_scans $f /data/OASIS3 $NITRC_USER $NITRC_PASS ALL 2>&1 | tee /data/OASIS3/${session}.log"
    # done
    #
    # # When tmux sessions have finished downloads:
    # gsutil -m rsync -r /data/OASIS3 gs://clinimcl-data/OASIS3/raw/

fi
