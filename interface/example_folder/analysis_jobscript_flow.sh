#!/bin/bash -x
#SBATCH --job-name=flow-analysis
#SBATCH --account=coldcluster
#SBATCH --partition=batch
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ========================
# User-editable parameters
# ========================
ANALYSIS_SCRIPT="/p/project1/coldcluster/lenard/energy-transfer-analysis/run_analysis.py"
SNAPSHOT_DIR=$SLURM_SUBMIT_DIR  # folder with existing snapshots
OUTFILE_PREFIX="flow_analysis_BB"
CHECK_INTERVAL=1               # in seconds
RESOLUTION=2048                 # simulation resolutio
EOS="isothermal"
BOX_LENGTH=6.283185

SNAPSHOT_INTERVAL=1            # Only process every Nth snapshot

# ========================
# Environment setup
# ========================

module purge; module load Stages/2025 GCC/13.3.0 OpenMPI CMake HDF5 Python FFTW Ninja ADIOS2; source /p/project1/coldcluster/lenard/venvs/batch/bin/activate

export NCCL_DEBUG=INFO
export KOKKOS_VERBOSE=1

# ========================
# Analysis loop
# ========================
LAST_PROCESSED_FILE=""

# ========================
# Analysis loop (tandem mode with graceful stop)
# ========================
export PYTHONUNBUFFERED=1  # forces Python to flush output immediately

LAST_PROCESSED_FILE=""

while true; do
    # Get all snapshot files (sorted oldest → newest)
    SNAPSHOT_FILES=($(ls -1tr $SNAPSHOT_DIR/parthenon.prim.*.phdf 2>/dev/null))

    if [[ ${#SNAPSHOT_FILES[@]} -eq 0 ]]; then
        echo "[Analysis] No snapshots found yet. Waiting ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
        continue
    fi

    for FILE in "${SNAPSHOT_FILES[@]}"; do
        # Extract numeric part and force base-10
        NUM=$(basename "$FILE" .phdf | awk -F. '{print $3}')
        NUM=$((10#$NUM))
        
        # Skip if the snapshot number is not a multiple of SNAPSHOT_INTERVAL
        if (( NUM % SNAPSHOT_INTERVAL != 0 )); then
            continue
        fi

        OUTFILE="$SNAPSHOT_DIR/${OUTFILE_PREFIX}_$(basename $FILE .phdf).hdf5"

        # Skip if already processed
        if [[ -f "$OUTFILE" ]]; then
            LAST_PROCESSED_FILE="$FILE"
            continue
        fi

        echo "[Analysis] Processing snapshot: $FILE"

        srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_NTASKS \
            python3 -u "$ANALYSIS_SCRIPT" \
            --res=$RESOLUTION \
            --type=flow \
            --data_type=AthenaPK \
            --data_path="$FILE" \
            --eos=$EOS \
	    --box_length=$BOX_LENGTH \
            --outfile="$OUTFILE" \
	    -b 

        echo "[Analysis] Completed $FILE"
        LAST_PROCESSED_FILE="$FILE"
    done

    # Check if the final snapshot exists and has been processed
    FINAL_SNAPSHOT="$SNAPSHOT_DIR/parthenon.prim.final.phdf"
    FINAL_OUTFILE="$SNAPSHOT_DIR/${OUTFILE_PREFIX}_parthenon.prim.final.pkl"

    if [[ -f "$FINAL_SNAPSHOT" && -f "$FINAL_OUTFILE" ]]; then
        echo "[Analysis] Final snapshot detected and processed — exiting."
        break
    fi

    echo "[Analysis] Waiting ${CHECK_INTERVAL}s for new snapshots..."
    sleep $CHECK_INTERVAL
done

echo "[Analysis] All snapshots processed. Exiting cleanly."
