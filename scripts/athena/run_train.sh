#!/bin/bash
set -uo pipefail

CYFRONET_HOST="athena"
REMOTE_BASE="scratch/pp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
SLURM_FILE="$SCRIPT_DIR/train.sbatch"
ACTIVE_JOB_FILE="$SCRIPT_DIR/.train_active_job"
LOCAL_RUNS="$SCRIPT_DIR/runs"

DETACHED=false
EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --detached) DETACHED=true; shift ;;
        --existing) EXISTING=true; shift ;;
        *)
            echo "Nieznany argument: $1"
            echo "Użycie: $0 [--detached] [--existing]"
            exit 1
            ;;
    esac
done

if [ "$DETACHED" = true ] && [ "$EXISTING" = true ]; then
    echo "--detached i --existing się wykluczają."
    exit 1
fi

JOB_ID=""

job_running() {
    STATE=$(ssh "$CYFRONET_HOST" "squeue -h -j $JOB_ID -o %T 2>/dev/null" | tr -d '[:space:]')
    [ "$STATE" == "RUNNING" ] || [ "$STATE" == "PENDING" ]
}

save_active_job() {
    echo "JOB_ID=$JOB_ID" > "$ACTIVE_JOB_FILE"
}

clear_active_job() {
    rm -f "$ACTIVE_JOB_FILE"
}

resolve_existing_job() {
    if [ -f "$ACTIVE_JOB_FILE" ]; then
        source "$ACTIVE_JOB_FILE"
        if [ -n "$JOB_ID" ] && job_running; then
            echo "Znaleziono zapisane zadanie $JOB_ID."
            return 0
        fi
    fi
    JOB_ID=$(ssh "$CYFRONET_HOST" "squeue -h -u \$USER -n ucad-bench -o %i | sort -n | tail -n 1")
    if [ -z "$JOB_ID" ]; then
        echo "Brak aktywnego zadania ucad-bench na $CYFRONET_HOST."
        exit 1
    fi
    echo "Podłączam się do zadania $JOB_ID."
}

sync_to_remote() {
    echo "Synchronizacja kodu..."
    rsync -az --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pt' --exclude 'runs' \
        "$REPO_DIR/src" "$REPO_DIR/examples" "$REPO_DIR/scripts" \
        "$CYFRONET_HOST:$REMOTE_BASE/code/pyCLAD/"
}

submit_job() {
    echo "Zgłaszanie zadania do SLURM..."
    JOB_ID=$(ssh "$CYFRONET_HOST" "cd $REMOTE_BASE && sbatch --parsable code/pyCLAD/scripts/athena/train.sbatch")
    echo "Job ID: $JOB_ID"
    save_active_job
}

stream_job_output() {
    local TAIL_FROM="-n +1"
    if [ "$EXISTING" = true ]; then
        TAIL_FROM="-n 20"
        echo "Ostatnie linie każdego tasku, potem na żywo..."
    fi
    ssh "$CYFRONET_HOST" "cd $REMOTE_BASE && \
        stdbuf -oL tail $TAIL_FROM -qF \$(for i in \$(seq 0 7); do echo slurm-${JOB_ID}_\$i.out; done) 2>/dev/null & \
        TAIL_PID=\$!; \
        while squeue -h -j $JOB_ID -o %T 2>/dev/null | grep -q .; do sleep 20; done; \
        sleep 3; kill \$TAIL_PID 2>/dev/null"
}

download_results() {
    mkdir -p "$LOCAL_RUNS/$JOB_ID"
    rsync -az "$CYFRONET_HOST:$REMOTE_BASE/runs/$JOB_ID/" "$LOCAL_RUNS/$JOB_ID/"
    rsync -az "$CYFRONET_HOST:$REMOTE_BASE/slurm-$JOB_ID.out" "$LOCAL_RUNS/$JOB_ID/" 2>/dev/null || true
    echo "Logi pobrane do $LOCAL_RUNS/$JOB_ID"
    echo "===== WYNIKI ====="
    grep -h "RESULT" "$LOCAL_RUNS/$JOB_ID"/*.log 2>/dev/null | sed 's/.* bench: //' | sort
}

finalize_job() {
    [ -z "$JOB_ID" ] && return
    echo "Anuluję zadanie $JOB_ID..."
    ssh "$CYFRONET_HOST" "scancel $JOB_ID"
    while job_running; do sleep 2; done
    clear_active_job
    download_results
}

cleanup() {
    echo ""
    echo "Wykryto przerwanie — kończę zadanie SLURM $JOB_ID..."
    finalize_job
    exit 1
}

detached_interrupt() {
    echo ""
    if [ -n "$JOB_ID" ] && job_running; then
        echo "Przerwano podgląd. Zadanie $JOB_ID nadal działa."
        save_active_job
        print_detached_info
    fi
    trap - INT TERM
    exit 0
}

finish_after_stream() {
    if job_running; then
        echo "Zadanie $JOB_ID wciąż działa — kończę przez anulowanie."
        finalize_job
    else
        echo "Trening zakończony, pobieram wyniki..."
        clear_active_job
        download_results
    fi
}

print_detached_info() {
    echo "Podgląd logu:   $0 --existing"
    echo "Anuluj:         ssh $CYFRONET_HOST scancel $JOB_ID"
}

if [ "$EXISTING" = true ]; then
    resolve_existing_job
    trap detached_interrupt INT TERM
    stream_job_output
    trap - INT TERM
    finish_after_stream
elif [ "$DETACHED" = true ]; then
    sync_to_remote
    submit_job
    print_detached_info
else
    trap cleanup INT TERM
    sync_to_remote
    submit_job
    echo "Oczekiwanie na alokację i logi..."
    stream_job_output
    trap - INT TERM
    finish_after_stream
fi
