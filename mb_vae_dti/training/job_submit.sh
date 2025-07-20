total_batches=$1

for i in $(seq 0 $((total_batches-1))); do
    echo "Submitting job $i out of $total_batches"
    qsub -v "batch_index=$i,total_batches=$total_batches" job_grid.pbs
done