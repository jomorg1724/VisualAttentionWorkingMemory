cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for t in orientation_cued spatial_binding memory_guided_search; do
  ( python features.py $t 3000 41 train tasks/$t/data_train.npz && python features.py $t 1500 42 val tasks/$t/data_val.npz ) > tasks/$t/log_render.txt 2>&1 &
done
wait
