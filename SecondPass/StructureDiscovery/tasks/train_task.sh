# usage: bash tasks/train_task.sh TASK   (run from SecondPass/StructureDiscovery)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
t=$1
python train_arms.py raw_flat,cell_raw,cell_canon,cell_raw_gate,cell_canon_gate,cell_gcanon_gate,cell_canon_oracle 200,1000,3000 main tasks/$t orient > tasks/$t/log_train_orient.txt 2>&1
python train_arms.py raw_flat,cell_canon_gate,cell_gcanon_gate 200,1000,3000 main tasks/$t flow > tasks/$t/log_train_flow.txt 2>&1
