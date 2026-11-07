#!/bin/bash
# RVIT_EXISTENCE_PROOF_PLAN.md rev 2 (2026-11-04): one existence-proof fine-tune, train -> assays; idempotent (requeue resumes).
#   run_ep.sh <arm> <seed> <start_high_loc>
# The run is named after its START model (rscaled10/loc<hl>_s<seed>); in *REV arms the table's high-value location is
# 3 - hl. Arms: W WREV WNULL (5k critic-only warm-ups, new 1201-particle head) | FREE FREEREV FROZ FROZREV FROZNULL
# (Part A) | REL RELK CONT RELREV RELKREV CONTREV (Part B, from FROZ / FROZREV final) | EDIT1 EDIT2 EDIT1H (Part D: the Part-C
# output-gate edit, k = 1 / 2 at the low-value location, k = 1 at the high-value one, written into W, then a 5k
# critic-only re-warm) and EDREL1 EDREL2 EDREL1H (standard learner from them).
# Every arm: q/k weight decay 0 (ep_train two-group Adam), theta frozen (--curr-threshold 1.01).
set -uo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}   # 3 runs share 8 CPUs (2026-11-04: CPU-bound at 60 ep/min)
ARM=$1; SEED=$2; HL=$3
BASE=/ix/jpherman/hermanj/rvit_zk; R=$BASE/runs; E=$R/existence
NAME=loc${HL}_s${SEED}; RUN=$E/$ARM/$NAME
cd $BASE/tree_zk; mkdir -p "$RUN"
TASK=luo2015_sensitivity
TH=$HL; SALL=1; K=0.25
case $ARM in *REV) TH=$((3 - HL)) ;; esac
case $ARM in WNULL|FROZNULL) SALL=10 ;; esac
case $ARM in RELK|RELKREV) K=0.125 ;; esac
EPS=${EP_EPISODES:-50000}; SNAP=(10000 25000); X=(--qk-decay 0); REF=""; REF2=""
case $ARM in
  W|WREV|WNULL) START=$R/whichloss/rscaled10/$NAME/ckpt_final.pt; X+=(--new-critic-head --critic-only); EPS=5000; SNAP=() ;;
  FREE)     START=$E/W/$NAME/ckpt_final.pt;     REF=$START ;;
  FREEREV)  START=$E/WREV/$NAME/ckpt_final.pt;  REF=$START ;;
  FROZ)     START=$E/W/$NAME/ckpt_final.pt;     REF=$START; X+=(--freeze-actor) ;;
  FROZREV)  START=$E/WREV/$NAME/ckpt_final.pt;  REF=$START; X+=(--freeze-actor) ;;
  FROZNULL) START=$E/WNULL/$NAME/ckpt_final.pt; REF=$START; X+=(--freeze-actor) ;;
  REL|RELK) START=$E/FROZ/$NAME/ckpt_final.pt;  REF=$E/W/$NAME/ckpt_final.pt; REF2=$START; SNAP=(5000 10000 25000) ;;
  CONT)     START=$E/FROZ/$NAME/ckpt_final.pt;  REF=$E/W/$NAME/ckpt_final.pt; REF2=$START; SNAP=(5000 10000 25000); X+=(--freeze-actor) ;;
  RELREV|RELKREV) START=$E/FROZREV/$NAME/ckpt_final.pt; REF=$E/WREV/$NAME/ckpt_final.pt; REF2=$START; SNAP=(5000 10000 25000) ;;
  CONTREV)  START=$E/FROZREV/$NAME/ckpt_final.pt; REF=$E/WREV/$NAME/ckpt_final.pt; REF2=$START; SNAP=(5000 10000 25000); X+=(--freeze-actor) ;;
  EDIT1|EDIT2|EDIT1H) START=$E/W/$NAME/ckpt_final.pt; REF=$START; EPS=5000; SNAP=(); X+=(--critic-only) ;;  # Part D: edit, critic re-warm
  EDREL1|EDREL2|EDREL1H) START=$E/EDIT${ARM#EDREL}/$NAME/ckpt_final.pt; REF=$E/W/$NAME/ckpt_final.pt; REF2=$START
            SNAP=(1000 5000 10000 25000) ;;
  # Part E (plan rev 4 §12): CD = critic detached from the trunk (--grad-route actor), AD = actor detached (critic);
  # from the Part-D EDIT checkpoints (edit at low: 1, at high: 1H) or from W (FREE control); ND = no trunk weight decay.
  EDRELCD1|EDRELAD1|EDRELND1|EDRELCD1H|EDRELAD1H|EDRELND1H|FREECD|FREEAD|FREEND)
            case $ARM in EDREL*1H) START=$E/EDIT1H/$NAME/ckpt_final.pt ;; EDREL*1) START=$E/EDIT1/$NAME/ckpt_final.pt ;;
                         FREE*) START=$E/W/$NAME/ckpt_final.pt ;; esac
            REF=$E/W/$NAME/ckpt_final.pt; [[ $ARM == EDREL* ]] && REF2=$START; SNAP=(1000 5000 10000 25000)
            case $ARM in *CD*) X+=(--grad-route actor --route-change-ok --trunk-decay 0) ;; *AD*) X+=(--grad-route critic --route-change-ok) ;;
                         *ND*) X+=(--trunk-decay 0) ;; esac ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
COMMON=(--task $TASK --seed "$SEED" --high-loc "$TH" --no-fixation --gamma 1.0 --value-ratio 25 --mean-reward 3.0
        --hit-cr-ratio 1.0 --low-hit-cr-ratio 0.44 --num-particles 1201 --lm-low-reward-scale "$K"
        --lm-reward-scale-all "$SALL" --curr-threshold 1.01 --qk-log)
echo "[ep] arm=$ARM run=$RUN start=$START table_high=$TH K=$K scale_all=$SALL eps=$EPS flags=${X[*]} host=$(hostname) $(date)"
[[ -f $START ]] || { echo "[ep] START MISSING $START"; exit 3; }
if [[ $ARM == EDIT* && ! -f $RUN/ckpt_final.pt ]]; then           # edit W, then re-warm the critic on the edited network
  KK=${ARM#EDIT}; AT=low; [[ $KK == *H ]] && { AT=high; KK=${KK%H}; }
  python -u -m zk.ep_make_edit --ckpt "$START" --hl "$TH" --k "$KK" --site O --at $AT --out "$RUN/edit_init.pt"     || { echo "[ep] EDIT FAILED"; exit 4; }
  python -u -m zk.ep_train "${X[@]}" --out-dir "$RUN" "${COMMON[@]}" --init-from "$RUN/edit_init.pt" --episodes "$EPS"     || { echo "[ep] REWARM FAILED"; exit 4; }
elif [[ ! -f $RUN/ckpt_final.pt ]]; then
  python -u -m zk.ep_train "${X[@]}" --out-dir "$RUN" "${COMMON[@]}" --init-from "$START" --episodes "$EPS" \
    ${SNAP[@]:+--snapshot-at "${SNAP[@]}"} || { echo "[ep] TRAIN FAILED"; exit 4; }
fi
assay() {  # ckpt_basename [extra lm_assay args]
  local ck=$1; shift
  [[ -f $RUN/$ck.pt && ! -f $RUN/roc_$ck.json ]] && python -u -m zk.lm_assay --family orig --checkpoint "$RUN/$ck.pt" \
    --condition-loc "$TH" --run-id "ep_${ARM}_${NAME}_$ck" --output "$RUN/roc_$ck.json" --no-fixation --task $TASK "$@" || true
}
for s in "${SNAP[@]}"; do assay ckpt_ep$s --thetas 18; done
assay ckpt_final
[[ -f $RUN/lmro_final.json ]] || python -u -m zk.lm_readout_probe --runs "$RUN" --out "$RUN/lmro_final.json" --n 600 --delta 18 || true
if [[ -n $REF ]]; then
  [[ -f $RUN/lm_perturb.json ]] || python -u -m zk.lm_perturb --family orig --checkpoint "$RUN/ckpt_final.pt" \
    --condition-loc "$TH" --output "$RUN/lm_perturb.json" --trials 1000 --no-fixation --task $TASK --thetas 18 || true
  python -u -m zk.ep_hybrid --mode cross --end "$RUN/ckpt_final.pt" --ref "$REF" --hl "$TH" --out "$RUN/hyb_W.json" || true
  python -u -m zk.ep_geom --end "$RUN/ckpt_final.pt" --ref "$REF" --hl "$TH" --out "$RUN/geom_W.json" || true
fi
if [[ -n $REF2 ]]; then
  python -u -m zk.ep_hybrid --mode cross --end "$RUN/ckpt_final.pt" --ref "$REF2" --hl "$TH" --out "$RUN/hyb_start.json" || true
  python -u -m zk.ep_geom --end "$RUN/ckpt_final.pt" --ref "$REF2" --hl "$TH" --out "$RUN/geom_start.json" || true
fi
case $ARM in FROZ|FROZREV|FROZNULL)
  python -u -m zk.ep_hybrid --mode transplant --end "$RUN/ckpt_final.pt" --ref "$REF" --hl "$TH" --out "$RUN/tpl_W.json" || true ;;
esac
[[ -f $RUN/roc_ckpt_final.json ]] || { echo "[ep] ASSAY FAILED"; exit 5; }
echo "[ep] DONE $RUN $(date)"
