# MengqingCao/ATOM#108: the hour's decode role with the decodes switched to C2 (#81/#73):
# DEP4 + MegaMoE a4w4, bf16 combine, DSpark draft at attention TP1, MNBT 512 as C2 ran.
# d_env.sh's transport, KV, admission, spec and capacity settings stay; the c17 decode switches
# are dropped (they target TP4 attention collectives; C2's decode ran none of them).
D_MNBT=${D_MNBT:-512}
source "$(dirname "${BASH_SOURCE[0]}")/d_env.sh"
_drop=" ATOM_M3_AK_DECODE ATOM_P1_INDEXER_SPLIT4 ATOM_P1_INDEXER_SPLIT4_WIDE ATOM_P1_INDEXER_SPLIT4_MIN8
  ATOM_P1_K2_SCORER ATOM_M3_AK_INSERT ATOM_M3_AK_DECODE_QOUT ATOM_P1_AK_WIDE ATOM_P1_L2_D2 ATOM_M3_FUSED_DECODE
  ATOM_AK_MXFP8_GEMM ATOM_P1_FRONT_QKV_CHAIN ATOM_P1_AR_QKV_CODES ATOM_P1_L4_PULLQ ATOM_MINIMAX_AK_AR_NORM
  ATOM_M3_AK_MOE ATOM_M3_DRAFT_AK_AR_NORM ATOM_M3_DRAFT_FUSED_CTX_ROPE ATOM_M3_DRAFT_MARKOV_WIDE
  ATOM_M3_DRAFT_POS_MAJOR ATOM_P1_DRAFT_MARKOV_PERSIST ATOM_P1_DRAFT_FC_AK ATOM_P1_FAST_CONCAT
  ATOM_P1_DSPARK_AUX_NEXT ATOM_M3_VERIFY_HEAD_CHUNK_TUNED "
# shellcheck disable=SC2086
unset ${_drop}
_keep=()
for _n in "${D_ENV_NAMES[@]}"; do [[ "${_drop}" == *" ${_n}"[[:space:]]* ]] || _keep+=("${_n}"); done
export ATOM_MEGA_A4W4=1
# MengqingCao/ATOM#109: ganyi's c17 DSpark decode features default on in this tree (verify head, draft context in the
# decode graph, fused draft metadata, sharded draft fc) and C2's tree had none of them. With them on, the first real
# decode step on one DP rank never completes beside its peers' dummy steps (decode GPUs spin, every rank stalls).
export ATOM_M3_VERIFY_HEAD=0 ATOM_M3_DRAFT_CTX_IN_GRAPH=0 ATOM_M3_DRAFT_FUSED_META=0 ATOM_M3_DRAFT_FC_SHARDED=0
D_ENV_NAMES=("${_keep[@]}" ATOM_MEGA_A4W4
  ATOM_M3_VERIFY_HEAD ATOM_M3_DRAFT_CTX_IN_GRAPH ATOM_M3_DRAFT_FUSED_META ATOM_M3_DRAFT_FC_SHARDED)
D_SERVER_ARGS+=(--enable-dp-attention --enable-expert-parallel --moe-backend mega)
unset _drop _keep _n
