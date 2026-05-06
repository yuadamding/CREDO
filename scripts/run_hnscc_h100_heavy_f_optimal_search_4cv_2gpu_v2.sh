#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck source=/dev/null
source scripts/_conda_init.sh
ensure_conda_available || {
  echo "conda is required but was not found on PATH." >&2
  exit 1
}
CONDA_BIN="$(resolve_conda_executable)" || {
  echo "A conda executable path could not be resolved." >&2
  exit 1
}

detect_gpu_devices() {
  local raw="${GPU_LIST:-}"
  if [[ -z "$raw" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    raw="${CUDA_VISIBLE_DEVICES}"
  fi
  if [[ -n "$raw" ]]; then
    local item=""
    IFS=',' read -r -a parsed <<< "$raw"
    for item in "${parsed[@]}"; do
      item="${item// /}"
      [[ -n "$item" ]] && printf '%s\n' "$item"
    done
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}'
    return 0
  fi
  return 1
}

is_disabled_optional_name() {
  local value="${1:-}"
  value="${value,,}"
  [[ -z "$value" || "$value" == "none" || "$value" == "null" || "$value" == "na" ]]
}

join_by_comma() {
  local joined=""
  local item=""
  for item in "$@"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$item"
  done
  printf '%s\n' "$joined"
}

visible_device_indices_csv() {
  local raw="$1"
  local item=""
  local idx=0
  local joined=""
  IFS=',' read -r -a visible_items <<< "$raw"
  for item in "${visible_items[@]}"; do
    item="${item// /}"
    [[ -z "$item" ]] && continue
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$idx"
    idx=$((idx + 1))
  done
  printf '%s\n' "$joined"
}

parse_search_folds() {
  SEARCH_FOLD_ITEMS=()
  local raw="${SEARCH_FOLDS:-0,1,2,3}"
  local item=""
  IFS=',' read -r -a parsed_folds <<< "$raw"
  for item in "${parsed_folds[@]}"; do
    item="${item// /}"
    [[ -z "$item" ]] && continue
    if ! [[ "$item" =~ ^[0-9]+$ ]]; then
      echo "Invalid SEARCH_FOLDS item: $item" >&2
      exit 1
    fi
    if [[ "$item" -ge "$CV_FOLDS" ]]; then
      echo "SEARCH_FOLDS item $item is outside CV_FOLDS=$CV_FOLDS." >&2
      exit 1
    fi
    SEARCH_FOLD_ITEMS+=("$item")
  done
  if [[ "${#SEARCH_FOLD_ITEMS[@]}" -eq 0 ]]; then
    echo "SEARCH_FOLDS resolved to no folds." >&2
    exit 1
  fi
}

load_settings() {
  if [[ -n "${SETTINGS_FILE:-}" ]]; then
    SETTINGS=()
    local line=""
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
      SETTINGS+=("$line")
    done < "$SETTINGS_FILE"
    return 0
  fi

  case "$SETTINGS_PRESET" in
    gpu_util_refine|util_refine)
      # Focused follow-up to the 20260504 utilization ladder. Fold 0 showed
      # that the accuracy/throughput sweet spot was not the largest 72-80 GB
      # particle-heavy model; it was the fatter active-batch h1344/p352/active40
      # branch. This preset searches around that boundary and probes active
      # 40-60 with lower particles to reduce the number of sequential
      # perturbation chunks per step.
      SETTINGS=(
        "heavy_f_ur01_h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|352|1408|2816|4096|40|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur02_h1344_d7_prog42_p336_s28_active44_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|336|1344|2688|4096|44|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur03_h1344_d7_prog42_p320_s28_active48_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|320|1280|2560|4096|48|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur04_h1344_d7_prog42_p288_s28_active60_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|288|1152|2304|4096|60|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur05_h1280_d7_prog40_p320_s28_active44_lc2e3_lw20_gr5e4_nogint_e1800|112|112|40|1280|7|320|1280|2560|4096|44|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur06_h1280_d7_prog40_p288_s28_active48_lc2e3_lw20_gr5e4_nogint_e1800|112|112|40|1280|7|288|1152|2304|4096|48|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur07_h1216_d7_prog38_p320_s28_active44_lc1e3_lw15_gr3e4_nogint_e1600|110|110|38|1216|7|320|1280|2560|4096|44|1e-3|0.15|3e-4|28|28|0|learned|on|1600"
        "heavy_f_ur08_h1152_d6_prog36_p288_s26_active44_lc1e3_lw15_gr3e4_nogint_e1500|108|108|36|1152|6|288|1152|2304|4096|44|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_ur09_h1088_d6_prog34_p288_s26_active40_lc1e3_lw15_gr3e4_nogint_e1500|106|106|34|1088|6|288|1152|2304|4096|40|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_ur10_h1088_d6_prog34_p256_s26_active48_lc1e3_lw15_gr3e4_nogint_e1500|106|106|34|1088|6|256|1024|2048|4096|48|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_ur11_h1024_d6_prog32_p288_s26_active40_lc1e3_lw15_gr3e4_nogint_e1400|104|104|32|1024|6|288|1152|2304|3072|40|1e-3|0.15|3e-4|26|26|0|learned|on|1400"
        "heavy_f_ur12_h1024_d6_prog32_p256_s26_active48_lc1e3_lw15_gr3e4_nogint_e1400|104|104|32|1024|6|256|1024|2048|3072|48|1e-3|0.15|3e-4|26|26|0|learned|on|1400"
        "heavy_f_ur13_h960_d6_prog30_p256_s24_active48_lc1e3_lw15_gr3e4_nogint_e1400|100|100|30|960|6|256|1024|2048|3072|48|1e-3|0.15|3e-4|24|24|0|learned|on|1400"
        "heavy_f_ur14_h896_d5_prog28_p256_s24_active50_lc1e3_lw12_gr2e4_nogint_e1200|96|96|28|896|5|256|1024|2048|3072|50|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_ur15_h1408_d7_prog44_p320_s28_active44_lc2e3_lw20_gr5e4_nogint_e1800|120|120|44|1408|7|320|1280|2560|4096|44|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_ur16_h1472_d7_prog46_p288_s28_active52_lc2e3_lw20_gr5e4_nogint_e1800|124|124|46|1472|7|288|1152|2304|4096|52|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
      )
      ;;
    gpu_util_ladder|util_ladder)
      # 20-setting utilization ladder. This keeps approximate 30-80 GB
      # single-H100 bands, but raises max_active_perturbations to reduce
      # chunk count and give each GPU fatter work per step. Particles are
      # reduced where active perturbations increase so the search remains
      # within the same rough memory envelope.
      SETTINGS=(
        "heavy_f_u30_h768_d5_prog24_p192_s24_active24_lc1e3_lw12_gr2e4_nogint_e1200|80|80|24|768|5|192|768|1536|2048|24|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_u32_h768_d5_prog24_p224_s24_active24_lc5e4_lw10_gr1e4_gint_e1200|80|80|24|768|5|224|896|1792|2048|24|5e-4|0.10|1e-4|24|24|1|learned|on|1200"
        "heavy_f_u35_h832_d5_prog26_p224_s24_active28_lc1e3_lw12_gr2e4_nogint_e1200|88|88|26|832|5|224|896|1792|3072|28|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_u38_h896_d5_prog28_p256_s24_active28_lc1e3_lw12_gr2e4_nogint_e1200|96|96|28|896|5|256|1024|2048|3072|28|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_u42_h960_d6_prog30_p256_s24_active32_lc1e3_lw15_gr3e4_nogint_e1400|100|100|30|960|6|256|1024|2048|3072|32|1e-3|0.15|3e-4|24|24|0|learned|on|1400"
        "heavy_f_u45_h1024_d6_prog32_p288_s26_active32_lc1e3_lw15_gr3e4_nogint_e1400|104|104|32|1024|6|288|1152|2304|3072|32|1e-3|0.15|3e-4|26|26|0|learned|on|1400"
        "heavy_f_u50_h1024_d6_prog32_p320_s24_active32_lc1e3_lw15_gr3e4_nogint_e1500|104|104|32|1024|6|320|1280|2560|3072|32|1e-3|0.15|3e-4|24|24|0|learned|on|1500"
        "heavy_f_u53_h1088_d6_prog34_p288_s26_active36_lc1e3_lw15_gr3e4_nogint_e1500|106|106|34|1088|6|288|1152|2304|4096|36|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_u56_h1152_d6_prog36_p320_s26_active36_lc1e3_lw15_gr3e4_nogint_e1500|108|108|36|1152|6|320|1280|2560|4096|36|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_u60_h1152_d7_prog36_p384_s28_active32_lc1e3_lw15_gr3e4_nogint_e1600|108|108|36|1152|7|384|1536|3072|4096|32|1e-3|0.15|3e-4|28|28|0|learned|on|1600"
        "heavy_f_u62_h1216_d7_prog38_p352_s28_active36_lc1e3_lw15_gr3e4_nogint_e1600|110|110|38|1216|7|352|1408|2816|4096|36|1e-3|0.15|3e-4|28|28|0|learned|on|1600"
        "heavy_f_u65_h1280_d7_prog40_p320_s28_active40_lc1e3_lw15_gr3e4_nogint_e1800|112|112|40|1280|7|320|1280|2560|4096|40|1e-3|0.15|3e-4|28|28|0|learned|on|1800"
        "heavy_f_u68_h1280_d7_prog40_p384_s28_active36_lc1e3_lw15_gr3e4_nogint_e1800|112|112|40|1280|7|384|1536|3072|4096|36|1e-3|0.15|3e-4|28|28|0|learned|on|1800"
        "heavy_f_u70_h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|352|1408|2816|4096|40|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_u72_h1408_d7_prog44_p384_s28_active36_lc2e3_lw20_gr5e4_nogint_e1800|120|120|44|1408|7|384|1536|3072|4096|36|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_u73_h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_nogint_e2000|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_u75_h1472_d7_prog46_p384_s28_active40_lc2e3_lw20_gr5e4_nogint_e2000|124|124|46|1472|7|384|1536|3072|4096|40|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_u77_h1408_d7_prog44_p416_s28_active40_lc2e3_lw20_gr5e4_nogint_e2000|120|120|44|1408|7|416|1664|3328|4096|40|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_u78_h1536_d7_prog48_p352_s28_active44_lc2e3_lw20_gr5e4_nogint_e2000|128|128|48|1536|7|352|1408|2816|4096|44|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_u79_h1280_d7_prog40_p448_s28_active32_lc1e3_lw15_gr3e4_nogint_e2000|112|112|40|1280|7|448|1792|3584|4096|32|1e-3|0.15|3e-4|28|28|0|learned|on|2000"
      )
      ;;
    vram_epoch_ladder|resource_ladder)
      # 20-setting resource ladder for first-fold search across approximate single-H100
      # VRAM bands and epoch budgets. Rows may include optional columns after
      # growth_reg:
      # steps|eval_steps|growth_intercept|program_basis|ecology|epochs
      # Names encode the intended VRAM band; measured peak GB is reported by
      # the runner/summarizer and should be used to prune the next round.
      SETTINGS=(
        "heavy_f_v30_h768_d5_prog24_p320_s24_active12_lc1e3_lw12_gr2e4_nogint_e1200|80|80|24|768|5|320|1280|2560|2048|12|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_v32_h768_d5_prog24_p384_s24_active12_lc5e4_lw10_gr1e4_gint_e1200|80|80|24|768|5|384|1536|3072|2048|12|5e-4|0.10|1e-4|24|24|1|learned|on|1200"
        "heavy_f_v35_h832_d5_prog26_p384_s24_active16_lc1e3_lw12_gr2e4_nogint_e1200|88|88|26|832|5|384|1536|3072|3072|16|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_v40_h896_d5_prog28_p384_s24_active16_lc1e3_lw12_gr2e4_nogint_e1200|96|96|28|896|5|384|1536|3072|3072|16|1e-3|0.12|2e-4|24|24|0|learned|on|1200"
        "heavy_f_v45_h960_d6_prog30_p416_s24_active16_lc1e3_lw15_gr3e4_nogint_e1400|100|100|30|960|6|416|1664|3328|3072|16|1e-3|0.15|3e-4|24|24|0|learned|on|1400"
        "heavy_f_v48_h1024_d6_prog32_p416_s26_active16_lc1e3_lw15_gr3e4_nogint_e1400|104|104|32|1024|6|416|1664|3328|3072|16|1e-3|0.15|3e-4|26|26|0|learned|on|1400"
        "heavy_f_v50_h1024_d6_prog32_p448_s24_active16_lc1e3_lw15_gr3e4_nogint_e1500|104|104|32|1024|6|448|1792|3584|3072|16|1e-3|0.15|3e-4|24|24|0|learned|on|1500"
        "heavy_f_v55_h1024_d6_prog32_p512_s24_active16_lc5e4_lw10_gr1e4_gint_e1500|104|104|32|1024|6|512|2048|4096|3072|16|5e-4|0.10|1e-4|24|24|1|learned|on|1500"
        "heavy_f_v56_h1088_d6_prog34_p480_s26_active20_lc1e3_lw15_gr3e4_nogint_e1500|106|106|34|1088|6|480|1920|3840|4096|20|1e-3|0.15|3e-4|26|26|0|learned|on|1500"
        "heavy_f_v60_h1152_d6_prog36_p512_s28_active20_lc1e3_lw15_gr3e4_nogint_e1600|108|108|36|1152|6|512|2048|4096|4096|20|1e-3|0.15|3e-4|28|28|0|learned|on|1600"
        "heavy_f_v62_h1152_d7_prog36_p480_s28_active24_lc1e3_lw15_gr3e4_nogint_e1600|108|108|36|1152|7|480|1920|3840|4096|24|1e-3|0.15|3e-4|28|28|0|learned|on|1600"
        "heavy_f_v66_h1216_d7_prog38_p512_s28_active24_lc1e3_lw15_gr3e4_nogint_e1800|110|110|38|1216|7|512|2048|4096|4096|24|1e-3|0.15|3e-4|28|28|0|learned|on|1800"
        "heavy_f_v68_h1280_d7_prog40_p512_s28_active24_lc1e3_lw15_gr3e4_nogint_e1800|112|112|40|1280|7|512|2048|4096|4096|24|1e-3|0.15|3e-4|28|28|0|learned|on|1800"
        "heavy_f_v70_h1280_d7_prog40_p544_s28_active20_lc1e3_lw15_gr3e4_nogint_e1800|112|112|40|1280|7|544|2176|4352|4096|20|1e-3|0.15|3e-4|28|28|0|learned|on|1800"
        "heavy_f_v72_h1344_d7_prog42_p512_s28_active24_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|1800"
        "heavy_f_v73_h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_nogint_e2000|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_v75_h1472_d7_prog46_p512_s28_active24_lc2e3_lw20_gr5e4_nogint_e2000|124|124|46|1472|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_v77_h1408_d7_prog44_p544_s28_active24_lc2e3_lw20_gr5e4_nogint_e2000|120|120|44|1408|7|544|2176|4352|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_v78_h1536_d7_prog48_p480_s28_active24_lc2e3_lw20_gr5e4_nogint_e2000|128|128|48|1536|7|480|1920|3840|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on|2000"
        "heavy_f_v79_h1280_d7_prog40_p576_s28_active24_lc1e3_lw15_gr3e4_nogint_e2000|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|3e-4|28|28|0|learned|on|2000"
      )
      ;;
    broad_search)
      # Broad first-fold search. Rows may include optional columns after
      # growth_reg:
      # steps|eval_steps|growth_intercept|program_basis|ecology|epochs
      # program_basis is learned or state_centroids; ecology is on or off.
      # Keep this preset explicit. The 20260504 fold-0 run showed the
      # h1344_p496_s32 branch can numerically collapse even though it fits
      # VRAM, so the default search stays on 28-step trajectories.
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on"
        "heavy_f_h1280_d7_prog40_p576_s28_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|3e-4|28|28|0|learned|on"
        "heavy_f_h1408_d6_prog44_p544_s28_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|6|544|2176|4352|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on"
        "heavy_f_h1280_d8_prog40_p512_s28_active24_lc2e3_lw20_gr5e4_nogint|112|112|40|1280|8|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|on"
        "heavy_f_h1152_d7_prog36_p640_s28_active16_lc1e3_lw15_gr3e4_nogint|104|104|36|1152|7|640|2560|5120|4096|16|1e-3|0.15|3e-4|28|28|0|learned|on"
        "heavy_f_h1152_d7_prog36_p544_s28_active32_lc1e3_lw15_gr3e4_nogint|104|104|36|1152|7|544|2176|4352|4096|32|1e-3|0.15|3e-4|28|28|0|learned|on"
        "heavy_f_h1408_d7_prog44_p512_s28_active24_lc2e3_lw15_gr3e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.15|3e-4|28|28|0|learned|on"
        "heavy_f_h1408_d7_prog44_p512_s28_active24_lc2e3_lw25_gr8e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.25|8e-4|28|28|0|learned|on"
        "heavy_f_h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_gint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|1|learned|on"
        "heavy_f_h1280_d7_statecent_p512_s28_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|512|2048|4096|4096|24|1e-3|0.15|3e-4|28|28|0|state_centroids|on"
        "heavy_f_h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_noeco|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4|28|28|0|learned|off"
      )
      ;;
    step32_vramfit)
      # Explicit high-risk probe: use longer 32-step trajectories, but
      # reduce particles so each job stays near the observed 72-80 GB
      # single-H100 band. The 20260504 h1344_p496_s32 branch is omitted
      # because it numerically collapsed after E~500.
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p480_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|480|1920|3840|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1280_d7_prog40_p512_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|512|2048|4096|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1472_d7_prog46_p464_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|124|124|46|1472|7|464|1856|3712|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1536_d7_prog48_p448_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|128|128|48|1536|7|448|1792|3584|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    stable_capacity_s28|refine_winner_s28)
      # Current default search around the 4-fold winner:
      # h1408_d7_prog44_p512_s28_active24_lc2e3_lw20_gr5e4_nogint.
      # 20260503/20260504 results showed: h1408/lc2 remains the best 4-fold
      # winner; h1280/lc1 is the closest accuracy competitor; nearby
      # regularization and 32-step branches can collapse. Keep the default
      # on stable 28-step candidates near the 72-80 GB H100 band.
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1440_d7_prog45_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|122|122|45|1440|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1472_d7_prog46_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|124|124|46|1472|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    particle_probe_s28)
      # Endpoint-biased probe. Fold 0 showed p544 improved UOT but hurt
      # dominant-state accuracy, so keep this separate from the default
      # accuracy-oriented search.
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1408_d7_prog44_p528_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|528|2112|4224|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1408_d7_prog44_p544_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|544|2176|4352|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    finalists_s28)
      # Fold-0 v2 screen winners. h1280/lc1 is the primary state-accuracy
      # candidate; h1408/lc2 is the balanced UOT/state-TV candidate; h1280/lc3
      # is the endpoint/mass candidate. Dropped dominated/unstable settings:
      # h1152/lc1, h1280/lc2, h1408/lc1.
      SETTINGS=(
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc3e3_lw25_gr1e3_nogint|112|112|40|1280|7|576|2304|4608|4096|24|3e-3|0.25|1e-3"
      )
      ;;
    acc_only_s28)
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    winner_s28)
      # Full 4-fold finalist winner from
      # hnscc_random_h100_heavy_f_optimal_search_v2_4cv_2gpu_20260502_192315:
      # mean test acc=0.4267, state TV=0.1932, UOT=32.9792, train peak=72.5 GB.
      SETTINGS=(
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    screen_s28)
      SETTINGS=(
        "heavy_f_h1152_d7_prog36_p576_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|104|104|36|1152|7|576|2304|4608|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|112|112|40|1280|7|576|2304|4608|4096|24|2e-3|0.20|5e-4"
        "heavy_f_h1280_d7_prog40_p576_s${N_STEPS}_active24_lc3e3_lw25_gr1e3_nogint|112|112|40|1280|7|576|2304|4608|4096|24|3e-3|0.25|1e-3"
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc1e3_lw15_gr3e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|1e-3|0.15|3e-4"
        "heavy_f_h1408_d7_prog44_p512_s${N_STEPS}_active24_lc2e3_lw20_gr5e4_nogint|120|120|44|1408|7|512|2048|4096|4096|24|2e-3|0.20|5e-4"
      )
      ;;
    *)
      echo "Unsupported SETTINGS_PRESET: $SETTINGS_PRESET" >&2
      echo "Use gpu_util_refine, util_refine, gpu_util_ladder, util_ladder, vram_epoch_ladder, resource_ladder, broad_search, step32_vramfit, stable_capacity_s28, refine_winner_s28, particle_probe_s28, winner_s28, finalists_s28, acc_only_s28, screen_s28, or provide SETTINGS_FILE." >&2
      exit 1
      ;;
  esac
}

ENV_NAME="${ENV_NAME:-cape-hnscc}"
DATA_PATH="${DATA_PATH:-../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad}"
CV_FOLDS="${CV_FOLDS:-4}"
SETTINGS_PRESET="${SETTINGS_PRESET:-gpu_util_ladder}"
if [[ -n "${SEARCH_N_STEPS:-}" ]]; then
  N_STEPS="$SEARCH_N_STEPS"
elif [[ -n "${N_STEPS:-}" ]]; then
  N_STEPS="$N_STEPS"
else
  case "$SETTINGS_PRESET" in
    step32_vramfit|*_s32)
      N_STEPS=32
      ;;
    *)
      N_STEPS=28
      ;;
  esac
fi
if [[ -n "${SEARCH_EVAL_STEPS:-}" ]]; then
  EVAL_STEPS="$SEARCH_EVAL_STEPS"
elif [[ -n "${EVAL_STEPS:-}" ]]; then
  EVAL_STEPS="$EVAL_STEPS"
else
  EVAL_STEPS="$N_STEPS"
fi
EPOCHS="${SEARCH_EPOCHS:-${EPOCHS:-2000}}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-150}"
STAGE_D_EPOCHS="${STAGE_D_EPOCHS:-150}"
STATE_KEY="${STATE_KEY:-Cell type annotation}"
RANDOM_STRATIFY_COLS="${RANDOM_STRATIFY_COLS:-Time point,perturbation_id}"
GUIDE_CONFIDENT_ONLY="${GUIDE_CONFIDENT_ONLY:-1}"
SHARED_GUIDE_EMBEDDING="${SHARED_GUIDE_EMBEDDING:-0}"
RUN_ROOT="${CV_ROOT:-${RUN_ROOT_PREFIX:-runs/hnscc_random_h100_heavy_f_optimal_search_v2_4cv_2gpu_$(date +%Y%m%d_%H%M%S)}}"
if [[ -z "${SUMMARY_RANKING_MODE:-}" ]]; then
  if is_disabled_optional_name "$STATE_KEY"; then
    SUMMARY_RANKING_MODE="balanced"
  else
    SUMMARY_RANKING_MODE="test_acc"
  fi
fi
MULTI_GPU_PER_JOB="${MULTI_GPU_PER_JOB:-0}"
PIN_CPU="${PIN_CPU:-1}"
NPROC_TOTAL="${NPROC_TOTAL:-$(nproc)}"
EXPRESSION_WORKERS="${EXPRESSION_WORKERS:-8}"
EXPRESSION_CHUNK_SIZE="${EXPRESSION_CHUNK_SIZE:-2048}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
ALLOW_UNSAFE_NO_CHECKPOINTING="${ALLOW_UNSAFE_NO_CHECKPOINTING:-0}"
GPU_MONITOR="${GPU_MONITOR:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"

mapfile -t GPU_DEVICES < <(detect_gpu_devices)
if [[ "${#GPU_DEVICES[@]}" -eq 0 ]]; then
  echo "No GPU devices could be detected. Set GPU_LIST explicitly." >&2
  exit 1
fi

if [[ "$MULTI_GPU_PER_JOB" == "1" ]]; then
  MAX_PARALLEL_JOBS=1
  ACTIVE_GPU_DEVICES=("${GPU_DEVICES[@]}")
  SCHEDULER_GPU_RESOURCES=("$(join_by_comma "${ACTIVE_GPU_DEVICES[@]}")")
  THREADS_PER_JOB="${THREADS_PER_GPU:-$NPROC_TOTAL}"
else
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${#GPU_DEVICES[@]}}"
  if [[ "$MAX_PARALLEL_JOBS" -gt "${#GPU_DEVICES[@]}" ]]; then
    MAX_PARALLEL_JOBS="${#GPU_DEVICES[@]}"
  fi
  if [[ "$MAX_PARALLEL_JOBS" -lt 1 ]]; then
    MAX_PARALLEL_JOBS=1
  fi
  ACTIVE_GPU_DEVICES=("${GPU_DEVICES[@]:0:$MAX_PARALLEL_JOBS}")
  SCHEDULER_GPU_RESOURCES=("${ACTIVE_GPU_DEVICES[@]}")
  THREADS_PER_JOB="${THREADS_PER_GPU:-$(( NPROC_TOTAL / ${#ACTIVE_GPU_DEVICES[@]} ))}"
fi
if [[ "$THREADS_PER_JOB" -lt 1 ]]; then
  THREADS_PER_JOB=1
fi
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-70000}"
if [[ "${ALLOW_SMALL_GPU:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in "${ACTIVE_GPU_DEVICES[@]}"; do
    gpu_mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$gpu_mem_mb" ]] && [[ "$gpu_mem_mb" =~ ^[0-9]+$ ]] && [[ "$gpu_mem_mb" -lt "$MIN_GPU_MEM_MB" ]]; then
      echo "GPU $gpu has ${gpu_mem_mb} MB, below MIN_GPU_MEM_MB=$MIN_GPU_MEM_MB for this H100 search." >&2
      echo "Set ALLOW_SMALL_GPU=1 only for deliberate dry runs or non-H100 smoke tests." >&2
      exit 1
    fi
  done
fi

declare -A GPU_SLOT=()
for slot_idx in "${!ACTIVE_GPU_DEVICES[@]}"; do
  GPU_SLOT["${ACTIVE_GPU_DEVICES[$slot_idx]}"]="$slot_idx"
done

load_settings
parse_search_folds
mkdir -p "$RUN_ROOT"

echo "CREDO optimal search v2 root: $RUN_ROOT"
echo "CREDO optimal search v2 settings: ${#SETTINGS[@]}"
echo "CREDO optimal search v2 preset: ${SETTINGS_FILE:-$SETTINGS_PRESET}"
echo "CREDO optimal search v2 folds: ${SEARCH_FOLD_ITEMS[*]} / CV_FOLDS=$CV_FOLDS"
echo "CREDO optimal search v2 default epochs: $EPOCHS"
echo "CREDO optimal search v2 default train/eval steps: $N_STEPS/$EVAL_STEPS"
echo "CREDO optimal search v2 state key: $STATE_KEY"
echo "CREDO optimal search v2 guide-confident only: $GUIDE_CONFIDENT_ONLY"
echo "CREDO optimal search v2 shared guide embedding: $SHARED_GUIDE_EMBEDDING"
echo "CREDO optimal search v2 ranking: $SUMMARY_RANKING_MODE"
echo "CREDO optimal search v2 strategy: multi_gpu_per_job=$MULTI_GPU_PER_JOB pin_cpu=$PIN_CPU nproc_total=$NPROC_TOTAL"
echo "CREDO optimal search v2 queue: jobs=$MAX_PARALLEL_JOBS resources=${SCHEDULER_GPU_RESOURCES[*]} active_gpus=${ACTIVE_GPU_DEVICES[*]} threads_per_job=$THREADS_PER_JOB"
echo "CREDO optimal search v2 expression loading: workers=$EXPRESSION_WORKERS chunk_size=$EXPRESSION_CHUNK_SIZE"
echo "CREDO optimal search v2 activation checkpointing: requested=$ACTIVATION_CHECKPOINTING allow_unsafe_no_checkpointing=$ALLOW_UNSAFE_NO_CHECKPOINTING"
echo "CREDO optimal search v2 gpu monitor: enabled=$GPU_MONITOR interval=${GPU_MONITOR_INTERVAL}s"
echo "CREDO optimal search v2 setting table:"
printf '%s\n' "${SETTINGS[@]}" | awk -F'|' -v epochs="$EPOCHS" -v force_epochs="${SEARCH_EPOCHS:-}" -v steps="$N_STEPS" -v eval_steps="$EVAL_STEPS" '{
  setting_steps=($15 == "" ? steps : $15)
  setting_eval_steps=($16 == "" ? eval_steps : $16)
  growth_intercept=($17 == "" ? "0" : $17)
  basis=($18 == "" ? "learned" : $18)
  ecology=($19 == "" ? "on" : $19)
  setting_epochs=(force_epochs != "" ? epochs : ($20 == "" ? epochs : $20))
  printf "  %s hidden=%s depth=%s programs=%s basis=%s ecology=%s particles=%s steps=%s eval_steps=%s active=%s max_atoms=%s eval_particles=%s eval_target=%s lam_ctrl=%s lam_weak=%s growth_reg=%s growth_intercept=%s epochs=%s\n", $1, $5, $6, $4, basis, ecology, $7, setting_steps, setting_eval_steps, $11, $10, $8, $9, $12, $13, $14, growth_intercept, setting_epochs
}'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "CREDO optimal search v2 dry run complete."
  exit 0
fi

FREE_GPUS=("${SCHEDULER_GPU_RESOURCES[@]}")
RUNNING_PIDS=()
declare -A PID_LABEL=()
declare -A PID_LOG=()
declare -A PID_GPU=()
FAILED_JOBS=0

cleanup_running_jobs() {
  local status=$?
  if [[ "$status" -ne 0 ]] && [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; then
    kill "${RUNNING_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_running_jobs EXIT

remove_running_pid() {
  local remove_pid="$1"
  local kept=()
  local pid=""
  for pid in "${RUNNING_PIDS[@]}"; do
    [[ "$pid" != "$remove_pid" ]] && kept+=("$pid")
  done
  RUNNING_PIDS=("${kept[@]}")
}

wait_for_one_job() {
  local finished_pid=""
  local status=0
  set +e
  wait -n -p finished_pid "${RUNNING_PIDS[@]}"
  status=$?
  set -e
  if [[ -z "$finished_pid" ]]; then
    FAILED_JOBS=$((FAILED_JOBS + 1))
    return 0
  fi
  local label="${PID_LABEL[$finished_pid]:-unknown}"
  local log_path="${PID_LOG[$finished_pid]:-}"
  local gpu="${PID_GPU[$finished_pid]:-unknown}"
  remove_running_pid "$finished_pid"
  FREE_GPUS+=("$gpu")
  unset "PID_LABEL[$finished_pid]" "PID_LOG[$finished_pid]" "PID_GPU[$finished_pid]"
  if [[ "$status" -ne 0 ]]; then
    FAILED_JOBS=$((FAILED_JOBS + 1))
    echo "CREDO optimal search v2 job failed: $label gpu=$gpu log=$log_path" >&2
    [[ -n "$log_path" ]] && tail -n 100 "$log_path" >&2 || true
  else
    echo "Completed job: $label gpu=$gpu"
  fi
}

wait_for_free_gpu() {
  while [[ "${#FREE_GPUS[@]}" -eq 0 ]]; do
    wait_for_one_job
  done
}

launch_job() {
  local gpu="$1"
  local fold="$2"
  local setting="$3"
  local tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms max_active lambda_ctrl lambda_weak growth_reg setting_steps setting_eval_steps growth_intercept program_basis ecology_mode setting_epochs
  IFS='|' read -r tag embedding mediator programs hidden depth particles eval_particles eval_target max_atoms max_active lambda_ctrl lambda_weak growth_reg setting_steps setting_eval_steps growth_intercept program_basis ecology_mode setting_epochs <<< "$setting"
  setting_steps="${setting_steps:-$N_STEPS}"
  setting_eval_steps="${setting_eval_steps:-$EVAL_STEPS}"
  growth_intercept="${growth_intercept:-0}"
  program_basis="${program_basis:-learned}"
  ecology_mode="${ecology_mode:-on}"
  setting_epochs="${setting_epochs:-$EPOCHS}"
  if [[ -n "${SEARCH_EPOCHS:-}" ]]; then
    setting_epochs="$EPOCHS"
  fi
  local job_activation_checkpointing="$ACTIVATION_CHECKPOINTING"
  if [[ "${job_activation_checkpointing,,}" =~ ^(0|false|off|no)$ ]] \
    && [[ "$ALLOW_UNSAFE_NO_CHECKPOINTING" != "1" ]] \
    && [[ "$max_active" =~ ^[0-9]+$ ]] \
    && [[ "$max_active" -gt 36 ]]; then
    job_activation_checkpointing=1
  fi
  local out_dir="$RUN_ROOT/$tag/fold_$fold"
  local log_path="$RUN_ROOT/${tag}.fold_${fold}.launcher.log"
  local cuda_visible_devices="$gpu"
  local multi_gpu_devices_arg=""
  local slot="${GPU_SLOT[$gpu]:-0}"
  local core_range=""
  if [[ "$MULTI_GPU_PER_JOB" == "1" ]]; then
    multi_gpu_devices_arg="$(visible_device_indices_csv "$cuda_visible_devices")"
    slot=0
  fi
  if [[ "$PIN_CPU" == "1" ]]; then
    local start_core=$(( slot * THREADS_PER_JOB ))
    local end_core=$(( start_core + THREADS_PER_JOB - 1 ))
    if [[ "$start_core" -lt "$NPROC_TOTAL" ]]; then
      if [[ "$end_core" -ge "$NPROC_TOTAL" ]]; then
        end_core=$((NPROC_TOTAL - 1))
      fi
      core_range="${start_core}-${end_core}"
    fi
  fi
  if [[ -f "$out_dir/results_summary.json" ]]; then
    echo "Skipping completed job: $tag fold_$fold"
    FREE_GPUS+=("$gpu")
    return 0
  fi
  mkdir -p "$out_dir"
  echo "Launching job: $tag fold_$fold gpu=$gpu"
  (
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
    export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
    export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_JOB}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS_PER_JOB}"
    CMD=(
      "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/run_credo_hnscc_full.py
      --data-path "$DATA_PATH"
      --output-dir "$out_dir"
      --latent-source vae
      --split-strategy random_kfold
      --cv-folds "$CV_FOLDS"
      --cv-fold-index "$fold"
      --random-stratify-cols "$RANDOM_STRATIFY_COLS"
      --seed 0
      --precision bf16
      --state-key "$STATE_KEY"
      --mass-scope subset_only
      --control-mode soft_ref
      --training-schedule staged
      --stage-c-epochs "$STAGE_C_EPOCHS"
      --stage-d-epochs "$STAGE_D_EPOCHS"
      --lambda-control-ref "$lambda_ctrl"
      --control-ref-warmup-epochs 150
      --n-programs "$programs"
      --embedding-dim "$embedding"
      --mediator-dim "$mediator"
      --hidden-dim "$hidden"
      --depth "$depth"
      --epochs "$setting_epochs"
      --n-particles "$particles"
      --n-steps "$setting_steps"
      --eval-particles "$eval_particles"
      --eval-steps "$setting_eval_steps"
      --eval-target-particles "$eval_target"
      --max-train-target-atoms "$max_atoms"
      --n-test-functions 12
      --lambda-weak "$lambda_weak"
      --lambda-reg-growth-bias "$growth_reg"
      --max-active-perturbations "$max_active"
      --min-cells-p4 20
      --min-cells-p60 20
      --cpu-threads "$THREADS_PER_JOB"
      --cpu-interop-threads 2
      --no-auto-scale-budget
      --expression-gene-mask-col hv_gene
      --expression-top-genes 2000
      --vae-latent-dim 50
      --vae-hidden-dim 512
      --vae-depth 2
      --vae-dropout 0.1
      --vae-epochs 50
      --vae-batch-size 2048
      --vae-lr 1e-3
      --vae-weight-decay 1e-6
      --vae-kl-weight 1e-3
      --vae-kl-warmup-epochs 20
      --vae-val-frac 0.1
      --vae-early-stop-patience 15
      --vae-grad-clip 1.0
      --vae-target-sum 10000
      --vae-encode-batch-size 8192
      --expression-workers "$EXPRESSION_WORKERS"
      --expression-chunk-size "$EXPRESSION_CHUNK_SIZE"
      --vae-hvg-batch-col Library
      --vae-hvg-time-col "Time point"
      --vae-hvg-min-cells-per-batch 256
      --vae-preload-dense-max-gb 4.0
      --vae-amp-dtype bf16
      --no-vae-use-raw
      --vae-batch-aware-hvg
      --no-vae-allow-full-gene-scan
      --vae-reuse-artifact
      --vae-use-amp
    )
    case "${GUIDE_CONFIDENT_ONLY,,}" in
      0|false|off|no)
        CMD+=(--include-nonconfident)
        ;;
      *)
        CMD+=(--guide-confident-only)
        ;;
    esac
    case "${SHARED_GUIDE_EMBEDDING,,}" in
      1|true|on|yes)
        CMD+=(--shared-guide-embedding)
        ;;
      *)
        CMD+=(--distinct-guide-embedding)
        ;;
    esac
    case "${job_activation_checkpointing,,}" in
      0|false|off|no)
        CMD+=(--no-activation-checkpointing)
        ;;
      *)
        CMD+=(--activation-checkpointing)
        ;;
    esac
    case "${ecology_mode,,}" in
      0|false|off|no)
        CMD+=(--ecology-off)
        ;;
      *)
        CMD+=(--ecology-on)
        ;;
    esac
    case "${growth_intercept,,}" in
      1|true|on|yes)
        CMD+=(--growth-intercept-on)
        ;;
      *)
        CMD+=(--growth-intercept-off)
        ;;
    esac
    case "${program_basis,,}" in
      state|state_centroid|state_centroids|centroid|centroids)
        CMD+=(--use-state-centroids)
        ;;
      *)
        CMD+=(--learned-programs)
        ;;
    esac
    if [[ -n "$multi_gpu_devices_arg" ]]; then
      CMD+=(--multi-gpu-devices "$multi_gpu_devices_arg")
    fi
    echo "CREDO resource plan: mode=v2-direct gpu_resource=$gpu split=$fold fold=$fold threads_per_job=$THREADS_PER_JOB cpu_affinity=${core_range:-unpinned} multi_gpu_per_job=$MULTI_GPU_PER_JOB epochs=$setting_epochs steps=$setting_steps eval_steps=$setting_eval_steps basis=$program_basis ecology=$ecology_mode growth_intercept=$growth_intercept guide_confident_only=$GUIDE_CONFIDENT_ONLY shared_guide_embedding=$SHARED_GUIDE_EMBEDDING activation_checkpointing=$job_activation_checkpointing expression_workers=$EXPRESSION_WORKERS expression_chunk_size=$EXPRESSION_CHUNK_SIZE"
    if [[ "$job_activation_checkpointing" != "$ACTIVATION_CHECKPOINTING" ]]; then
      echo "CREDO warning: requested ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING but max_active=$max_active is high; using activation_checkpointing=$job_activation_checkpointing for this job. Set ALLOW_UNSAFE_NO_CHECKPOINTING=1 to override."
    fi
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    if [[ -n "$multi_gpu_devices_arg" ]]; then
      echo "CREDO multi-gpu devices=$multi_gpu_devices_arg"
    fi
    printf 'CREDO command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    monitor_pid=""
    if [[ "$GPU_MONITOR" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      monitor_log="$out_dir/gpu_monitor.csv"
      (
        echo "timestamp,index,utilization_gpu_pct,memory_used_mb,memory_total_mb,power_draw_w"
        while true; do
          nvidia-smi -i "$cuda_visible_devices" \
            --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits
          sleep "$GPU_MONITOR_INTERVAL"
        done
      ) > "$monitor_log" 2>/dev/null &
      monitor_pid="$!"
      echo "GPU monitor log: $monitor_log"
    fi
    set +e
    if [[ "$PIN_CPU" == "1" ]] && [[ -n "$core_range" ]] && command -v taskset >/dev/null 2>&1; then
      taskset -c "$core_range" "${CMD[@]}"
      cmd_status=$?
    else
      "${CMD[@]}"
      cmd_status=$?
    fi
    set -e
    if [[ -n "$monitor_pid" ]]; then
      kill "$monitor_pid" 2>/dev/null || true
      wait "$monitor_pid" 2>/dev/null || true
    fi
    exit "$cmd_status"
  ) > "$log_path" 2>&1 &
  local pid=$!
  RUNNING_PIDS+=("$pid")
  PID_LABEL["$pid"]="$tag fold_$fold"
  PID_LOG["$pid"]="$log_path"
  PID_GPU["$pid"]="$gpu"
}

for setting in "${SETTINGS[@]}"; do
  for fold in "${SEARCH_FOLD_ITEMS[@]}"; do
    wait_for_free_gpu
    if [[ "$FAILED_JOBS" -ne 0 ]]; then
      break 2
    fi
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    launch_job "$gpu" "$fold" "$setting"
  done
done

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_one_job
done
if [[ "$FAILED_JOBS" -ne 0 ]]; then
  echo "CREDO optimal search v2 failed jobs: $FAILED_JOBS" >&2
  exit 1
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python runners/summarize_hnscc_cv.py \
  --cv-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT" \
  --group-by setting \
  --ranking-mode "$SUMMARY_RANKING_MODE"

echo "CREDO optimal search v2 summary:"
sed -n '1,120p' "$RUN_ROOT/cv_summary.md"
echo "$RUN_ROOT"
