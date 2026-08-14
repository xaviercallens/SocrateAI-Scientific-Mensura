#!/usr/bin/env bash
# F3 diagnostic: run the Benettin sweep for the Kaplan-Yorke targets.
#
# Plain xargs parallelism rather than a Python process pool: this box is shared
# with other long jobs and the pool deadlocked on it. One output file per run,
# named by its parameters, so a partial sweep is still usable and a re-run only
# recomputes what is missing.
#
# Grid (per system): 8 initial conditions x 3 time steps x 2 reorth intervals.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b/scratchpad/f3_runs"
mkdir -p "$OUT"

ICS=("1.0 1.0 1.0" "-1.0 2.0 5.0" "5.0 -3.0 20.0" "0.5 0.5 0.5" \
     "-8.0 7.0 27.0" "2.0 -2.0 12.0" "10.0 10.0 30.0" "-4.0 -4.0 15.0")

JOBS="$OUT/jobs.txt"
: > "$JOBS"

emit () {  # sys dt t_total t_reorth t_transient ic_index
  local sys=$1 dt=$2 tt=$3 tr=$4 ttr=$5 i=$6
  local tag="${sys}_dt${dt}_tr${tr}_ic${i}"
  if [ ! -s "$OUT/$tag.txt" ]; then
    echo "$sys $dt $tt $tr $ttr ${ICS[$i]} 400 $OUT/$tag.txt" >> "$JOBS"
  fi
}

for i in 0 1 2 3 4 5 6 7; do
  for dt in 0.005 0.002 0.001; do
    for tr in 0.5 1.0; do
      emit lorenz "$dt" 100000 "$tr" 100 "$i"
    done
  done
  for dt in 0.01 0.005 0.002; do
    for tr in 0.5 1.0; do
      emit rossler "$dt" 200000 "$tr" 500 "$i"
    done
  done
done

echo "$(wc -l < "$JOBS") runs queued"
xargs -a "$JOBS" -P 6 -L 1 bash -c '"'"$HERE"'/f3_lyap" "$0" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" > "$9"'
echo "sweep complete: $(ls "$OUT"/*.txt | wc -l) result files"
