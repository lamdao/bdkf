#!/bin/bash
PRO="$1"
VOL="$2"
DFV="$3"
OUT="$4"
./bdkf_$PRO -i "$VOL" -d "$DFV" -o "$OUT" -s 0.075 -k 0.05 -n 100
