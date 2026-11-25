#!/bin/bash
# T13 v4 section 6: copy one finished run back from the cloud and verify sha256 file by file. Read-only on the cloud.
set -u
RUN=$1
cd /home/kevin/duet-system1
read -r -a SSH < <(python3 -c "import json;print(' '.join(json.load(open('logs_system1/t13v4/cloud/hosts.json'))['runs']['$RUN']['ssh']))")
D=$(python3 -c "import json;print(json.load(open('logs_system1/t13v4/cloud/hosts.json'))['runs']['$RUN']['dir'])")
HOST=${SSH[-1]}; PORT=${SSH[6]}
OUT=logs_system1/t13v4/runs/$RUN; LOG=logs_system1/t13v4/copyback/$RUN
mkdir -p $OUT $LOG
echo "start $(date -Is)" > $LOG/copyback.txt
rsync -a -e "ssh -i /home/kevin/.ssh/id_ed25519_autodl -o IdentitiesOnly=yes -o BatchMode=yes -p $PORT" "$HOST:$D/" "$OUT/" > $LOG/rsync.log 2>&1
echo "rsync rc $? $(date -Is)" >> $LOG/copyback.txt
"${SSH[@]}" -o BatchMode=yes "cd $D && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum" > $LOG/sha256_cloud.txt 2>>$LOG/rsync.log
(cd $OUT && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $LOG/sha256_local.txt
if cmp -s $LOG/sha256_cloud.txt $LOG/sha256_local.txt; then echo "sha256 match $(wc -l < $LOG/sha256_local.txt) files $(date -Is)" >> $LOG/copyback.txt; else echo "sha256 MISMATCH $(date -Is)" >> $LOG/copyback.txt; diff $LOG/sha256_cloud.txt $LOG/sha256_local.txt > $LOG/sha256_diff.txt; fi
echo COPYBACK_DONE
