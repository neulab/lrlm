#!/usr/bin/env bash
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "Dry run"
else
    DRY_RUN=0
    # Credit: https://stackoverflow.com/questions/1885525/how-do-i-prompt-a-user-for-confirmation-in-bash-script
    read -p "Will remove intermediate model states under \"result/\". Confirm? y/[n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    fi
fi

for x in `ls results`; do
    if ! ls results/${x}/* &> /dev/null; then
        continue
    fi
    if ! ls results/${x}/model?*.pt &> /dev/null; then
        echo "Remove $x"
        if [[ ${DRY_RUN} == 0 ]]; then
            rm -r results/${x}
        fi
        continue
    fi
    keep=`ls results/${x}/model?*.pt | sort -V | tail -n 1`
    echo "Keep: $keep"
    for y in `ls results/${x}/model*.pt`; do
        if [[ ${y} != ${keep} ]]; then
            echo ${y}
            if [[ ${DRY_RUN} -eq 0 ]]; then
                rm ${y}
            fi
        fi
    done
done
