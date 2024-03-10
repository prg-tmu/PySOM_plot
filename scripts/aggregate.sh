#!/bin/bash

dir=$1
max=$2

if [ -z $max ]; then
    echo "$0 [dir] [number of max invocation]"
    exit 1
fi

if [ ! -d "${dir}" ]; then
    echo "${dir}" not found
    exit 1
fi

agg=$dir/rebench_aggregate.data

[ -f $agg ] && rm $agg

i=1
for log in `find "$dir" -type f -name "*.data"`; do
    [ $i -gt $max ] && break
    sed -e "s/^1/${i}/g" "${log}" >> "${agg}"
    i=$(expr $i + 1)
done
