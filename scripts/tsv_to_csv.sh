#!/bin/bash

file=$1
[ ! -f $file ] && echo "file $file not found" && exit 1

output=${file%.tsv}.csv

tr '\t' ',' < $file > $output
