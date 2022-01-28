#!/usr/bin/env bash

phase=$1
only_validate=$2

./build.sh $phase $only_validate

docker save airogs_$phase | xz -c > exports/airogs_$phase.tar.xz
