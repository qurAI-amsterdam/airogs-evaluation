#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

phase=$1
only_validate=$2

./build.sh $phase $only_validate

docker volume create airogs_$phase-output

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/$phase/:/input/ \
        -v $phase-output:/output/ \
        airogs_$phase

docker run --rm \
        -v $phase-output:/output/ \
        python:3.7-slim cat /output/metrics.json

docker volume rm $phase-output
