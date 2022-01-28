#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

phase=$1
only_validate=$2

docker build -t airogs_$phase "$SCRIPTPATH" --build-arg phase=$phase --build-arg only_validate=$only_validate
