#!/bin/zsh
PD=$1
echo $PD
pgrep -x python | grep $PD  
while :
do
    if pgrep -x python | grep $PD  
    then
        echo "Running"
        sleep 60
    else
        echo "Stopped"
        echo "Starting new process"
        bash script_cross.sh
        exit
    fi
done


