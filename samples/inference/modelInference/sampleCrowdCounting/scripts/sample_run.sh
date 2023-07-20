#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
Src=${ScriptPath}/../src

mkdir ${Src}/../out
cd ${Src}
echo "[INFO] The sample starts to run"
running_command="python3.7 sampleCrowdCounting.py"
${running_command}
if [ $? -ne 0 ];then
    echo "[INFO] The program runs failed"
else
    echo "[INFO] The program runs successfully"
fi