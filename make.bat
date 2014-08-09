@echo off
go install
cd cmds\svm-train
go install
cd ..\svm-predict
go install
cd ..\..
