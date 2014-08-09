
all: libsvm svm-train svm-predict

libsvm:
	go install

svm-train:
	cd cmds/svm-train && go install

svm-predict:
	cd cmds/svm-predict && go install

.PHONY: libsvm svm-train svm-predict

~
~
~
~
~
~
~
~
~
"Makefile" 14L, 180C                                          14,0-1        All

