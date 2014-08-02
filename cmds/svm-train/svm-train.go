package main

import (
	"fmt"
	"github.com/ewalker544/libsvm-go"
	"os"
)

func main() {
	param := libSvm.NewParameter()                      // create a parameter type
	nrFold, trainFile, modelFile := parseOptions(param) // parse command-line flags for SVM parameter

	prob, err := libSvm.NewProblem(trainFile, param) // create a problem type from the train file and the parameter
	if err != nil {
		fmt.Fprint(os.Stderr, "Fail to create a libSvm.Problem: ", err)
		os.Exit(1)
	}

	if nrFold > 0 {
		doCrossValidation(prob, param, nrFold)
	} else {
		model := libSvm.NewModel(param) // create a model from specified parameter
		model.Train(prob)               // use model to train on the problem data
		model.Dump(modelFile)           // dump model into the user-specified file
	}
}
