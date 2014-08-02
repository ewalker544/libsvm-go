package main

import (
	"fmt"
	"github.com/ewalker544/libsvm-go"
	"os"
)

func main() {
	param := libSvm.NewParameter() // create a parameter type

	testFile, modelFile, outputFile := parseOptions(param) // parse command-line flags for SVM parameter
	outputFp, err := os.Create(outputFile)                 // create output file
	if err != nil {
		panic(err)
	}

	prob, err := libSvm.NewProblem(testFile, param) // create a problem type
	if err != nil {
		fmt.Fprint(os.Stderr, "Fail to create a problem type:", err)
		os.Exit(1)
	}

	model := libSvm.NewModel(param) // create a model type

	if err := model.ReadModel(modelFile); err != nil { // populate model with properties in model file
		fmt.Fprint(os.Stderr, "Fail to read model file: ", err)
		os.Exit(1)
	}

	runPrediction(prob, param, model, outputFp) // run the prediction loop
}
