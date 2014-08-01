package main

import (
	"github.com/ewalker544/libsvm-go"
)

func main() {
	var param libSvm.Parameter
	trainFile, modelFile := parseOptions(&param) // parse options for SVM parameter

	var prob libSvm.Problem      // create a problem type
	prob.Read(trainFile, &param) // read in the problem from the user-specified file

	model := libSvm.NewModel(&param) // create a model from specified parameter
	model.Train(&prob)               // use model to train on the problem data

	model.Dump(modelFile) // dump model into the user-specified file
}
