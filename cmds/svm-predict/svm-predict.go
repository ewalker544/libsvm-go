/*
** Copyright 2014 Edward Walker
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
** http ://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
**
** @author: Ed Walker
 */
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
