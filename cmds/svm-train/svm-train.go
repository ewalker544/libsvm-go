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
