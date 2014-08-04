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
)

func doCrossValidation(prob *libSvm.Problem, param *libSvm.Parameter, nrFold int) {

	targets := libSvm.CrossValidation(prob, param, nrFold)

	if param.SvmType == libSvm.EPSILON_SVR || param.SvmType == libSvm.NU_SVR {

		squareErr := libSvm.NewSquareErrorComputer()

		var i int = 0
		for prob.Begin(); !prob.Done(); prob.Next() {
			y, _ := prob.GetLine()
			v := targets[i]
			squareErr.Sum(v, y)
			i++
		}

		fmt.Fprintf(outFP, "Cross Validation Mean squared error = %.6g\n", squareErr.MeanSquareError())
		fmt.Fprintf(outFP, "Cross Validation Squared correlation coefficient = %.6g\n", squareErr.SquareCorrelationCoeff())
	} else {
		var i int = 0
		var correct int = 0
		for prob.Begin(); !prob.Done(); prob.Next() {
			y, _ := prob.GetLine()
			if targets[i] == y {
				correct++
			}
			i++
		}
		fmt.Fprintf(outFP, "Cross Validation Accuracy = %.6g%%\n", 100*float64(correct)/float64(prob.ProblemSize()))

	}
}
