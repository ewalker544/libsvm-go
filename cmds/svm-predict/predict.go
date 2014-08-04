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
	"io"
)

func runPrediction(prob *libSvm.Problem, param *libSvm.Parameter, model *libSvm.Model, outputFp io.Writer) {

	squareErr := libSvm.NewSquareErrorComputer()
	var total int = 0
	var correct int = 0

	for prob.Begin(); !prob.Done(); prob.Next() { // Iterate through the entire label/vector problem set

		// read each vector in the problem file, one at a time
		targetLabel, x := prob.GetLine() // get the target label and its vector

		var predictLabel float64
		if param.Probability && (param.SvmType == libSvm.C_SVC || param.SvmType == libSvm.NU_SVC) {
			label, probabilityEstimate := model.PredictProbability(x)
			predictLabel = label
			for j := 0; j < model.NrClass(); j++ {
				fmt.Fprintf(outputFp, " %g", probabilityEstimate[j])
			}
			fmt.Fprintln(outputFp, "")
		} else {
			predictLabel = model.Predict(x)
			fmt.Fprintf(outputFp, " %g\n", predictLabel)
		}

		if predictLabel == targetLabel { // does the prediciton match the target label
			correct++
		}

		squareErr.Sum(predictLabel, targetLabel)
		total++
	}

	if param.SvmType == libSvm.NU_SVR || param.SvmType == libSvm.EPSILON_SVR {
		fmt.Fprintf(outFP, "Mean squared error = %.6g (regression)\n", squareErr.MeanSquareError())
		fmt.Fprintf(outFP, "Squared correlation coefficient = %.6g (regression)\n", squareErr.SquareCorrelationCoeff())
	} else {
		accuracy := float64(correct) / float64(total) * 100
		fmt.Fprintf(outFP, "Accuracy = %.6g%% (%d/%d) (classification)\n", accuracy, correct, total)
	}
}
