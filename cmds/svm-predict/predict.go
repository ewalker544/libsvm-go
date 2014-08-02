package main

import (
	"fmt"
	"github.com/ewalker544/libsvm-go"
	"io"
)

func runPrediction(prob *libSvm.Problem, param *libSvm.Parameter, model *libSvm.Model, outputFp io.Writer) {
	var sump float64 = 0
	var sumt float64 = 0
	var sumpp float64 = 0
	var sumtt float64 = 0
	var sumpt float64 = 0
	var correct int = 0
	var total int = 0
	var err float64 = 0

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

		err += (predictLabel - targetLabel) * (predictLabel - targetLabel)
		sump += predictLabel
		sumt += targetLabel
		sumpp += predictLabel * predictLabel
		sumtt += targetLabel * targetLabel
		sumpt += predictLabel * targetLabel
		total++
	}

	if param.SvmType == libSvm.NU_SVR || param.SvmType == libSvm.EPSILON_SVR {
		fmt.Printf("Mean squared error = %g (regression)\n", float64(err)/float64(total))
		squaredCoeff := ((float64(total)*sumpt - sump*sumt) * (float64(total)*sumpt - sump*sumt)) /
			((float64(total)*sumpp - sump*sump) * (float64(total)*sumtt - sumt*sumt))
		fmt.Printf("Squared correlation coefficient = %g (regression)\n", squaredCoeff)
	} else {
		accuracy := float64(correct) / float64(total) * 100
		fmt.Printf("Accuracy = %g%% (%d/%d) (classification)\n", accuracy, correct, total)
	}
}
