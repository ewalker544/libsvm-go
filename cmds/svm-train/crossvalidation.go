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

		fmt.Printf("Cross Validation Mean squared error = %g\n", squareErr.MeanSquareError())
		fmt.Printf("Cross Validation Squared correlation coefficient = %g\n", squareErr.SquareCorrelationCoeff())
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
		fmt.Printf("Cross Validation Accuracy = %g%%\n", 100*float64(correct)/float64(prob.ProblemSize()))

	}
}
