package libSvm

// import "fmt" // DEBUG

/**
*  This function gives decision values on a test vector x given a
   model, and return the predicted label (classification) or
   the function value (regression).

   For a classification model with nrClass classes, this function
   gives nrClass*(nrClass-1)/2 decision values in the array
   decisionValues, where nrClass can be obtained from the model.
   The order is label[0] vs. label[1], ...,
   label[0] vs. label[nr_class-1], label[1] vs. label[2], ...,
   label[nrClass-2] vs. label[nrClass-1], where label can be
   obtained from the model. The returned value is
   the predicted class for x. Note that when nrClass = 1, this
   function does not give any decision value.

   For a regression model, decisionValues[0] and the returned value are
   both the function value of x calculated using the model. For a
   one-class model, decisionValues[0] is the decision value of x, while
   the returned value is +1/-1.

*/
func (model Model) PredictValues(x map[int]float64) (returnValue float64, decisionValues []float64) {
	returnValue = 0

	px := MapToSnode(x)

	switch model.param.SvmType {
	case ONE_CLASS, EPSILON_SVR, NU_SVR:
		var svCoef []float64 = model.svCoef[0]

		var sum float64 = 0
		for i := 0; i < model.l; i++ {
			var idx_y int = model.sV[i]
			py := model.svSpace[idx_y:]
			sum += svCoef[i] * computeKernelValue(px, py, model.param)
		}
		sum -= model.rho[0]

		decisionValues = append(decisionValues, sum)

		if model.param.SvmType == ONE_CLASS {
			if sum > 0 {
				returnValue = 1 // return
			} else {
				returnValue = -1 // return
			}
			return // returnValue, decisionValues
		} else {
			returnValue = sum
			return // returnValue, decisionValues
		}

	case C_SVC, NU_SVC:
		var nrClass int = model.nrClass
		var l int = model.l

		kvalue := make([]float64, l)
		for i := 0; i < l; i++ {
			var idx_y int = model.sV[i]
			py := model.svSpace[idx_y:]
			kvalue[i] = computeKernelValue(px, py, model.param)
			/*
				if i < 3 { // DEBUG
					dumpSnode("px: ", px)
					dumpSnode("py: ", py)
					fmt.Printf("kvalue[%d]=%f\n", i, kvalue[i])
				}
			*/
		}

		start := make([]int, nrClass)
		start[0] = 0
		for i := 1; i < nrClass; i++ {
			start[i] = start[i-1] + model.nSV[i-1]
		}

		vote := make([]int, nrClass)
		for i := 0; i < nrClass; i++ {
			vote[i] = 0
		}

		var p int = 0
		for i := 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				var sum float64 = 0

				var si int = start[i]
				var sj int = start[j]

				var ci int = model.nSV[i]
				var cj int = model.nSV[j]

				coef1 := model.svCoef[j-1]
				coef2 := model.svCoef[i]
				for k := 0; k < ci; k++ {
					sum += coef1[si+k] * kvalue[si+k]
				}
				for k := 0; k < cj; k++ {
					sum += coef2[sj+k] * kvalue[sj+k]
				}
				sum -= model.rho[p]
				decisionValues = append(decisionValues, sum)
				if sum > 0 {
					vote[i]++
				} else {
					vote[j]++
				}
				p++
			}
		}

		var maxIdx int = 0
		for i := 1; i < nrClass; i++ {
			if vote[i] > vote[maxIdx] {
				maxIdx = i
			}
		}

		returnValue = float64(model.label[maxIdx])
		return // returnValue, decisionValues
	}

	return
}

/**
* This function does classification or regression on a test vector x
   given a model.

   For a classification model, the predicted class for x is returned.
   For a regression model, the function value of x calculated using
   the model is returned. For an one-class model, +1 or -1 is
   returned.

*/
func (model Model) Predict(x map[int]float64) float64 {

	predict, _ := model.PredictValues(x)

	return predict
}
