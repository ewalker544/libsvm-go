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
** Description: Model describes the properties of the Support Vector Machine after training.
** @author: Ed Walker
 */
package libSvm

import (
	"fmt"
	"math"
	"os"
)

type Model struct {
	param     *Parameter
	l         int
	nrClass   int
	label     []int
	rho       []float64
	nSV       []int
	sV        []int
	svSpace   []snode
	svIndices []int
	svCoef    [][]float64
	probA     []float64
	probB     []float64
}

func NewModel(param *Parameter) *Model {
	return &Model{param: param}
}

func NewModelFromFile(file string) *Model {
	param := NewParameter()
	model := NewModel(param)
	model.ReadModel(file)	
	return model
}

func (model Model) NrClass() int {
	return model.nrClass
}

func groupClasses(prob *Problem) (nrClass int, label []int, start []int, count []int, perm []int) {
	var l int = prob.l

	label = make([]int, 0)
	count = make([]int, 0)
	data_label := make([]int, l)

	for i := 0; i < l; i++ { // find unqie labels and put them in the label slice
		this_label := int(prob.y[i])
		var j int
		for j = 0; j < len(label); j++ {
			if this_label == label[j] {
				count[j]++
				break
			}
		}
		if j == len(label) { // this is a new label we just encountered
			label = append(label, this_label)
			count = append(count, 1)
		}
		data_label[i] = j // remember what label index was assigned to SV i
	}

	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	if len(label) == 2 && label[0] == -1 && label[1] == 1 {
		label[0], label[1] = label[1], label[0] // swap
		count[0], count[1] = count[1], count[0] // swap
		for i := 0; i < l; i++ {
			if data_label[i] == 0 {
				data_label[i] = 1
			} else {
				data_label[i] = 0
			}
		}
	}

	nrClass = len(label) // number of unique labels found
	start = make([]int, nrClass)
	start[0] = 0
	for i := 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}

	perm = make([]int, l)
	for i := 0; i < l; i++ {
		label_idx := data_label[i]
		next_avail_pos := start[label_idx]
		perm[next_avail_pos] = i // index i will be assigned to this position
		start[label_idx]++       // move to the next available position for this label
	}

	start[0] = 0
	for i := 1; i < nrClass; i++ { // reset the starting position again
		start[i] = start[i-1] + count[i-1]
	}

	return // nrClass, label, start, count, perm
}

func (model *Model) classification(prob *Problem) {

	nrClass, label, start, count, perm := groupClasses(prob) // group SV with the same labels together

	var l int = prob.l
	x := make([]int, l)
	for i := 0; i < l; i++ {
		x[i] = prob.x[perm[i]] // this is the new x slice with the grouped SVs
	}

	weighted_C := make([]float64, nrClass)
	for i := 0; i < nrClass; i++ {
		weighted_C[i] = model.param.C
	}
	for i := 0; i < model.param.NrWeight; i++ { // this is only done if the relative weight of the labels have been set by the user
		var j int = 0
		for j = 0; j < nrClass; j++ {
			if model.param.WeightLabel[i] == label[j] {
				break
			}
		}
		if j == nrClass {
			fmt.Fprintf(os.Stderr, "WARNING: class label %d specified in weight is not found\n", model.param.WeightLabel[i])
		} else {
			weighted_C[j] = weighted_C[j] * model.param.Weight[i] // multiple with user specified weight for label
		}
	}

	nonzero := make([]bool, l)
	for i := 0; i < l; i++ {
		nonzero[i] = false
	}

	totalCompares := nrClass * (nrClass - 1) / 2
	decisions := make([]decision, totalCompares) // slice for appending all our decisions.
	var probA, probB []float64
	if model.param.Probability {
		probA = make([]float64, totalCompares)
		probB = make([]float64, totalCompares)
	}

	var p int = 0
	for i := 0; i < nrClass; i++ {
		for j := i + 1; j < nrClass; j++ {
			var subProb Problem

			si := start[i] // SV starting from x[si] are related to label i
			sj := start[j] // SV starting from x[sj] are related to label j

			ci := count[i] // number of SV from x[si] that are related to label i
			cj := count[j] // number of SV from x[sj] that are related to label j

			subProb.xSpace = prob.xSpace // inherits the space
			subProb.l = ci + cj          // focus only on 2 labels
			subProb.x = make([]int, subProb.l)
			subProb.y = make([]float64, subProb.l)
			for k := 0; k < ci; k++ {
				subProb.x[k] = x[si+k] // starting indices for first label
				subProb.y[k] = 1
			}

			for k := 0; k < cj; k++ {
				subProb.x[ci+k] = x[sj+k] // starting indices for second label
				subProb.y[ci+k] = -1
			}

			if model.param.Probability {
				probA[p], probB[p] = binarySvcProbability(&subProb, model.param, weighted_C[i], weighted_C[j])
			}

			if decision_result, err := train_one(&subProb, model.param, weighted_C[i], weighted_C[j]); err == nil { // no error in training

				decisions[p] = decision_result

				for k := 0; k < ci; k++ {
					if !nonzero[si+k] && math.Abs(decisions[p].alpha[k]) > 0 {
						nonzero[si+k] = true
					}
				}
				for k := 0; k < cj; k++ {
					if !nonzero[sj+k] && math.Abs(decisions[p].alpha[ci+k]) > 0 {
						nonzero[sj+k] = true
					}
				}

			} else {
				fmt.Fprintln(os.Stderr, "WARNING: training failed: ", err)
				return // no point in continuing
			}

			p++
		}
	}

	// Update the model!
	model.nrClass = nrClass
	model.label = make([]int, nrClass)
	for i := 0; i < nrClass; i++ {
		model.label[i] = label[i]
	}

	model.rho = make([]float64, len(decisions))
	for i := 0; i < len(decisions); i++ {
		model.rho[i] = decisions[i].rho
	}

	if model.param.Probability {
		model.probA = probA
		model.probB = probB
	}

	var totalSV int = 0
	nz_count := make([]int, nrClass)
	model.nSV = make([]int, nrClass)
	for i := 0; i < nrClass; i++ {
		var nSV int = 0
		for j := 0; j < count[i]; j++ {
			if nonzero[start[i]+j] {
				nSV++
				totalSV++
			}
		}
		model.nSV[i] = nSV
		nz_count[i] = nSV
	}

	if !model.param.QuietMode {
		fmt.Printf("Total nSV = %d\n", totalSV)
	}

	model.l = totalSV
	model.svSpace = prob.xSpace

	model.sV = make([]int, totalSV)
	model.svIndices = make([]int, totalSV)

	p = 0
	for i := 0; i < l; i++ {
		if nonzero[i] {
			model.sV[p] = x[i]
			model.svIndices[p] = perm[i] + 1
			p++
		}
	}

	nzStart := make([]int, nrClass)
	nzStart[0] = 0
	for i := 1; i < nrClass; i++ {
		nzStart[i] = nzStart[i-1] + nz_count[i-1]
	}

	model.svCoef = make([][]float64, nrClass-1)
	for i := 0; i < nrClass-1; i++ {
		model.svCoef[i] = make([]float64, totalSV)
	}

	p = 0
	for i := 0; i < nrClass; i++ {
		for j := i + 1; j < nrClass; j++ {

			// classifier (i,j): coefficients with
			// i are in svCoef[j-1][nzStart[i]...],
			// j are in svCoef[i][nzStart[j]...]

			si := start[i]
			sj := start[j]

			ci := count[i]
			cj := count[j]

			q := nzStart[i]
			for k := 0; k < ci; k++ {
				if nonzero[si+k] {
					model.svCoef[j-1][q] = decisions[p].alpha[k]
					q++
				}
			}
			q = nzStart[j]
			for k := 0; k < cj; k++ {
				if nonzero[sj+k] {
					model.svCoef[i][q] = decisions[p].alpha[ci+k]
					q++
				}
			}
			p++
		}
	}

}

func (model *Model) regressionOneClass(prob *Problem) {

	model.nrClass = 2

	if model.param.Probability &&
		(model.param.SvmType == EPSILON_SVR || model.param.SvmType == NU_SVR) {
		model.probA = make([]float64, 1)
		model.probA[0] = svrProbability(prob, model.param)
	}

	if decision_result, err := train_one(prob, model.param, 0, 0); err == nil { // no error in training
		model.rho = append(model.rho, decision_result.rho)

		var nSV int = 0
		for i := 0; i < prob.l; i++ {
			if math.Abs(decision_result.alpha[i]) > 0 {
				nSV++
			}
		}

		model.l = nSV
		model.svSpace = prob.xSpace
		model.sV = make([]int, nSV)
		model.svCoef = make([][]float64, 1)
		model.svCoef[0] = make([]float64, nSV)
		model.svIndices = make([]int, nSV)

		var j int = 0
		for i := 0; i < prob.l; i++ {
			if math.Abs(decision_result.alpha[i]) > 0 {
				model.sV[j] = prob.x[i]
				model.svCoef[0][j] = decision_result.alpha[i]
				model.svIndices[j] = i + 1
				j++
			}
		}
	} else {
		fmt.Fprintln(os.Stderr, "WARNING: training failed: ", err)
	}
}

func (model *Model) Train(prob *Problem) error {
	switch model.param.SvmType {
	case C_SVC, NU_SVC:
		model.classification(prob)
	case ONE_CLASS, EPSILON_SVR, NU_SVR:
		model.regressionOneClass(prob)
	}
	return nil
}

