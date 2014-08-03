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
** Description: Cross validation API
** @author: Ed Walker
 */
package libSvm

import (
	"fmt"
	"math/rand"
	"time"
)

/**
*  This function conducts cross validation. Data are separated to
   nrFold folds. Under given parameters, sequentially each fold is
   validated using the model from training the remaining. Predicted
   labels (of all prob's instances) in the validation process are
   stored in the slice called target.
*/
func CrossValidation(prob *Problem, param *Parameter, nrFold int) (target []float64) {
	var l int = prob.l

	target = make([]float64, l) // slice to return

	if nrFold > l {
		nrFold = l
		fmt.Printf("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n")
	}

	foldStart := make([]int, nrFold+1)

	perm := make([]int, l)
	random := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if (param.SvmType == C_SVC || param.SvmType == NU_SVC) && nrFold < l {

		nrClass, _, start, count, localPerm := groupClasses(prob) // group SV with the same labels together
		perm = localPerm
		// random shuffle and then data grouped by fold using the array perm
		foldCount := make([]int, nrFold)
		index := make([]int, l)
		for i := 0; i < l; i++ {
			index[i] = perm[i]
		}

		for c := 0; c < nrClass; c++ {
			for i := 0; i < count[c]; i++ {
				j := i + random.Intn(count[c]-i)
				//j := i + randIntn(count[c]-i)
				index[start[c]+j], index[start[c]+i] = index[start[c]+i], index[start[c]+j]
			}
		}

		for i := 0; i < nrFold; i++ {
			foldCount[i] = 0
			for c := 0; c < nrClass; c++ {
				foldCount[i] += (i+1)*count[c]/nrFold - i*count[c]/nrFold
			}
		}

		foldStart[0] = 0
		for i := 1; i <= nrFold; i++ {
			foldStart[i] = foldStart[i-1] + foldCount[i-1]
		}

		for c := 0; c < nrClass; c++ {
			for i := 0; i < nrFold; i++ {
				begin := start[c] + i*count[c]/nrFold
				end := start[c] + (i+1)*count[c]/nrFold
				for j := begin; j < end; j++ {
					perm[foldStart[i]] = index[j]
					foldStart[i]++
				}
			}
		}

		foldStart[0] = 0
		for i := 1; i <= nrFold; i++ {
			foldStart[i] = foldStart[i-1] + foldCount[i-1]
		}
	} else {

		for i := 0; i < l; i++ {
			perm[i] = i
		}

		for i := 0; i < l; i++ {
			j := i + random.Intn(l-i)
			perm[i], perm[j] = perm[j], perm[i]
		}

		for i := 0; i <= nrFold; i++ {
			foldStart[i] = i * l / nrFold
		}
	}

	for i := 0; i < nrFold; i++ {
		begin := foldStart[i]
		end := foldStart[i+1]

		var subProb Problem

		subProb.xSpace = prob.xSpace // inherit problem space
		subProb.l = l - (end - begin)
		subProb.x = make([]int, subProb.l)
		subProb.y = make([]float64, subProb.l)

		var k int = 0
		for j := 0; j < begin; j++ {
			subProb.x[k] = prob.x[perm[j]]
			subProb.y[k] = prob.y[perm[j]]
			k++
		}
		for j := end; j < l; j++ {
			subProb.x[k] = prob.x[perm[j]]
			subProb.y[k] = prob.y[perm[j]]
			k++
		}

		subModel := NewModel(param)
		subModel.Train(&subProb)

		if param.Probability &&
			(param.SvmType == C_SVC || param.SvmType == NU_SVC) {
			for j := begin; j < end; j++ {
				idx := prob.x[perm[j]]
				x := SnodeToMap(prob.xSpace[idx:])
				target[perm[j]], _ = subModel.PredictProbability(x)
			}
		} else {
			for j := begin; j < end; j++ {
				idx := prob.x[perm[j]]
				x := SnodeToMap(prob.xSpace[idx:])
				target[perm[j]] = subModel.Predict(x)
			}
		}
	}

	return
}
