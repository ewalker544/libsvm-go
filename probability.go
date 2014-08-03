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
** Description: Probability estimation APIs
** @author: Ed Walker
 */
package libSvm

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

/**
* This function does classification or regression on a test vector x
   given a model with probability information.

   For a classification model with probability information, this
   function gives nrClass probability estimates in the slice
   probabilityEstimate. The class with the highest probability is returned
   in returnValue. For regression/one-class SVM, probabilityEsstimate is nil,
   and returnValue is the same as that of Predict.

*/
func (model Model) PredictProbability(x map[int]float64) (returnValue float64, probabilityEstimate []float64) {

	if (model.param.SvmType == C_SVC || model.param.SvmType == NU_SVC) &&
		model.probA != nil && model.probB != nil {

		var nrClass int = model.nrClass
		_, decisionValues := model.PredictValues(x)

		var minProb float64 = 1e-7

		pairWiseProb := make([][]float64, nrClass)
		for i := 0; i < nrClass; i++ {
			pairWiseProb[i] = make([]float64, nrClass)
		}

		var k int = 0
		for i := 0; i < nrClass; i++ {
			for j := i + 1; j < nrClass; j++ {
				m := maxf(sigmoidPredict(decisionValues[k], model.probA[k], model.probB[k]), minProb)
				pairWiseProb[i][j] = minf(m, 1-minProb)
				pairWiseProb[j][i] = 1 - pairWiseProb[i][j]
				k++
			}
		}

		probabilityEstimate = multiClassProbability(nrClass, pairWiseProb)

		var maxIdx int = 0
		for i := 1; i < nrClass; i++ {
			if probabilityEstimate[i] > probabilityEstimate[maxIdx] {
				maxIdx = i
			}
		}

		returnValue = float64(model.label[maxIdx])
		return // returnValue, probabilityEstimates
	} else {
		probabilityEstimate = nil
		returnValue = model.Predict(x)
		return // returnValue, probabilityEstimates
	}

}

func sigmoidPredict(decisionValue, A, B float64) float64 {
	fApB := decisionValue*A + B
	if fApB >= 0 {
		return math.Exp(-fApB) / (1 + math.Exp(-fApB))
	} else {
		return 1 / (1 + math.Exp(fApB))
	}
}

func multiClassProbability(k int, r [][]float64) []float64 {
	p := make([]float64, k)

	Q := make([][]float64, k)
	Qp := make([]float64, k)
	eps := 0.005 / float64(k)

	for t := 0; t < k; t++ {
		p[t] = 1.0 / float64(k)
		Q[t] = make([]float64, k)
		Q[t][t] = 0
		for j := 0; j < t; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = Q[j][t]
		}
		for j := t + 1; j < k; j++ {
			Q[t][t] += r[j][t] * r[j][t]
			Q[t][j] = -r[j][t] * r[t][j]
		}
	}

	var pQp float64
	var iter int = 0
	var maxIter int = maxi(100, k)
	for iter = 0; iter < maxIter; iter++ {
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp = 0
		for t := 0; t < k; t++ {
			Qp[t] = 0
			for j := 0; j < k; j++ {
				Qp[t] += Q[t][j] * p[j]
			}
			pQp += p[t] * Qp[t]
		}

		var maxError float64 = 0
		for t := 0; t < k; t++ {
			err := math.Abs(Qp[t] - pQp)
			if err > maxError {
				maxError = err
			}
		}

		if maxError < eps {
			break
		}

		for t := 0; t < k; t++ {
			diff := (-Qp[t] + pQp) / Q[t][t]
			p[t] += diff
			pQp = (pQp + diff*(diff*Q[t][t]+2*Qp[t])) / (1 + diff) / (1 + diff)

			for j := 0; j < k; j++ {
				Qp[j] = (Qp[j] + diff*Q[t][j]) / (1 + diff)
				p[j] /= (1 + diff)
			}
		}
	}

	if iter >= maxIter {
		fmt.Println("Exceeds max_iter in multiclass_prob")
	}

	return p
}

/**
 * Cross-validation decision values for probability estimates
 * @return probA, probB
 */
func binarySvcProbability(prob *Problem, param *Parameter, Cp, Cn float64) (probA float64, probB float64) {
	var nrFold int = 5
	perm := make([]int, prob.l)
	decisionValues := make([]float64, prob.l)

	for i := 0; i < prob.l; i++ {
		perm[i] = i
	}

	random := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	for i := 0; i < prob.l; i++ {
		j := i + random.Intn(prob.l-i)
		//j := i + randIntn(prob.l-i) // DEBUG
		perm[i], perm[j] = perm[j], perm[i]
	}

	for i := 0; i < nrFold; i++ {
		begin := i * prob.l / nrFold
		end := (i + 1) * prob.l / nrFold

		var subProb Problem
		subProb.xSpace = prob.xSpace
		subProb.l = prob.l - (end - begin)
		subProb.x = make([]int, subProb.l)
		subProb.y = make([]float64, subProb.l)

		var k int = 0
		for j := 0; j < begin; j++ {
			subProb.x[k] = prob.x[perm[j]]
			subProb.y[k] = prob.y[perm[j]]
			k++
		}
		for j := end; j < prob.l; j++ {
			subProb.x[k] = prob.x[perm[j]]
			subProb.y[k] = prob.y[perm[j]]
			k++
		}

		var pCount int = 0
		var nCount int = 0
		for j := 0; j < k; j++ {
			if subProb.y[j] > 0 {
				pCount++
			} else {
				nCount++
			}
		}

		if pCount == 0 && nCount == 0 {
			for j := begin; j < end; j++ {
				decisionValues[perm[j]] = 0
			}
		} else if pCount > 0 && nCount == 0 {
			for j := begin; j < end; j++ {
				decisionValues[perm[j]] = 1
			}
		} else if pCount == 0 && nCount > 0 {
			for j := begin; j < end; j++ {
				decisionValues[perm[j]] = -1
			}
		} else {
			subParam := *param
			subParam.Probability = false
			subParam.C = 1
			subParam.NrWeight = 2
			subParam.WeightLabel = make([]int, 2)
			subParam.Weight = make([]float64, 2)
			subParam.WeightLabel[0] = 1
			subParam.WeightLabel[1] = -1
			subParam.Weight[0] = Cp
			subParam.Weight[1] = Cn
			subModel := NewModel(&subParam)
			subModel.Train(&subProb)
			for j := begin; j < end; j++ {
				idx := prob.x[perm[j]]
				x := SnodeToMap(prob.xSpace[idx:])
				_, subProbDecision := subModel.PredictValues(x)
				decisionValues[perm[j]] = subProbDecision[0] * float64(subModel.label[0])
			}
		}
	}

	probA, probB = sigmoidTrain(prob.l, decisionValues, prob.y)
	return // probA, probB
}

func sigmoidTrain(l int, decisionValues, labels []float64) (probA float64, probB float64) {
	var prior1 float64 = 0
	var prior0 float64 = 0
	probA = 0
	probB = 0

	for i := 0; i < l; i++ {
		if labels[i] > 0 {
			prior1++
		} else {
			prior0++
		}
	}

	var maxIter = 100
	var minStep float64 = 1e-10
	var sigma float64 = 1e-12
	var eps float64 = 1e-5
	hiTarget := (prior1 + 1) / (prior1 + 2)
	loTarget := 1 / (prior0 + 2)
	t := make([]float64, l)

	var fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize float64
	var newA, newB, newf, d1, d2 float64
	var iter int

	probA = 0
	probB = math.Log((prior0 + 1) / (prior1 + 1))
	var fval float64 = 0

	for i := 0; i < l; i++ {
		if labels[i] > 0 {
			t[i] = hiTarget
		} else {
			t[i] = loTarget
		}
		fApB = decisionValues[i]*probA + probB
		if fApB > 0 {
			fval += t[i]*fApB + math.Log(1+math.Exp(-fApB))
		} else {
			fval += (t[i]-1)*fApB + math.Log(1+math.Exp(fApB))
		}
	}

	for iter = 0; iter < maxIter; iter++ {
		h11 = sigma
		h22 = sigma
		h21 = 0
		g1 = 0
		g2 = 0
		for i := 0; i < l; i++ {
			fApB = decisionValues[i]*probA + probB

			if fApB >= 0 {
				p = math.Exp(-fApB) / (1 + math.Exp(-fApB))
				q = 1 / (1 + math.Exp(-fApB))
			} else {
				p = 1 / (1 + math.Exp(fApB))
				q = math.Exp(fApB) / (1 + math.Exp(fApB))
			}

			d2 = p * q
			h11 += decisionValues[i] * decisionValues[i] * d2
			h22 += d2
			h21 += decisionValues[i] * d2
			d1 = t[i] - p
			g1 += decisionValues[i] * d1
			g2 += d1
		}

		// Stopping criteria
		if math.Abs(g1) < eps && math.Abs(g2) < eps {
			break
		}

		// Finding Newton direction: -inv(H') * g
		det = h11*h22 - h21*h21
		dA = -(h22*g1 - h21*g2) / det
		dB = -(-h21*g1 + h11*g2) / det
		gd = g1*dA + g2*dB

		stepsize = 1 // Line Search
		for stepsize >= minStep {
			newA = probA + stepsize*dA
			newB = probB + stepsize*dB

			// New function value
			newf = 0.0
			for i := 0; i < l; i++ {
				fApB = decisionValues[i]*newA + newB
				if fApB >= 0 {
					newf += t[i]*fApB + math.Log(1+math.Exp(-fApB))
				} else {
					newf += (t[i]-1)*fApB + math.Log(1+math.Exp(fApB))
				}
			}
			// Check sufficient decrease
			if newf < fval+0.0001*stepsize*gd {
				probA = newA
				probB = newB
				fval = newf
				break
			} else {
				stepsize = stepsize / 2.0
			}
		}

		if stepsize < minStep {
			fmt.Printf("Line search fails in two-class probability estimates\n")
			break
		}

	}

	if iter >= maxIter {
		fmt.Printf("Reaching maximal iterations in two-class probability estimates\n")
	}

	return // probA, probB
}

/**
 * Return parameter of a Laplace distribution
 */
func svrProbability(prob *Problem, param *Parameter) float64 {
	var nrFold int = 5
	var mae float64 = 0

	var newParam Parameter = *param
	newParam.Probability = false

	ymv := CrossValidation(prob, &newParam, nrFold)

	for i := 0; i < prob.l; i++ {
		ymv[i] = prob.y[i] - ymv[i]
		mae += math.Abs(ymv[i])
	}

	mae /= float64(prob.l)
	std := math.Sqrt(2 * mae * mae)

	var count int = 0
	mae = 0
	for i := 0; i < prob.l; i++ {
		if math.Abs(ymv[i]) > 5*std {
			count = count + 1
		} else {
			mae += math.Abs(ymv[i])
		}
	}
	mae /= float64(prob.l - count)
	fmt.Printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n", mae)

	return mae
}
