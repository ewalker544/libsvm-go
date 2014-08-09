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
** Description: Functions for calling the solver for different problem scenerios, i.e. SVC, SVR, or One-Class
** @author: Ed Walker
 */
package libSvm

import (
	"fmt"
	"math"
)

type trainError struct {
	val int
	msg string
}

func (e *trainError) Error() string {
	return fmt.Sprintf("%d -- %s\n", e.val, e.msg)
}

type solution struct {
	obj           float64
	rho           float64
	upper_bound_p float64
	upper_bound_n float64
	alpha         []float64
	r             float64
}

type decision struct {
	alpha []float64
	rho   float64
}

func train_one(prob *Problem, param *Parameter, Cp, Cn float64) (decision, error) {

	var si solution
	switch param.SvmType {
	case C_SVC:
		si = solveCSVC(prob, param, Cp, Cn)
	case NU_SVC:
		si = solveNuSVC(prob, param)
	case ONE_CLASS:
		si = solveOneClass(prob, param)
	case EPSILON_SVR:
		si = solveEpsilonSVR(prob, param)
	case NU_SVR:
		si = solveNuSVR(prob, param)
	default:
		return decision{}, &trainError{val: param.SvmType, msg: "svm type not supported"}
	}

	if !param.QuietMode {
		fmt.Printf("obj = %f, rho = %f\n", si.obj, si.rho)
	}
	alpha := si.alpha

	var nSV int = 0
	var nBSV int = 0
	for i := 0; i < prob.l; i++ {
		if math.Abs(alpha[i]) > 0 {
			nSV++
			if prob.y[i] > 0 {
				if math.Abs(alpha[i]) >= si.upper_bound_p {
					nBSV++
				}
			} else {
				if math.Abs(alpha[i]) >= si.upper_bound_n {
					nBSV++
				}
			}
		}
	}

	if !param.QuietMode {
		fmt.Printf("nSV = %d, nBSV = %d\n", nSV, nBSV)
	}

	return decision{alpha: alpha, rho: si.rho}, nil
}

func solveCSVC(prob *Problem, param *Parameter, Cp, Cn float64) solution {
	var l int = prob.l

	alpha := make([]float64, l)
	minus_one := make([]float64, l)
	y := make([]int8, l)

	for i := 0; i < l; i++ {
		alpha[i] = 0
		minus_one[i] = -1
		if prob.y[i] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	s := newSolver(l, newSVCQ(prob, param, y), minus_one, y, alpha, Cp, Cn, param.Eps, false /*not nu*/, param.QuietMode, param.NumCPU)
	si := s.solve() // generate solution

	var sum_alpha float64 = 0
	for i := 0; i < l; i++ {
		sum_alpha = sum_alpha + si.alpha[i]
		si.alpha[i] = si.alpha[i] * float64(y[i])
	}

	if Cp == Cn {
		if !param.QuietMode {
			t := Cp * float64(l)
			fmt.Printf("nu = %f\n", sum_alpha/t)
		}
	}

	return si // return solution
}

func solveNuSVC(prob *Problem, param *Parameter) solution {
	var l int = prob.l
	var nu float64 = param.Nu

	alpha := make([]float64, l)
	y := make([]int8, l)
	zeros := make([]float64, l)

	for i := 0; i < l; i++ {
		if prob.y[i] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	sum_pos := nu * float64(l) / 2
	sum_neg := sum_pos

	for i := 0; i < l; i++ {
		if y[i] == 1 {
			alpha[i] = minf(1, sum_pos)
			sum_pos -= alpha[i]
		} else {
			alpha[i] = minf(1, sum_neg)
			sum_neg -= alpha[i]
		}
	}

	for i := 0; i < l; i++ {
		zeros[i] = 0
	}

	s := newSolver(l, newSVCQ(prob, param, y), zeros, y, alpha, 1, 1, param.Eps, true /*nu*/, param.QuietMode, param.NumCPU)
	si := s.solve()

	r := si.r
	if !param.QuietMode {
		fmt.Printf("C = %v\n", 1.0/r)
	}

	for i := 0; i < l; i++ {
		si.alpha[i] *= (float64(y[i]) / r)
	}

	si.rho /= r
	si.obj /= (r * r)
	si.upper_bound_p = 1 / r
	si.upper_bound_n = 1 / r

	return si
}

func solveOneClass(prob *Problem, param *Parameter) solution {
	var l int = prob.l

	alpha := make([]float64, l)
	zeros := make([]float64, l)
	ones := make([]int8, l)

	var n int = int(param.Nu) * prob.l
	for i := 0; i < n; i++ {
		alpha[i] = 1
	}
	if n < l {
		alpha[n] = param.Nu*float64(l) - float64(n)
	}
	for i := n + 1; i < l; i++ {
		alpha[i] = 0
	}

	for i := 0; i < l; i++ {
		zeros[i] = 0
		ones[i] = 1
	}

	s := newSolver(l, newOneClassQ(prob, param), zeros, ones, alpha, 1, 1, param.Eps, false /*not nu*/, param.QuietMode, param.NumCPU)
	si := s.solve()

	return si
}

func solveEpsilonSVR(prob *Problem, param *Parameter) solution {
	var l int = prob.l

	alpha := make([]float64, 2*l)
	linear_term := make([]float64, 2*l)
	y := make([]int8, 2*l)

	for i := 0; i < l; i++ {
		alpha[i] = 0
		linear_term[i] = param.P - prob.y[i]
		y[i] = 1

		alpha[i+l] = 0
		linear_term[i+l] = param.P + prob.y[i]
		y[i+l] = -1
	}

	s := newSolver(2*l, newSVRQ(prob, param), linear_term, y, alpha, param.C, param.C, param.Eps, false /*not nu*/, param.QuietMode, param.NumCPU)
	si := s.solve()

	var sum_alpha float64 = 0
	for i := 0; i < l; i++ {
		si.alpha[i] = si.alpha[i] - si.alpha[i+l]
		sum_alpha += math.Abs(si.alpha[i])
	}
	si.alpha = si.alpha[:l]

	var nu float64 = sum_alpha / (param.C * float64(l))
	if !param.QuietMode {
		fmt.Printf("nu = %v\n", nu)
	}

	return si
}

func solveNuSVR(prob *Problem, param *Parameter) solution {
	var l int = prob.l
	var C float64 = param.C

	alpha := make([]float64, 2*l)
	linear_term := make([]float64, 2*l)
	y := make([]int8, 2*l)

	var sum float64 = C * param.Nu * float64(l) / 2.0

	for i := 0; i < l; i++ {
		alpha[i] = minf(sum, C)
		alpha[i+l] = alpha[i]

		sum -= alpha[i]

		linear_term[i] = -prob.y[i]
		y[i] = 1

		linear_term[i+l] = prob.y[i]
		y[i+l] = -1
	}

	s := newSolver(2*l, newSVRQ(prob, param), linear_term, y, alpha, param.C, param.C, param.Eps, true /*nu*/, param.QuietMode, param.NumCPU)
	si := s.solve()

	if !param.QuietMode {
		fmt.Printf("epsilon = %f\n", -si.r)
	}

	for i := 0; i < l; i++ {
		si.alpha[i] = si.alpha[i] - si.alpha[i+l]
	}
	si.alpha = si.alpha[:l]

	return si
}
