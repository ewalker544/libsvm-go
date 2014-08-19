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
** Description: Sequential Minimal Optimization (SMO) solver
** Ref: C.-C. Chang, C.-J. Lin. "LIBSVM: A library for support vector machines". ACM Transactions on Intelligent Systems and Technology 2 (2011)
** @author: Ed Walker
 */
package libSvm

import (
	"fmt"
	"math"
)

const (
	LOWER_BOUND = iota
	UPPER_BOUND = iota
	FREE        = iota
)

type solver struct {
	l            int     // problem size
	q            matrixQ // Q matrix
	p            []float64
	gradient     []float64
	alpha        []float64
	alpha_status []int8
	qd           []float64 // Q matrix diagonial values
	penaltyCp    float64
	penaltyCn    float64
	y            []int8 // class, +1 or -1
	eps          float64
	workingSet   workingSetSelecter
	parRunner    parallelRunner
	quietMode    bool
	numCPU       int
}

func (solver solver) isUpperBound(i int) bool {
	if solver.alpha_status[i] == UPPER_BOUND {
		return true
	} else {
		return false
	}
}

func (solver solver) isLowerBound(i int) bool {
	if solver.alpha_status[i] == LOWER_BOUND {
		return true
	} else {
		return false
	}
}

func (solver solver) getC(i int) float64 {
	if solver.y[i] > 0 {
		return solver.penaltyCp
	} else {
		return solver.penaltyCn
	}
}

func (solver *solver) updateAlphaStatus(i int) {
	if solver.alpha[i] >= solver.getC(i) {
		solver.alpha_status[i] = UPPER_BOUND
	} else if solver.alpha[i] <= 0 {
		solver.alpha_status[i] = LOWER_BOUND
	} else {
		solver.alpha_status[i] = FREE
	}
}

func (solver *solver) solve() solution {

	solver.alpha_status = make([]int8, solver.l)
	for i := 0; i < solver.l; i++ {
		solver.updateAlphaStatus(i)
	}

	// Initialize gradient
	solver.gradient = make([]float64, solver.l)
	for i := 0; i < solver.l; i++ {
		solver.gradient[i] = solver.p[i]
	}

	for i := 0; i < solver.l; i++ {
		var alpha_i float64 = solver.alpha[i]
		Q_i := solver.q.getQ(i, solver.l) // getQ() is parallelized in the respective matrixQ implementation
		solver.initGradientInnerLoop(Q_i, alpha_i)
	}
	// solver.initGradient() // Alternative parallelization strategy - no improvement

	var iter int = 0
	var max_iter int = 0
	if solver.l > math.MaxInt32/100 {
		max_iter = math.MaxInt32
	} else {
		max_iter = 100 * solver.l
	}
	max_iter = maxi(10000000, max_iter)
	var counter = mini(solver.l, 1000) + 1

	for iter < max_iter {
		if counter = counter - 1; counter == 0 {
			counter = mini(solver.l, 1000)
			if !solver.quietMode {
				fmt.Print(".")
			}
		}

		var i int = 0
		var j int = 0
		var rc int = 0
		if i, j, rc = solver.workingSet.workingSetSelect(solver); rc != 0 {
			if !solver.quietMode {
				fmt.Print("*")
			}
			break
		}

		iter++

		C_i := solver.getC(i)
		C_j := solver.getC(j)

		oldAlpha_i := solver.alpha[i]
		oldAlpha_j := solver.alpha[j]

		Q_i := solver.q.getQ(i, solver.l) // row i of Q matrix
		Q_j := solver.q.getQ(j, solver.l) // row j of Q matrix

		if solver.y[i] != solver.y[j] {

			quad_coef := solver.qd[i] + solver.qd[j] + 2*float64(Q_i[j])
			if quad_coef <= 0 {
				quad_coef = TAU
			}

			delta := (-solver.gradient[i] - solver.gradient[j]) / quad_coef
			diff := solver.alpha[i] - solver.alpha[j]
			solver.alpha[i] += delta
			solver.alpha[j] += delta

			if diff > 0 {
				if solver.alpha[j] < 0 {
					solver.alpha[j] = 0
					solver.alpha[i] = diff
				}
			} else {
				if solver.alpha[i] < 0 {
					solver.alpha[i] = 0
					solver.alpha[j] = -diff
				}
			}

			if diff > C_i-C_j {
				if solver.alpha[i] > C_i {
					solver.alpha[i] = C_i
					solver.alpha[j] = C_i - diff
				}
			} else {
				if solver.alpha[j] > C_j {
					solver.alpha[j] = C_j
					solver.alpha[i] = C_j + diff
				}
			}

		} else {

			quad_coef := solver.qd[i] + solver.qd[j] - 2*float64(Q_i[j])
			if quad_coef <= 0 {
				quad_coef = TAU
			}

			delta := (solver.gradient[i] - solver.gradient[j]) / quad_coef
			sum := solver.alpha[i] + solver.alpha[j]
			solver.alpha[i] -= delta
			solver.alpha[j] += delta

			if sum > C_i {
				if solver.alpha[i] > C_i {
					solver.alpha[i] = C_i
					solver.alpha[j] = sum - C_i
				}
			} else {
				if solver.alpha[j] < 0 {
					solver.alpha[j] = 0
					solver.alpha[i] = sum
				}
			}

			if sum > C_j {
				if solver.alpha[j] > C_j {
					solver.alpha[j] = C_j
					solver.alpha[i] = sum - C_j
				}
			} else {
				if solver.alpha[i] < 0 {
					solver.alpha[i] = 0
					solver.alpha[j] = sum
				}
			}
		}

		deltaAlpha_i := solver.alpha[i] - oldAlpha_i
		deltaAlpha_j := solver.alpha[j] - oldAlpha_j
		solver.updateGradient(Q_i, Q_j, deltaAlpha_i, deltaAlpha_j)

		solver.updateAlphaStatus(i)
		solver.updateAlphaStatus(j)
	}

	var si solution

	si.rho, si.r = solver.workingSet.calculateRho(solver)

	var v float64 = 0 // calculate objective value
	for i := 0; i < solver.l; i++ {
		v += solver.alpha[i] * (solver.gradient[i] + solver.p[i])
	}
	si.obj = v / 2

	si.upper_bound_p = solver.penaltyCp
	si.upper_bound_n = solver.penaltyCn

	si.alpha = solver.alpha

	if !solver.quietMode {
		fmt.Printf("\noptimization finished, #iter = %d\n", iter)
	}
	// solver.q.showCacheStats() // show cache statistics

	return si
}

func (solver *solver) initGradientInnerLoop(Q_i []cacheDataType, alpha_i float64) {

	run := func(tid, start, end int) {
		// for j := 0; j < solver.l; j++
		for j := start; j < end; j++ {
			solver.gradient[j] += alpha_i * float64(Q_i[j])
		}
	}

	solver.parRunner.run(run)
	solver.parRunner.waitAll()
}

func (solver *solver) initGradient() {
	run := func(tid, start, end int) {
		//for j := 0; j < solver.l; j++ {
		for j := start; j < end; j++ {
			for i := 0; i < solver.l; i++ {
				solver.gradient[j] += solver.alpha[i] + solver.q.computeQ(i, j)
			}
		}
	}

	solver.parRunner.run(run)
	solver.parRunner.waitAll()
}

func (solver *solver) updateGradient(Q_i, Q_j []cacheDataType, deltaAlpha_i, deltaAlpha_j float64) {

	run := func(tid, start, end int) {
		for k := start; k < end; k++ {
			t := float64(Q_i[k])*deltaAlpha_i + float64(Q_j[k])*deltaAlpha_j
			solver.gradient[k] += t
		}
	}

	solver.parRunner.run(run)  // run the closure in parallel
	solver.parRunner.waitAll() // wait for all the parallel runs to complete
}

func newSolver(l int, q matrixQ, p []float64, y []int8, alpha []float64, penaltyCp, penaltyCn, eps float64, nu bool, quietMode bool, numCPU int) solver {

	solver := solver{l: l, q: q, p: p, y: y, alpha: alpha,
		penaltyCp: penaltyCp, penaltyCn: penaltyCn, eps: eps, quietMode: quietMode, numCPU: numCPU}
	if nu {
		solver.workingSet = selectWorkingSetNU{}
	} else {
		solver.workingSet = selectWorkingSet{}
	}
	solver.qd = q.getQD()
	solver.parRunner = newParallelRunner(solver.l, numCPU)

	return solver
}
