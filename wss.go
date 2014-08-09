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
** Description: Working-set selection
** Ref: R.-E. Fan, P.-H. Chen, and C.-J. Lin. "Working set selection using second order information for training SVM". Journal of Machine Learning Research 6 (2005)
** @author: Ed Walker
 */
package libSvm

import (
	"math"
)

type workingSetSelecter interface {
	workingSetSelect(solver *solver) (int, int, int)
	calculateRho(solver *solver) (float64, float64)
}

type selectWorkingSet struct{} // for classical solver

type selectWorkingSetNU struct{} // for NU solver

func (s selectWorkingSet) workingSetSelect(solver *solver) (int, int, int) {
	var gmax float64 = -math.MaxFloat64
	var gmax2 float64 = -math.MaxFloat64
	var gmax_idx int = -1
	var gmin_idx int = -1

	for i := 0; i < solver.l; i++ {
		if solver.y[i] == 1 {
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmax {
					gmax = -solver.gradient[i]
					gmax_idx = i
				}
			}
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmax2 {
					gmax2 = solver.gradient[i]
				}
			}
		} else {
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmax {
					gmax = solver.gradient[i]
					gmax_idx = i
				}
			}
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmax2 {
					gmax2 = -solver.gradient[i]
				}
			}
		}
	}

	if gmax+gmax2 < solver.eps {
		return -1, -1, 1
	}

	i := gmax_idx

	Qi := solver.q.getQ(i, solver.l)

	gmin_idx = s.findGminIdx(i, gmax, Qi, solver)

	//fmt.Printf("gmax_idx=%d, gmin_idx=%d\n", gmax_idx, gmin_idx)
	return gmax_idx, gmin_idx, 0
}

func (s *selectWorkingSet) findGminIdx(i int, gmax float64, Qi []cacheDataType, solver *solver) (gmin_idx int) {

	objDiffMin := make([]float64, solver.parRunner.numCPU)
	idx := make([]int, solver.parRunner.numCPU)
	for k := 0; k < solver.parRunner.numCPU; k++ {
		objDiffMin[k] = math.MaxFloat64
		idx[k] = -1
	}

	run := func(tid, start, end int) {
		//for j := 0; j < solver.l; j++ {
		for j := start; j < end; j++ {
			if solver.y[j] == 1 {
				if !solver.isLowerBound(j) {
					grad_diff := gmax + solver.gradient[j]
					if grad_diff > 0 {
						var obj_diff float64
						quad_coef := solver.qd[i] + solver.qd[j] - 2.0*float64(solver.y[i])*float64(Qi[j])
						if quad_coef > 0 {
							obj_diff = -(grad_diff * grad_diff) / quad_coef
						} else {
							obj_diff = -(grad_diff * grad_diff) / TAU
						}
						if obj_diff <= objDiffMin[tid] {
							objDiffMin[tid] = obj_diff
							idx[tid] = j
						}
					}
				}
			} else {
				if !solver.isUpperBound(j) {
					grad_diff := gmax - solver.gradient[j]
					if grad_diff > 0 {
						var obj_diff float64
						quad_coeff := solver.qd[i] + solver.qd[j] + 2.0*float64(solver.y[i])*float64(Qi[j])
						if quad_coeff > 0 {
							obj_diff = -(grad_diff * grad_diff) / quad_coeff
						} else {
							obj_diff = -(grad_diff * grad_diff) / quad_coeff
						}
						if obj_diff <= objDiffMin[tid] {
							objDiffMin[tid] = obj_diff
							idx[tid] = j
						}
					}
				}
			}
		}
	}

	solver.parRunner.run(run)
	solver.parRunner.waitAll()

	var objMin float64 = objDiffMin[0]
	gmin_idx = idx[0]
	for k := 1; k < solver.parRunner.numCPU; k++ {
		if objDiffMin[k] <= objMin {
			objMin = objDiffMin[k]
			gmin_idx = idx[k]
		}
	}

	return
}

func (s selectWorkingSet) calculateRho(solver *solver) (float64, float64) {
	var ub float64 = math.MaxFloat64
	var lb float64 = -math.MaxFloat64
	var sum_free float64 = 0
	var nr_free int = 0
	var r float64 = 0
	for i := 0; i < solver.l; i++ {
		yG := float64(solver.y[i]) * solver.gradient[i]
		if solver.isUpperBound(i) {
			if solver.y[i] == -1 {
				ub = minf(ub, yG)
			} else {
				lb = maxf(lb, yG)
			}
		} else if solver.isLowerBound(i) {
			if solver.y[i] == 1 {
				ub = minf(ub, yG)
			} else {
				lb = maxf(lb, yG)
			}
		} else {
			nr_free = nr_free + 1
			sum_free = sum_free + yG
		}
	}

	if nr_free > 0 {
		r = sum_free / float64(nr_free)
	} else {
		r = (ub + lb) / 2
	}

	return r, 0
}

func (s selectWorkingSetNU) workingSetSelect(solver *solver) (int, int, int) {
	var gmaxp float64 = -math.MaxFloat64
	var gmaxp2 float64 = -math.MaxFloat64
	var gmaxp_idx int = -1

	var gmaxn float64 = -math.MaxFloat64
	var gmaxn2 float64 = -math.MaxFloat64
	var gmaxn_idx int = -1

	var gmin_idx int = -1

	for i := 0; i < solver.l; i++ {
		if solver.y[i] == 1 {
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmaxp {
					gmaxp = -solver.gradient[i]
					gmaxp_idx = i
				}
			}
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmaxp2 {
					gmaxp2 = solver.gradient[i]
				}
			}
		} else {
			if !solver.isLowerBound(i) {
				if solver.gradient[i] >= gmaxn {
					gmaxn = solver.gradient[i]
					gmaxn_idx = i
				}
			}
			if !solver.isUpperBound(i) {
				if -solver.gradient[i] >= gmaxn2 {
					gmaxn2 = -solver.gradient[i]
				}
			}
		}
	}

	if maxf(gmaxp+gmaxp2, gmaxn+gmaxn2) < solver.eps {
		return -1, -1, 1 // done!
	}

	ip := gmaxp_idx
	in := gmaxn_idx

	var Qip []cacheDataType
	if ip != -1 {
		Qip = solver.q.getQ(ip, solver.l)
	}
	var Qin []cacheDataType
	if in != -1 {
		Qin = solver.q.getQ(in, solver.l)
	}

	gmin_idx = s.findGminIdx(ip, in, gmaxp, gmaxn, Qip, Qin, solver)

	var out_j int = gmin_idx
	var out_i int
	if solver.y[out_j] == 1 {
		out_i = gmaxp_idx
	} else {
		out_i = gmaxn_idx
	}

	return out_i, out_j, 0
}

func (s selectWorkingSetNU) findGminIdx(ip, in int, gmaxp, gmaxn float64, Qip, Qin []cacheDataType, solver *solver) (gmin_idx int) {

	objDiffMin := make([]float64, solver.parRunner.numCPU)
	idx := make([]int, solver.parRunner.numCPU)
	for k := 0; k < solver.parRunner.numCPU; k++ {
		objDiffMin[k] = math.MaxFloat64
		idx[k] = -1
	}

	run := func(tid, start, end int) {
		//for j := 0; j < solver.l; j++ {
		for j := start; j < end; j++ {
			if solver.y[j] == 1 {
				if !solver.isLowerBound(j) {
					grad_diff := gmaxp + solver.gradient[j]
					if grad_diff > 0 {
						var obj_diff float64
						quad_coef := solver.qd[ip] + solver.qd[j] - 2*float64(Qip[j])
						if quad_coef > 0 {
							obj_diff = -(grad_diff * grad_diff) / quad_coef
						} else {
							obj_diff = -(grad_diff * grad_diff) / TAU
						}

						if obj_diff <= objDiffMin[tid] {
							objDiffMin[tid] = obj_diff
							idx[tid] = j
						}
					}
				}
			} else {
				if !solver.isUpperBound(j) {
					grad_diff := gmaxn - solver.gradient[j]
					if grad_diff > 0 {
						var obj_diff float64
						quad_coef := solver.qd[in] + solver.qd[j] - 2*float64(Qin[j])
						if quad_coef > 0 {
							obj_diff = -(grad_diff * grad_diff) / quad_coef
						} else {
							obj_diff = -(grad_diff * grad_diff) / TAU
						}

						if obj_diff <= objDiffMin[tid] {
							objDiffMin[tid] = obj_diff
							idx[tid] = j
						}
					}
				}
			}
		}
	}

	solver.parRunner.run(run)
	solver.parRunner.waitAll()

	var objMin float64 = objDiffMin[0]
	gmin_idx = idx[0]
	for k := 1; k < solver.parRunner.numCPU; k++ {
		if objDiffMin[k] <= objMin {
			objMin = objDiffMin[k]
			gmin_idx = idx[k]
		}
	}

	return
}

func (s selectWorkingSetNU) calculateRho(solver *solver) (float64, float64) {
	var nr_free1 int = 0
	var nr_free2 int = 0
	var ub1 float64 = math.MaxFloat64
	var ub2 float64 = math.MaxFloat64
	var lb1 float64 = -math.MaxFloat64
	var lb2 float64 = -math.MaxFloat64
	var sum_free1 float64 = 0
	var sum_free2 float64 = 0

	for i := 0; i < solver.l; i++ {
		if solver.y[i] == 1 {
			if solver.isUpperBound(i) {
				lb1 = maxf(lb1, solver.gradient[i])
			} else if solver.isLowerBound(i) {
				ub1 = minf(ub1, solver.gradient[i])
			} else {
				nr_free1++
				sum_free1 += solver.gradient[i]
			}
		} else {
			if solver.isUpperBound(i) {
				lb2 = maxf(lb2, solver.gradient[i])
			} else if solver.isLowerBound(i) {
				ub2 = minf(ub2, solver.gradient[i])
			} else {
				nr_free2++
				sum_free2 += solver.gradient[i]
			}

		}
	}

	var r1 float64
	var r2 float64

	if nr_free1 > 0 {
		r1 = sum_free1 / float64(nr_free1)
	} else {
		r1 = (ub1 + lb1) / 2.0
	}

	if nr_free2 > 0 {
		r2 = sum_free2 / float64(nr_free2)
	} else {
		r2 = (ub2 + lb2) / 2.0
	}

	return (r1 - r2) / 2, (r1 + r2) / 2
}
