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
** Description: Q matrix for Support Vector Classification (svcQ), Support Vector Regression  (svrQ),
**              and One-Class Support Vector Machines (oneClassQ)
** @author: Ed Walker
 */
package libSvm

type matrixQ interface {
	getQ(i, l int) []cacheDataType // Returns all the Q matrix values for row i
	getQD() []float64              // Returns the Q matrix values for the diagonal
	computeQ(i, j int) float64     // Returns the Q matrix value at (i,j)
	showCacheStats()
}

/**
 * Q matrix for support vector classification (SVC)
 */
type svcQ struct {
	y         []int8
	qd        []float64
	kernel    kernelFunction
	parRunner parallelRunner
	colCache  *cache
}

/**
 * Returns the diagonal values
 */
func (q *svcQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for row i
 */
func (q *svcQ) getQ(i, l int) []cacheDataType {

	rcq, newData := q.colCache.getData(i)
	if newData {
		run := func(tid, start, end int) {
			for j := start; j < end; j++ { // compute column elements
				rcq[j] = cacheDataType(q.y[i]*q.y[j]) * cacheDataType(q.kernel.compute(i, j))
			}
		}

		q.parRunner.run(run)
		q.parRunner.waitAll()
	}

	return rcq
}

/**
 * Computes the Q[i,j] entry
 */
func (q *svcQ) computeQ(i, j int) float64 {
	return float64(q.y[i]*q.y[j]) * q.kernel.compute(i, j)
}

/**
 * Prints out the cache performance statistics
 */
func (q *svcQ) showCacheStats() {
	q.colCache.stats()
}

func newSVCQ(prob *Problem, param *Parameter, y []int8) *svcQ {
	kernel, err := newKernel(prob, param)
	if err != nil {
		panic(err)
	}

	qd := make([]float64, prob.l)
	for i := 0; i < prob.l; i++ {
		qd[i] = kernel.compute(i, i)
	}

	return &svcQ{y: y, qd: qd, kernel: kernel, parRunner: newParallelRunner(prob.l, param.NumCPU), colCache: newCache(prob.l, prob.l, param.CacheSize)}
}

/**
 * Q matrix for one-class support vector machines: determines if new data is likely to be in one class (novality detection).
 */
type oneClassQ struct {
	qd        []float64
	kernel    kernelFunction
	parRunner parallelRunner
	colCache  *cache
}

/**
 * Returns the diagonal values
 */
func (q *oneClassQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for row i
 */
func (q *oneClassQ) getQ(i, l int) []cacheDataType {

	rcq, newData := q.colCache.getData(i)
	if newData {
		run := func(tid, start, end int) {
			for j := start; j < end; j++ { // compute column elements
				rcq[j] = cacheDataType(q.kernel.compute(i, j))
			}
		}

		q.parRunner.run(run)
		q.parRunner.waitAll()
	}

	return rcq
}

/**
 * Computes the Q[i,j] entry
 */
func (q *oneClassQ) computeQ(i, j int) float64 {
	return q.kernel.compute(i, j)
}

/**
 * Prints out the cache performance statistics
 */
func (q *oneClassQ) showCacheStats() {
	q.colCache.stats()
}

func newOneClassQ(prob *Problem, param *Parameter) *oneClassQ {
	kernel, err := newKernel(prob, param)
	if err != nil {
		panic(err)
	}

	qd := make([]float64, prob.l)
	for i := 0; i < prob.l; i++ {
		qd[i] = kernel.compute(i, i)
	}

	return &oneClassQ{qd: qd, kernel: kernel, parRunner: newParallelRunner(prob.l, param.NumCPU), colCache: newCache(prob.l, prob.l, param.CacheSize)}
}

/**
 * Q matrix for support vector regression
 */
type svrQ struct {
	l         int       // problem size
	qd        []float64 // Q matrix diagonial values
	kernel    kernelFunction
	parRunner parallelRunner
	colCache  *cache
}

func (q *svrQ) real_idx(i int) int {
	if i < q.l {
		return i
	} else {
		return i - q.l
	}
}

func (q *svrQ) sign(i int) float64 {
	if i < q.l {
		return 1
	} else {
		return -1
	}
}

/**
 * Returns the diagonal values
 */
func (q *svrQ) getQD() []float64 {
	return q.qd
}

/**
 * Get Q values for row i
 */
func (q *svrQ) getQ(i, l int) []cacheDataType { // @param l is 2 * q.l
	sign_i := q.sign(i)
	real_i := q.real_idx(i)

	// NOTE: query cache with "real_i" since cache stores [0,l)
	rcq, newData := q.colCache.getData(real_i)
	if newData {
		run := func(tid, start, end int) {
			for j := start; j < end; j++ { // compute column elements
				t := q.kernel.compute(real_i, j)
				rcq[j] = cacheDataType(sign_i * q.sign(j) * t)
				rcq[j+q.l] = cacheDataType(sign_i * q.sign(j+l) * t)
			}
		}

		q.parRunner.run(run)
		q.parRunner.waitAll()
	}
	return rcq
}

/**
 * Computes the Q[i,j] entry
 */
func (q *svrQ) computeQ(i, j int) float64 {
	real_i := q.real_idx(i)
	real_j := q.real_idx(j)

	return q.sign(i) * q.sign(j) * q.kernel.compute(real_i, real_j)
}

/**
 * Prints out the cache performance statistics
 */
func (q *svrQ) showCacheStats() {
	q.colCache.stats()
}

func newSVRQ(prob *Problem, param *Parameter) *svrQ {
	kernel, err := newKernel(prob, param)
	if err != nil {
		panic(err)
	}

	l := prob.l
	qd := make([]float64, 2*l)
	for i := 0; i < l; i++ {
		qd[i] = kernel.compute(i, i)
		qd[i+l] = qd[i]
	}

	return &svrQ{l: l, qd: qd, kernel: kernel, parRunner: newParallelRunner(prob.l, param.NumCPU), colCache: newCache(prob.l, 2*prob.l, param.CacheSize)}
}
