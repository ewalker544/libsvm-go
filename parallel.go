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
** Description: Useful types/methods for running loops in parallel.
** @author: Ed Walker
 */
package libSvm

import (
	//"fmt"
	"runtime"
)

type parallelRunner struct {
	N        int
	numCPU   int
	rem      int
	constRem int
	block    int
	start    int
	done     chan bool
}

/**
 * Returns the start and end iterations for a CPU
 */
func (p *parallelRunner) next() (int, int) {
	start := p.start
	end := p.start + p.block
	if p.rem > 0 {
		end++
		p.rem--
	}
	p.start = end
	return start, end
}

func (p *parallelRunner) reset() {
	p.rem = p.constRem
	p.start = 0
}

func (p parallelRunner) run(f func(int, int, int)) {

	funcWrap := func(tid, start, end int) {
		f(tid, start, end)
		p.done <- true
	}

	p.reset()
	for i := 0; i < p.numCPU; i++ {
		start, end := p.next()
		go funcWrap(i, start, end)
	}
}

func (p parallelRunner) waitAll() {

	for i := 0; i < p.numCPU; i++ {
		<-p.done
	}

}

func newParallelRunner(n, nCPU int) parallelRunner {
	var cpus int
	if nCPU < 1 {
		cpus = runtime.NumCPU()  // query the number of available CPUs
		runtime.GOMAXPROCS(cpus) // set this as the number of usable CPUs
	} else {
		runtime.GOMAXPROCS(nCPU)
	}
	cpus = runtime.GOMAXPROCS(0) // get the new max number of CPUs

	p := parallelRunner{N: n, numCPU: cpus}

	p.rem = p.N % p.numCPU
	p.constRem = p.rem
	p.block = p.N / p.numCPU
	p.start = 0

	p.done = make(chan bool, 1) // synchronization channel for run()/waitAll()

	return p
}
