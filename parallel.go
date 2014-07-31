package libsvm

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

func (p parallelRunner) run(f func(int, int)) {

	funcWrap := func(start, end int) {
		f(start, end)
		p.done <- true
	}

	p.reset()
	for i := 0; i < p.numCPU; i++ {
		start, end := p.next()
		go funcWrap(start, end)
	}
}

func (p parallelRunner) waitAll() {

	for i := 0; i < p.numCPU; i++ {
		<-p.done
	}

}

func NewParallelRunner(n int) parallelRunner {
	cpus := runtime.NumCPU() // query the number of available CPUs
	runtime.GOMAXPROCS(cpus) // set this as the number of usable CPUs

	cpus = runtime.GOMAXPROCS(0) // get the new max number of CPUs

	p := parallelRunner{N: n, numCPU: cpus}

	p.rem = p.N % p.numCPU
	p.constRem = p.rem
	p.block = p.N / p.numCPU
	p.start = 0

	p.done = make(chan bool, 1) // synchronization channel for run()/waitAll()

	return p
}
