package libsvm

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type snode struct {
	index int     // dimension (-1 indicates end of SV)
	value float64 // coeff
}

type Problem struct {
	l      int       // #SVs
	y      []float64 // labels
	x      []int     // starting indices in xSpace defining SVs
	xSpace []snode   // SV coeffs
	i      int       // counter for iterator
}

func (problem *Problem) Read(file string, param *Parameter) error { // reads the problem from the specified file
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	scanner := bufio.NewScanner(f)
	var max_idx int = 0
	var l int = 0
	//var j int = 0
	for scanner.Scan() {
		problem.x = append(problem.x, len(problem.xSpace))
		line := scanner.Text()
		lineSansComments := strings.Split(line, "#")  // remove any comments
		tokens := strings.Fields(lineSansComments[0]) // get all the word tokens (seperated by white spaces)
		if label, err := strconv.ParseFloat(tokens[0], 64); err == nil {
			problem.y = append(problem.y, label)
		} else {
			return fmt.Errorf("Fail to parse label\n")
		}

		space := tokens[1:]
		for _, w := range space {
			if len(w) > 0 {
				node := strings.Split(w, ":")
				if len(node) > 1 {
					var index int
					var value float64
					if index, err = strconv.Atoi(node[0]); err != nil {
						return fmt.Errorf("Fail to parse index from token %v\n", w)
					}
					if value, err = strconv.ParseFloat(node[1], 64); err != nil {
						return fmt.Errorf("Fail to parse value from token %v\n", w)
					}
					problem.xSpace = append(problem.xSpace, snode{index: index, value: value})
					if index > max_idx {
						max_idx = index
					}

				}
			}
		}

		problem.xSpace = append(problem.xSpace, snode{index: -1})
		l++
	}
	problem.l = l

	if param.Gamma == 0 && max_idx > 0 {
		param.Gamma = 1.0 / float64(max_idx)
	}

	return scanner.Err()
}

func (problem *Problem) Begin() {
	problem.i = 0
}

func (problem *Problem) Done() bool {
	if problem.i >= problem.l {
		return true
	}
	return false
}

func (problem *Problem) Next() {
	problem.i++
	return
}

func (problem *Problem) Get() (y float64, x map[int]float64) {
	y = problem.y[problem.i]
	idx := problem.x[problem.i]
	x = SnodeToMap(problem.xSpace[idx:])
	return // y, x
}

func (problem *Problem) ProblemSize() int {
	return problem.l
}
