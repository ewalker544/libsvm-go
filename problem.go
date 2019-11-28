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
** Description: Describes problem, i.e. label/vector set
** @author: Ed Walker
 */
package libSvm

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
	param  *Parameter
}

// Struct for training
type TrainData struct {
	Y []float64 // Labels for multilabel classification problem like [1, 2, 3 ...]
	X []map[int]float64 // Values by map as index like [map[1:11.0 2:10 ...]]
	lY int // Init length of the labels slice, we should check by this param if can train
	lX int // Length of the values for every label point
}


func NewProblem(file string, param *Parameter) (*Problem, error) {
	prob := &Problem{l: 0, i: 0}
	err := prob.Read(file, param)
	return prob, err
}

func NewTrainData(labels []float64, values []map[int]float64) *TrainData {
	return &TrainData{
		Y: labels,
		X: values,
		lY: len(labels),
		lX: len(values),
	}
}

// Create initial problem object
// when we can start add row instances
func InitProblem(param *Parameter) *Problem {
	return &Problem{
		l:      0,
		y:      nil,
		x:      nil,
		xSpace: nil,
		i:      0,
		param:  param,
	}
}

func (problem *Problem) FitProblem(train *TrainData) error {
	if train.lY != train.lX {
		return fmt.Errorf("Train data is not have equal params for futures and classes labels len: %d <> values len: %d\n", train.lY, train.lX)
	}
	var max_idx int = 0
	problem.l = train.lY

	var xIndex = make(map[int]int, 0)
	for problem.Begin(); !problem.Done(); problem.Next() {
		problem.x = append(problem.x, len(problem.xSpace))
		label := train.Y[problem.i]
		problem.y = append(problem.y, label)

		for index, value := range train.X[problem.i] {
			problem.xSpace = append(problem.xSpace, snode{index: index, value: value})
			if index > max_idx {
				max_idx = index
			}

			if xIndex[index] != index {
				xIndex[index] = index

				//problem.x = append(problem.x, index)
			}
		}

		problem.xSpace = append(problem.xSpace, snode{index: -1})
	}

	if problem.param.Gamma == 0 && max_idx > 0 {
		problem.param.Gamma = 1.0 / float64(max_idx)
	}

	fmt.Println("Problem l", problem.l)
	fmt.Println("Problem y", problem.y)
	fmt.Println("Problem x", problem.x)
	fmt.Println("Problem xSpace", len(problem.xSpace), problem.xSpace)
	fmt.Println("Problem i", problem.i)

	return nil
}

func (problem *Problem) Read(file string, param *Parameter) error { // reads the problem from the specified file
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	problem.y = nil
	problem.x = nil
	problem.xSpace = nil

	reader := bufio.NewReader(f)
	var max_idx int = 0
	var l int = 0

	for {
		line, err := readline(reader)
		if err != nil {
			break
		}
		problem.x = append(problem.x, len(problem.xSpace))

		lineSansComments := strings.Split(line, "#") // remove any comments

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

	return nil
}

/**
 * Initialize the start of iterating through the labels and vectors in the problem set
 */
func (problem *Problem) Begin() {
	problem.i = 0
}

/**
 * Finished iterating through all the labels and vectors in the problem set
 */
func (problem *Problem) Done() bool {
	if problem.i >= problem.l {
		return true
	}
	return false
}

/**
 * Move to the next label and vector in the problem set
 */
func (problem *Problem) Next() {
	problem.i++
	return
}

/**
 * Return one label and vector from the problem set
 * @return y label
 * @return x vector (map of dimension/value)
 */
func (problem *Problem) GetLine() (y float64, x map[int]float64) {
	y = problem.y[problem.i]
	idx := problem.x[problem.i]
	x = SnodeToMap(problem.xSpace[idx:])
	return // y, x
}

/**
 * Returns number of label and vectors in the problem set
 * @return problem set size
 */
func (problem *Problem) ProblemSize() int {
	return problem.l
}
