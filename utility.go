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
** Description: Useful functions used in various parts of the library
** @author: Ed Walker
 */
package libSvm

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

const TAU float64 = 1e-12

func absi(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func mini(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func maxi(a, b int) int {
	if b < a {
		return a
	} else {
		return b
	}
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	} else {
		return b
	}
}

func maxf(a, b float64) float64 {
	if b < a {
		return a
	} else {
		return b
	}
}

func MapToSnode(m map[int]float64) []snode {

	keys := make([]int, len(m))
	var i int = 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Ints(keys) // We MUST do this to ensure that we add snodes in ascending key order!
	// Just iterating over the map does not ensure the keys are returned in ascending order.

	x := make([]snode, len(m)+1)

	i = 0
	for _, k := range keys {
		x[i] = snode{index: k, value: m[k]}
		i++
	}
	x[i] = snode{index: -1}

	return x
}

func SnodeToMap(x []snode) map[int]float64 {

	m := make(map[int]float64)

	for i := 0; x[i].index != -1; i++ {
		m[x[i].index] = x[i].value
	}

	return m
}

/**
 * Reads a complete line with bufio.Reader and returns it as a string
 * Attribution: http://stackoverflow.com/questions/8757389/reading-file-line-by-line-in-go
 * Thank you Malcolm!
 */
func readline(r *bufio.Reader) (string, error) {
	var (
		isPrefix bool  = true
		err      error = nil
		line, ln []byte
	)
	for isPrefix && err == nil {
		line, isPrefix, err = r.ReadLine()
		ln = append(ln, line...)
	}
	return string(ln), err
}

// Mostly for Debugging
func getModelFileName(file string) string {
	var model_file []string
	model_file = append(model_file, file)
	model_file = append(model_file, ".model")
	return strings.Join(model_file, "")
}

func getTrainFileName(file string) string {
	var train_file []string
	train_file = append(train_file, file)
	train_file = append(train_file, ".train")
	return strings.Join(train_file, "")
}

func getTestFileName(file string) string {
	var test_file []string
	test_file = append(test_file, file)
	test_file = append(test_file, ".test")
	return strings.Join(test_file, "")
}

func dumpSnode(msg string, px []snode) {
	fmt.Print(msg)
	for i := 0; px[i].index != -1; i++ {
		fmt.Printf("%d:%g ", px[i].index, px[i].value)
	}
	fmt.Println("")
}

func printSpace(x []int, x_space []snode) {
	for idx, i := range x {
		fmt.Printf("x[%d]=%d: ", idx, i)
		for x_space[i].index != -1 {
			fmt.Printf("%d:%g ", x_space[i].index, x_space[i].value)
			i++
		}
		fmt.Printf("\n")
	}
	os.Exit(0)
}

func dump(g []float64) {
	for i, v := range g {
		fmt.Printf("[%d]=%g\n", i, v)
	}
	os.Exit(0)
}

/**
 * simple predictable random number generator in rang
 */
var nSeedi int = 1

func randIntn(rang int) int {
	nSeedi = (7 * nSeedi) % 11
	p := float64(nSeedi) / 11.0
	return int(p * float64(rang))
}
