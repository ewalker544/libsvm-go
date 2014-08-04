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
** @author: Ed Walker
 */
package main

import (
	"flag"
	"fmt"
	"github.com/ewalker544/libsvm-go"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

var outFP io.Writer = os.Stdout
var gParam *libSvm.Parameter

type probabilityType int

func (q *probabilityType) String() string {
	return ("Probability Type")
}

func (q *probabilityType) Set(value string) error {
	val, err := strconv.Atoi(value)
	if err != nil || val < 0 || val > 1 {
		return fmt.Errorf("Invalid probability value (-b %d)\n", val)
	}
	if val == 0 {
		gParam.Probability = false
	} else {
		gParam.Probability = true
	}
	return nil
}

func usage() {
	fmt.Print(
		"Usage: svm-predict [options] test_file model_file [output_file]\n",
		"options:\n",
		"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n",
		"-q : quiet mode (no outputs)\n")
}

func parseOptions(param *libSvm.Parameter) (testFile string, modelFile string, outputFile string) {

	gParam = param

	var probabilityTypeFlag probabilityType

	flag.Var(&probabilityTypeFlag, "b", "")
	flag.BoolVar(&param.QuietMode, "q", false, "")

	flag.Usage = usage
	flag.Parse()

	switch {
	case len(flag.Args()) < 2:
		usage()
		os.Exit(1)
	case len(flag.Args()) == 2:
		testFile = flag.Arg(0)
		modelFile = flag.Arg(1)
		outputFile = getOutputFileName(testFile)
	default:
		testFile = flag.Arg(0)
		modelFile = flag.Arg(1)
		outputFile = flag.Arg(2)
	}

	if param.QuietMode {
		outFP = ioutil.Discard
	}

	return // testFile, modelFile, outputFile
}

func getOutputFileName(file string) string {
	var outputFile []string
	outputFile = append(outputFile, file)
	outputFile = append(outputFile, ".out")
	return strings.Join(outputFile, "")
}
