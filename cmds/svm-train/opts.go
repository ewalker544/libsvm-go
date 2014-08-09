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

type svmType int

func (q *svmType) String() string {
	return string("Svm Type")
}

func (q *svmType) Set(value string) error {
	val, err := strconv.Atoi(value)
	if err != nil || val < 0 || val > 4 {
		return fmt.Errorf("Invalid svm type (-s %d)\n", val)
	}
	gParam.SvmType = val
	return nil
}

type kernelType int

func (q *kernelType) String() string {
	return string("Kernel type")
}

func (q *kernelType) Set(value string) error {
	val, err := strconv.Atoi(value)
	if err != nil || val < 0 || val > 4 {
		return fmt.Errorf("Invalid kernel type (-t %d)\n", val)
	}
	gParam.KernelType = val
	return nil
}

type weightType int

func (q *weightType) String() string {
	return string("Weight Type")
}

func (q *weightType) Set(value string) error {
	val := strings.Split(value, ",")
	if len(val) != 2 {
		return fmt.Errorf("Incorrect weight format.  The class and weight should be comma-separated, E.g. 1,0.5")
	}
	weightLabel, err := strconv.Atoi(val[0])
	if err != nil {
		return fmt.Errorf("Invalid label")
	}
	weight, err := strconv.ParseFloat(val[1], 64)
	if err != nil {
		return fmt.Errorf("Invalid label weight")
	}

	gParam.WeightLabel = append(gParam.WeightLabel, weightLabel)
	gParam.Weight = append(gParam.Weight, weight)

	return nil
}

func usage() {
	fmt.Print(
		"Usage: svm-train [options] training_set_file [model_file]\n",
		"options:\n",
		"-s svm_type : set type of SVM (default 0)\n",
		"	0 -- C-SVC		(multi-class classification)\n",
		"	1 -- nu-SVC		(multi-class classification)\n",
		"	2 -- one-class SVM\n",
		"	3 -- epsilon-SVR	(regression)\n",
		"	4 -- nu-SVR		(regression)\n",
		"-t kernel_type : set type of kernel function (default 2)\n",
		"	0 -- linear: u'*v\n",
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n",
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n",
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n",
		"	4 -- precomputed kernel (kernel values in training_set_file)\n",
		"-d degree : set degree in kernel function (default 3)\n",
		"-g gamma : set gamma in kernel function (default 1/num_features)\n",
		"-r coef0 : set coef0 in kernel function (default 0)\n",
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n",
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n",
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n",
		"-m cachesize : set cache memory size in MB (default 100)\n",
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n",
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n",
		"-w i,weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n",
		"-v n: n-fold cross validation mode\n",
		"-q : quiet mode (no outputs)\n",
		"-N n: number of CPUs to use (default -1 uses all available logical CPUs)\n")
}

func parseOptions(param *libSvm.Parameter) (nrFold int, trainFile string, modelFile string) {
	gParam = param // set gParam to the param so we can have svmType, kernelType, and weightType update it

	var svmTypeFlag svmType
	var kernelTypeFlag kernelType
	var weightTypeFlag weightType
	var probabilityTypeFlag probabilityType

	flag.Var(&svmTypeFlag, "s", "")
	flag.Var(&kernelTypeFlag, "t", "")
	flag.IntVar(&param.Degree, "d", 3, "")
	flag.Float64Var(&param.Gamma, "g", 0, "")
	flag.Float64Var(&param.C, "r", 0, "")
	flag.Float64Var(&param.C, "c", 1, "")
	flag.Float64Var(&param.Nu, "n", 0.5, "")
	flag.Float64Var(&param.P, "p", 0.1, "")
	flag.IntVar(&param.CacheSize, "m", 100, "")
	flag.Float64Var(&param.Eps, "e", 0.001, "")
	flag.Var(&weightTypeFlag, "w", "")
	flag.IntVar(&nrFold, "v", 0, "")
	flag.Var(&probabilityTypeFlag, "b", "")
	flag.BoolVar(&param.QuietMode, "q", false, "")
	flag.IntVar(&param.NumCPU, "N", -1, "")

	flag.Usage = usage
	flag.Parse()

	switch {
	case len(flag.Args()) < 1:
		usage()
		os.Exit(1)
	case len(flag.Args()) == 1:
		trainFile = flag.Arg(0)
		modelFile = getModelFileName(trainFile)
	default:
		trainFile = flag.Arg(0)
		modelFile = flag.Arg(1)
	}

	if param.QuietMode {
		outFP = ioutil.Discard
	}

	return // crossValidation, trainFile, modelFile
}

func getModelFileName(file string) string {
	var modelFile []string
	modelFile = append(modelFile, file)
	modelFile = append(modelFile, ".model")
	return strings.Join(modelFile, "")
}
