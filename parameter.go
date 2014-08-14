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
** Description: Describes the parameters of the Supper Vector Machine solver
** @author: Ed Walker
 */
package libSvm

const LibSvmGoVersion = 0.318

const (
	C_SVC       = iota
	NU_SVC      = iota
	ONE_CLASS   = iota
	EPSILON_SVR = iota
	NU_SVR      = iota
)

const (
	LINEAR      = iota
	POLY        = iota
	RBF         = iota
	SIGMOID     = iota
	PRECOMPUTED = iota
)

var svm_type_string = []string{"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"}
var kernel_type_string = []string{"linear", "polynomial", "rbf", "sigmoid", "precomputed"}

type Parameter struct {
	SvmType    int     // Support vector type
	KernelType int     // Kernel type
	Degree     int     // Degree used in polynomial kernel
	Gamma      float64 // Gamma used in rbf, polynomial, and sigmoid kernel
	Coef0      float64 // Coef0 used in polynomial and sigmoid kernel

	Eps         float64 // stopping criteria
	C           float64 // penality
	NrWeight    int
	WeightLabel []int
	Weight      []float64
	Nu          float64
	P           float64
	Probability bool // Should probability estimation be performed?
	CacheSize   int  // Size of Q matrix cache
	QuietMode   bool // quiet mode
	NumCPU      int  // Number of CPUs to use
}

func NewParameter() *Parameter {
	return &Parameter{SvmType: C_SVC, KernelType: RBF, Degree: 3, Gamma: 0, Coef0: 0, Nu: 0.5, C: 1, Eps: 1e-3, P: 0.1,
		NrWeight: 0, Probability: false, CacheSize: 100, QuietMode: false, NumCPU: -1}
}
