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
** Description: Input/output routines for the Support Vector Machine model
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

// Dump saves the model parameters into a text file
func (model *Model) Dump(file string) error {
	f, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	var output []string

	output = append(output, fmt.Sprintf("svm_type %s\n", svm_type_string[model.param.SvmType]))

	output = append(output, fmt.Sprintf("kernel_type %s\n", kernel_type_string[model.param.KernelType]))

	if model.param.KernelType == POLY {
		output = append(output, fmt.Sprintf("degree %d\n", model.param.Degree))
	}

	if model.param.KernelType == POLY || model.param.KernelType == RBF || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("gamma %.6g\n", model.param.Gamma))
	}

	if model.param.KernelType == POLY || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("coef0 %.6g\n", model.param.Coef0))
	}

	output = append(output, fmt.Sprintf("nr_class %d\n", model.nrClass))

	output = append(output, fmt.Sprintf("total_sv %d\n", model.l))

	if len(model.rho) > 0 {
		output = append(output, "rho")
		for i := range model.rho {
			output = append(output, fmt.Sprintf(" %.6g", model.rho[i]))
		}
		output = append(output, "\n")
	}

	if len(model.label) > 0 {
		output = append(output, "label")
		for i := range model.label {
			output = append(output, fmt.Sprintf(" %d", model.label[i]))
		}
		output = append(output, "\n")
	}
	if len(model.probA) > 0 {
		output = append(output, "probA")
		for i := range model.probA {
			output = append(output, fmt.Sprintf(" %.8g", model.probA[i]))
		}
		output = append(output, "\n")
	}

	if len(model.probB) > 0 {
		output = append(output, "probB")
		for i := range model.probB {
			output = append(output, fmt.Sprintf(" %.8g", model.probB[i]))
		}
		output = append(output, "\n")
	}

	if len(model.nSV) > 0 {
		output = append(output, "nr_sv")
		for i := range model.nSV {
			output = append(output, fmt.Sprintf(" %d", model.nSV[i]))
		}
		output = append(output, "\n")
	}

	output = append(output, "SV\n")
	if len(model.svCoef) == model.l {
		for i := 0; i < model.l; i++ {
			if len(model.svCoef[i]) == model.nrClass-1 {
				for j := 0; j < model.nrClass-1; j++ {
					output = append(output, fmt.Sprintf("%.16g ", model.svCoef[i][j]))
				}
				i_idx := model.sV[i]
				if model.param.KernelType == PRECOMPUTED {
					output = append(output, fmt.Sprintf("0:%d ", model.svSpace[i_idx]))
				} else {
					for model.svSpace[i_idx].index != -1 {
						index := model.svSpace[i_idx].index
						value := model.svSpace[i_idx].value
						output = append(output, fmt.Sprintf("%d:%.8g ", index, value))
						i_idx++
					}
					output = append(output, "\n")
				}
			}
		}
	}
	f.WriteString(strings.Join(output, ""))

	return nil
}

func (model *Model) readHeader(reader *bufio.Reader) error {

	for {
		var i int = 0
		var err error
		var line string

		line, err = readline(reader)
		if err != nil { // We should not encounter an EOF.  If we do, it is an error.
			return err
		}

		tokens := strings.Fields(line)

		switch tokens[0] {
		case "svm_type":

			for i = 0; i < len(svm_type_string); i++ {
				if svm_type_string[i] == tokens[1] {
					model.param.SvmType = i
					break
				}
			}

			if i == len(svm_type_string) {
				return fmt.Errorf("fail to parse svm model %s\n", tokens[1])
			}

		case "kernel_type":

			for i = 0; i < len(kernel_type_string); i++ {
				if kernel_type_string[i] == tokens[1] {
					model.param.KernelType = i
					break
				}
			}

			if i == len(kernel_type_string) {
				return fmt.Errorf("fail to parse kernel type %s\n", tokens[1])
			}

		case "degree":

			if model.param.Degree, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "gamma":

			if model.param.Gamma, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return err
			}

		case "coef0":

			if model.param.Coef0, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return err
			}

		case "nr_class":

			if model.nrClass, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "total_sv":

			if model.l, err = strconv.Atoi(tokens[1]); err != nil {
				return err
			}

		case "rho":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of rhos %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.rho = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.rho[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "label":

			if model.nrClass != len(tokens)-1 {
				return fmt.Errorf("Number of labels %d does not appear in the file\n", model.nrClass)
			}

			model.label = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.label[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return err
				}
			}

		case "probA":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of probA %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probA = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probA[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "probB":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return fmt.Errorf("Number of probB %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probB = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probB[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return err
				}
			}

		case "nr_sv":

			if model.nrClass != len(tokens)-1 {
				return fmt.Errorf("Number of nSV %d does not appear in the file %v\n", model.nrClass, tokens)
			}

			model.nSV = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.nSV[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return err
				}
			}

		case "SV":
			return nil // done reading the header!
		default:
			return fmt.Errorf("unknown text in model file: [%s]\n", tokens[0])

		}
	}

	return fmt.Errorf("Fail to completely read header")
}

// ReadModel reads the model parameters from a text file
func (model *Model) ReadModel(file string) error {
	f, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("Fail to open file %s\n", file)
	}

	defer f.Close() // close f on method return

	reader := bufio.NewReader(f)

	if err := model.readHeader(reader); err != nil {
		return err
	}

	model.svCoef = make([][]float64, model.l)
	for i := 0; i < model.l; i++ {
		model.svCoef[i] = make([]float64, model.nrClass-1)
	}

	model.sV = make([]int, model.l)
	var sVInd = 0
	for {
		line, err := readline(reader) // read a line
		if err != nil {
			break
		}

		tokens := strings.Fields(line) // get all the word tokens (seperated by white spaces)
		if len(tokens) < 2 {           // there should be at least 2 fields -- label + SV
			continue
		}
		if sVInd >= model.l {
			return fmt.Errorf("Error in reading support vectors.  sVInd=%d and l=%d\n", sVInd, model.l)
		}

		model.sV[sVInd] = len(model.svSpace) // starting index into svSpace for this SV

		var k = 0
		for _, token := range tokens {
			if k < model.nrClass-1 {
				model.svCoef[sVInd][k], err = strconv.ParseFloat(token, 64)
				k++
			} else {
				node := strings.Split(token, ":")
				if len(node) < 2 {
					return fmt.Errorf("Fail to parse svSpace from token %v\n", token)
				}
				var index int
				var value float64
				if index, err = strconv.Atoi(node[0]); err != nil {
					return fmt.Errorf("Fail to parse index from token %v\n", token)
				}
				if value, err = strconv.ParseFloat(node[1], 64); err != nil {
					return fmt.Errorf("Fail to parse value from token %v\n", token)
				}
				model.svSpace = append(model.svSpace, snode{index: index, value: value})
			}
		}
		model.svSpace = append(model.svSpace, snode{index: -1})
		sVInd++
	}

	return nil
}

// DumpToString saves the model parameters into a string
func (model *Model) DumpToString() (string, error) {
	var output []string

	output = append(output, fmt.Sprintf("svm_type %s\n", svm_type_string[model.param.SvmType]))

	output = append(output, fmt.Sprintf("kernel_type %s\n", kernel_type_string[model.param.KernelType]))

	if model.param.KernelType == POLY {
		output = append(output, fmt.Sprintf("degree %d\n", model.param.Degree))
	}

	if model.param.KernelType == POLY || model.param.KernelType == RBF || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("gamma %.6g\n", model.param.Gamma))
	}

	if model.param.KernelType == POLY || model.param.KernelType == SIGMOID {
		output = append(output, fmt.Sprintf("coef0 %.6g\n", model.param.Coef0))
	}

	output = append(output, fmt.Sprintf("nr_class %d\n", model.nrClass))

	output = append(output, fmt.Sprintf("total_sv %d\n", model.l))

	if len(model.rho) > 0 {
		output = append(output, "rho")
		for i := range model.rho {
			output = append(output, fmt.Sprintf(" %.6g", model.rho[i]))
		}
		output = append(output, "\n")
	}

	if len(model.label) > 0 {
		output = append(output, "label")
		for i := range model.label {
			output = append(output, fmt.Sprintf(" %d", model.label[i]))
		}
		output = append(output, "\n")
	}
	if len(model.probA) > 0 {
		output = append(output, "probA")
		for i := range model.probA {
			output = append(output, fmt.Sprintf(" %.8g", model.probA[i]))
		}
		output = append(output, "\n")
	}

	if len(model.probB) > 0 {
		output = append(output, "probB")
		for i := range model.probB {
			output = append(output, fmt.Sprintf(" %.8g", model.probB[i]))
		}
		output = append(output, "\n")
	}

	if len(model.nSV) > 0 {
		output = append(output, "nr_sv")
		for i := range model.nSV {
			output = append(output, fmt.Sprintf(" %d", model.nSV[i]))
		}
		output = append(output, "\n")
	}

	output = append(output, "SV\n")
	if len(model.svCoef) == model.l {
		for i := 0; i < model.l; i++ {
			if len(model.svCoef[i]) == model.nrClass-1 {
				for j := 0; j < model.nrClass-1; j++ {
					output = append(output, fmt.Sprintf("%.16g ", model.svCoef[i][j]))
				}
				i_idx := model.sV[i]
				if model.param.KernelType == PRECOMPUTED {
					output = append(output, fmt.Sprintf("0:%d ", model.svSpace[i_idx]))
				} else {
					for model.svSpace[i_idx].index != -1 {
						index := model.svSpace[i_idx].index
						value := model.svSpace[i_idx].value
						output = append(output, fmt.Sprintf("%d:%.8g ", index, value))
						i_idx++
					}
					output = append(output, "\n")
				}
			}
		}
	}
	return strings.Join(output, ""), nil
}

func (model *Model) readHeaderString(str []string) (int, error) {

	var lineNumber int

	for l := range str {
		var i = 0
		var err error

		tokens := strings.Fields(str[l])
		lineNumber = l

		switch tokens[0] {
		case "svm_type":

			for i = 0; i < len(svm_type_string); i++ {
				if svm_type_string[i] == tokens[1] {
					model.param.SvmType = i
					break
				}
			}

			if i == len(svm_type_string) {
				return lineNumber, fmt.Errorf("fail to parse svm model %s\n", tokens[1])
			}

		case "kernel_type":

			for i = 0; i < len(kernel_type_string); i++ {
				if kernel_type_string[i] == tokens[1] {
					model.param.KernelType = i
					break
				}
			}

			if i == len(kernel_type_string) {
				return lineNumber, fmt.Errorf("fail to parse kernel type %s\n", tokens[1])
			}

		case "degree":

			if model.param.Degree, err = strconv.Atoi(tokens[1]); err != nil {
				return lineNumber, err
			}

		case "gamma":

			if model.param.Gamma, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return lineNumber, err
			}

		case "coef0":

			if model.param.Coef0, err = strconv.ParseFloat(tokens[1], 64); err != nil {
				return lineNumber, err
			}

		case "nr_class":

			if model.nrClass, err = strconv.Atoi(tokens[1]); err != nil {
				return lineNumber, err
			}

		case "total_sv":

			if model.l, err = strconv.Atoi(tokens[1]); err != nil {
				return lineNumber, err
			}

		case "rho":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return lineNumber, fmt.Errorf("Number of rhos %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.rho = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.rho[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return lineNumber, err
				}
			}

		case "label":

			if model.nrClass != len(tokens)-1 {
				return lineNumber, fmt.Errorf("Number of labels %d does not appear in the file\n", model.nrClass)
			}

			model.label = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.label[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return lineNumber, err
				}
			}

		case "probA":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return lineNumber, fmt.Errorf("Number of probA %d does not mactch the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probA = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probA[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return lineNumber, err
				}
			}

		case "probB":

			total_class_comparisons := model.nrClass * (model.nrClass - 1) / 2
			if total_class_comparisons != len(tokens)-1 {
				return lineNumber, fmt.Errorf("Number of probB %d does not match the required number %d\n", len(tokens)-1, total_class_comparisons)
			}

			model.probB = make([]float64, total_class_comparisons)
			for i = 0; i < total_class_comparisons; i++ {
				if model.probB[i], err = strconv.ParseFloat(tokens[i+1], 64); err != nil {
					return lineNumber, err
				}
			}

		case "nr_sv":

			if model.nrClass != len(tokens)-1 {
				return lineNumber, fmt.Errorf("Number of nSV %d does not appear in the file %v\n", model.nrClass, tokens)
			}

			model.nSV = make([]int, model.nrClass)
			for i = 0; i < model.nrClass; i++ {
				if model.nSV[i], err = strconv.Atoi(tokens[i+1]); err != nil {
					return lineNumber, err
				}
			}

		case "SV":
			return lineNumber, nil // done reading the header!
		default:
			return lineNumber, fmt.Errorf("unknown str in model file: [%s]\n", tokens[0])

		}
	}

	return lineNumber, fmt.Errorf("fail to completely read header")
}

// ReadModelFromString reads the model parameters from a data string
func (model *Model) ReadModelFromString(str string) error {

	text := strings.Split(str, "\n")

	lineNumber, err := model.readHeaderString(text)
	if err != nil {
		return err
	}

	model.svCoef = make([][]float64, model.l)
	for i := 0; i < model.l; i++ {
		model.svCoef[i] = make([]float64, model.nrClass-1)
	}

	model.sV = make([]int, model.l)
	var sVInd = 0
	for i := lineNumber + 1; i < len(text); i++ {
		line := text[i]

		tokens := strings.Fields(line) // get all the word tokens (seperated by white spaces)
		if len(tokens) < 2 {           // there should be at least 2 fields -- label + SV
			continue
		}
		if sVInd >= model.l {
			return fmt.Errorf("Error in reading support vectors.  sVInd=%d and l=%d\n", sVInd, model.l)
		}

		model.sV[sVInd] = len(model.svSpace) // starting index into svSpace for this SV

		var k = 0
		for _, token := range tokens {
			if k < model.nrClass-1 {
				model.svCoef[sVInd][k], err = strconv.ParseFloat(token, 64)
				k++
			} else {
				node := strings.Split(token, ":")
				if len(node) < 2 {
					return fmt.Errorf("Fail to parse svSpace from token %v\n", token)
				}
				var index int
				var value float64
				if index, err = strconv.Atoi(node[0]); err != nil {
					return fmt.Errorf("Fail to parse index from token %v\n", token)
				}
				if value, err = strconv.ParseFloat(node[1], 64); err != nil {
					return fmt.Errorf("Fail to parse value from token %v\n", token)
				}
				model.svSpace = append(model.svSpace, snode{index: index, value: value})
			}
		}
		model.svSpace = append(model.svSpace, snode{index: -1})
		sVInd++
	}

	return nil
}
