package libSvm

import (
	"errors"
	"math"
)

/**
Interface for all kernel functions
*/
type kernelFunction interface {
	compute(i, j int) float64
}

/**
Returns the dot product of SVs px and py
*/
func dot(px, py []snode) float64 {
	var sum float64 = 0
	var i int = 0
	var j int = 0
	for px[i].index != -1 && py[j].index != -1 {
		if px[i].index == py[j].index {
			sum = sum + px[i].value*py[j].value
			i++
			j++
		} else {
			if px[i].index > py[j].index {
				j++
			} else {
				i++
			}
		}
	}
	return sum
}

/********** LINEAR KERNEL ***************/
type linear struct {
	x      []int
	xSpace []snode
}

func (k linear) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	return dot(k.xSpace[idx_i:], k.xSpace[idx_j:])
}

func NewLinear(x []int, xSpace []snode) linear {
	return linear{x: x, xSpace: xSpace}
}

/************** RBF KERNEL ***************/
type rbf struct {
	x        []int
	xSpace   []snode
	x_square []float64
	gamma    float64
}

func (k rbf) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.x_square[i] + k.x_square[j] - 2.0*dot(k.xSpace[idx_i:], k.xSpace[idx_j:])
	return math.Exp(-k.gamma * q)
}

func NewRBF(x []int, xSpace []snode, l int, gamma float64) rbf {
	x_square := make([]float64, l)

	for i := 0; i < l; i++ {
		var idx_i int = x[i]
		x_square[i] = dot(xSpace[idx_i:], xSpace[idx_i:])
	}

	return rbf{x: x, xSpace: xSpace, x_square: x_square, gamma: gamma}
}

/***************** POLY KERNEL *************/
type poly struct {
	x      []int
	xSpace []snode
	gamma  float64
	coef0  float64
	degree int
}

func (k poly) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.gamma*dot(k.xSpace[idx_i:], k.xSpace[idx_j:]) + k.coef0
	return math.Pow(q, float64(k.degree))
}

func NewPoly(x []int, xSpace []snode, gamma, coef0 float64, degree int) poly {
	return poly{x: x, xSpace: xSpace, gamma: gamma, coef0: coef0, degree: degree}
}

/*************** SIGMOID KERNEL *************/
type sigmoid struct {
	x      []int
	xSpace []snode
	gamma  float64
	coef0  float64
}

func (k sigmoid) compute(i, j int) float64 {
	var idx_i int = k.x[i]
	var idx_j int = k.x[j]
	q := k.gamma*dot(k.xSpace[idx_i:], k.xSpace[idx_j:]) + k.coef0
	return math.Tanh(q)
}

func NewSigmoid(x []int, xSpace []snode, gamma, coef0 float64) sigmoid {
	return sigmoid{x: x, xSpace: xSpace, gamma: gamma, coef0: coef0}
}

/************** Factory ***************/
func NewKernel(prob *Problem, param *Parameter) (kernelFunction, error) {
	switch param.KernelType {
	case LINEAR:
		return NewLinear(prob.x, prob.xSpace), nil
	case POLY:
		return NewPoly(prob.x, prob.xSpace, param.Gamma, param.Coef0, param.Degree), nil
	case RBF:
		return NewRBF(prob.x, prob.xSpace, prob.l, param.Gamma), nil
	case SIGMOID:
		return NewSigmoid(prob.x, prob.xSpace, param.Gamma, param.Coef0), nil
	}
	return nil, errors.New("unsupported kernel")
}

func computeKernelValue(px, py []snode, param *Parameter) float64 {
	switch param.KernelType {
	case LINEAR:
		return dot(px, py)
	case RBF:
		q := dot(px, px) + dot(py, py) - 2*dot(px, py)
		return math.Exp(-param.Gamma * q)
	case POLY:
		q := param.Gamma*dot(px, py) + param.Coef0
		return math.Pow(q, float64(param.Degree))
	case SIGMOID:
		q := param.Gamma*dot(px, py) + param.Coef0
		return math.Tanh(q)
	case PRECOMPUTED:
		var idx_j int = int(py[0].value)
		return px[idx_j].value
	}

	return 0
}
