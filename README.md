# libsvm-go: Support Vector Machine


This is a full port of LIBSVM in the Go programming language.  [LIBSVM][1] is a suite of tools and an API library for support vector classification, regression, and distribution estimation.  This port implements the libsvm library in the form of a Go package called <code>libSvm</code>.  It also implements the <code>svm-train</code> and <code>svm-predict</code> command line tools.

This port has no external package dependencies, and uses only the native standard library.

## Installation

    go get github.com/ewalker544/libsvm-go
    make

## Compatibility Notes 

I have tried to make the Go implementation of <code>svm-train</code> and <code>svm-predict</code> plug-in compatibile with the original LIBSVM 3.18 distribution.  This is to allow you to use the other tools available in the original distribution, like <code>easy.py</code> and <code>grid.py</code>.

<code>svm-predict</code> should be 100% plug-in compatibile.  However, <code>svm-train</code> is plug-in compatible with one exception.  The exception is the parameter weight flag used in the command.  In this implementation, the flag is

    -w i,weight : set the parameter C of class i to weight*C, for C-SVC (default 1)

For full documentation of the <code>svm-train</code> and <code>svm-predict</code> commands, please refer to the original [LIBSVM][1] web site.

## API Example

### Training
```go
import "github.com/ewalker544/libsvm-go"
    
param := libSvm.NewParameter()      // Create a parameter object with default values
param.KernelType = libSvm.POLY      // Use the polynomial kernel
    
model := libSvm.NewModel(param)     // Create a model object from the parameter attributes
    
// Create a problem specification from the training data and parameter attributes
problem, err := libSvm.NewProblem("a9a.train", param) 
    
model.Train(problem)                // Train the model from the problem specification
    
model.Dump("a9a.model")             // Dump the model into a user-specified file
```    

### Manual create problem with train data
```go
// As example if you want to train your model
// by classification you can do next
import (
    ...
    libSvm "github.com/killer-djon/libsvm-go"
)

param := libSvm.NewParameter()
param.SvmType = libSvm.NU_SVC
param.Eps = 1e-4
param.Probability = true

// labels []float64 // slice of all labels 
// values []map[int]float64 // slice is equal to nr labels 
// and contains mapping values with index:float64_value by this index
// as example:
// labels := []float64{1, 4, 6, 8} - slice of random labels
// nrLabels in this case is 4
// values := []map[int]float64{} by eny values

// you can create train data with labels and its values
trainData := libSvm.NewTrainData(labels, values)
// Create new clean problem
prob := libSvm.InitProblem()
// After this you can fit with train data your problem
err := prob.FitProblem(trainData, param)

// After this you can start training model
model := libSvm.NewModel(param)
model.Train(prob)
model.Dump("train.model")

```
    
### Predicting
```go
import "github.com/ewalker544/libsvm-go"
    
// Create a model object from the model file generated from training
model := libSvm.NewModelFromFile("a9a.model")  
    
x := make(map[int]float64)
// Populate x with the test vector
    
predictLabel := model.Predict(x)    // Predicts a float64 label given the test vector 
```   
    
    

[1]: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    
    
