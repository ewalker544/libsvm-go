# libsvm-go


This is a full port of LIBSVM in the Go programming language.  LIBSVM is a suite of tools and library for support vector classification, regression, and distribution estimation.  This port implements the libsvm library in the form of a Go package called <code>libSvm</code>.  It also implements the <code>svm-train</code> and <code>svm-predict</code> command line tools.

This port has no external package dependencies, and uses only the native standard library.

## Installation

    go get github.com/ewalker544/libsvm-go
    make

## Compatibility Notes 

I have tried to make the Go implementation of <code>svm-train</code> and <code>svm-predict</code> plug-in compatibile with the original LIBSVM distribution.  This is to allow you to use the other tools available in the original distribution, like <code>easy.py</code> and <code>grid.py</code>.

<code>svm-predict</code> should be 100% plug-in compatibile.  However, <code>svm-train</code> is plug-in compatible with one exception.  The exception is the parameter weight flag used in the command.  In this this implementation, the flag is

    -w i,weight : set the parameter C of class i to weight*C, for C-SVC (default 1)

For full documentation of the <code>svm-train</code> and <code>svm-predict</code> commands, please refer to the original [LIBSVM][1] web site.

## API Example

### Training
    import "github.com/ewalker544/libsvm-go"
    
    param := libsvm.NewParameter()      // Create a parameter object with default values
    param.KernelType = libSvm.LINEAR    // Use the linear (dot product) kernel
    
    model := libSvm.NewModel(param)     // Create a model object from the parameter attributes
    
    // Create a problem specification from the training data and parameter attributes
    problem, err := libSvm.NewProblem("a9a.train", param) 
    
    model.Train(problem)                // Train the model from the problem specification
    
    model.Dump("a9a.model")     // Dump the model into a user-specified file
    
    
### Predicting
    import "github.com/ewalker544/libsvm-go"
    
    param := libSvm.NewParameter()      // Create a parameter object with default values
    
    model := libSvm.NewModel(param)     // Create a model object from the parameter attributes
    
    model.ReadModel("a9a.model")   // Populate the model from the model file generated from training
    
    x := make(map[int]float64)
    // Populate x with the test vector
    
    predictLabel := model.Predict(x)    // Predicts a float64 label given the test vector 
    
    
    

[1]: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    
    
