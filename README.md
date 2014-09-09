# libsvm-go


Full port of LIBSVM in the Go programming language.
This port has no external package dependencies, and uses only the native standard library.

## Installation

    go get github.com/ewalker544/libsvm-go
    make

## API Example

    import "github.com/ewalker544/libsvm-go"
    
    param := libsvm.NewParameter()      // Create a parameter struct with default values
    param.SvmType = libSvm.EPSILON_SVR
    param.KernelType = libSvm.POLY
    
    model := libSvm.NewModel(param)     // Create a model from the parameter
    
    // Create a problem specification from the training data (in trainFile) and parameter
    problem, err := libSvm.NewProblem("log1p.E2006.train", param)
    
    model.Train(problem)                // Train the model from the problem specification
    
    model.Dump(modelFile)               // Dump the model into a user-specified file
    
    
    
    
    
    
    
    
