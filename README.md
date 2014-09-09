# libsvm-go


Full port of LIBSVM in the Go programming language.
This port has no external package dependencies, and uses only the native standard library.

## Installation

    go get github.com/ewalker544/libsvm-go
    make

## API Example

    import "github.com/ewalker544/libsvm-go"
    param := libsvm.NewParameter()      // Create a parameter
    model := libSvm.NewModel(param)     // Create a model from the parameter (using the default values)
    problem, err := libSvm.NewProblem(trainFile, param) // Create a problem specification described by the training file and parameter
    model.Train(problem)                // Train the model from the problem specification
    model.Dump(modelFile)               // Dump the model into a user-specified file
    
    
    
    
    
    
    
    
