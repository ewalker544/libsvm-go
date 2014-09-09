# libsvm-go


This is a full port of LIBSVM in the Go programming language.  LIBSVM is a suite of tools and library for support vector classification, regression, and distribution estimation.  This port implements the libsvm library in the form of a Go package called libSvm.  It also implements the svm-train and svm-predict command line tools.

This port has no external package dependencies, and uses only the native standard library.

## Installation

    go get github.com/ewalker544/libsvm-go
    make

## API Example

### Training
    import "github.com/ewalker544/libsvm-go"
    
    param := libsvm.NewParameter()      // Create a parameter object with default values
    param.KernelType = libSvm.LINEAR    // Use the linear (dot product) kernel
    
    model := libSvm.NewModel(param)     // Create a model object from the parameter attributes
    
    // Create a problem specification from the training data and parameter attributes
    problem, err := libSvm.NewProblem("log1p.E2006.train", param) 
    
    model.Train(problem)                // Train the model from the problem specification
    
    model.Dump("log1p.E2006.model")     // Dump the model into a user-specified file
    
    
### Predicting
    import "github.com/ewalker544/libsvm-go"
    
    param := libSvm.NewParameter()      // Create a parameter object with default values
    
    model := libSvm.NewModel(param)     // Create a model object from the parameter attributes
    
    model.ReadModel("log1p.E2006.model")   // Populate the model from the model file generating from training
    
    x := make(map[int]float64)
    // Populate x with the test vector
    
    predictLabel := model.Predict(x)    // Predicts a float64 label given the test vector 
    
    
    
    
    
