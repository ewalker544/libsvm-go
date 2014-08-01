package main

import (
	"flag"
	"fmt"
	"github.com/ewalker544/libsvm-go"
)

func usage() {
	fmt.Printf(
		"Usage: svm-predict [options] test_file model_file [output_file]\n",
		"options:\n",
		"-b probability_estimates: whether to predict probability estimates, true or false (default false); for one-class SVM only false is supported\n",
		"-q : quiet mode (no outputs)\n")
}

func parseOptions(param *libSvm.Parameter) (testFile string, modelFile string, outputFile string) {

	flag.BoolVar(&param.Prob, "b", false, "")
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
		outputFile = nil
	default:
		testFile = flag.Arg(0)
		modelFile = flag.Arg(1)
		outputFile = flag.Arg(2)
	}

	return // testFile, modelFile, outputFile
}
