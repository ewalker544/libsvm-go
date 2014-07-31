package libsvm

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
	SvmType    int
	KernelType int
	Degree     int
	Gamma      float64
	Coef0      float64

	Eps         float64 // stopping criteria
	C           float64 // penality
	NrWeight    int
	WeightLabel []int
	Weight      []float64
	Nu          float64
	P           float64
	Probability bool
}

func NewParameter() *Parameter {
	return &Parameter{SvmType: C_SVC, KernelType: RBF, Degree: 3, Gamma: 0, Coef0: 0, Nu: 0.5, C: 1, Eps: 1e-3, P: 0.1,
		NrWeight: 0, Probability: false}
}
