/**
 * Computes the mean square error, and the square correlation coefficient
 */
package libSvm

type SquareError struct {
	err     float64
	sump    float64
	sumt    float64
	sumpp   float64
	sumtt   float64
	sumpt   float64
	correct int
	total   int
}

func (s *SquareError) Sum(predict, target float64) {
	s.err += (predict - target) * (predict - target)
	s.sump += predict
	s.sumt += target
	s.sumpp += predict * predict
	s.sumtt += target * target
	s.sumpt += predict * target
	s.total++
}

func (s *SquareError) MeanSquareError() (err float64) {
	err = s.err / float64(s.total)
	return
}

func (s *SquareError) SquareCorrelationCoeff() (coeff float64) {
	coeff = ((float64(s.total)*s.sumpt - s.sump*s.sumt) * (float64(s.total)*s.sumpt - s.sump*s.sumt)) /
		((float64(s.total)*s.sumpp - s.sump*s.sump) * (float64(s.total)*s.sumtt - s.sumt*s.sumt))
	return
}
