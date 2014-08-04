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
** Description: Calculate the mean square error, and the square correlation coefficient
** @author: Ed Walker
 */

package libSvm

type SquareErrorComputer struct {
	err   float64
	sump  float64
	sumt  float64
	sumpp float64
	sumtt float64
	sumpt float64
	total int
}

func (s *SquareErrorComputer) Sum(predict, target float64) {
	s.err += (predict - target) * (predict - target)
	s.sump += predict
	s.sumt += target
	s.sumpp += predict * predict
	s.sumtt += target * target
	s.sumpt += predict * target
	s.total++
}

func (s *SquareErrorComputer) MeanSquareError() (err float64) {
	err = s.err / float64(s.total)
	return
}

func (s *SquareErrorComputer) SquareCorrelationCoeff() (coeff float64) {
	coeff = ((float64(s.total)*s.sumpt - s.sump*s.sumt) * (float64(s.total)*s.sumpt - s.sump*s.sumt)) /
		((float64(s.total)*s.sumpp - s.sump*s.sump) * (float64(s.total)*s.sumtt - s.sumt*s.sumt))
	return
}

func NewSquareErrorComputer() SquareErrorComputer {
	return SquareErrorComputer{err: 0, sump: 0, sumt: 0, sumpp: 0, sumtt: 0, sumpt: 0, total: 0}
}
