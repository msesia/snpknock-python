/*
  This file is part of SNPknock.

    Copyright (C) 2017 Matteo Sesia

    SNPknock is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SNPknock is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SNPknock.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef KNOCKOFFS_H
#define KNOCKOFFS_H

/*
Knockoffs for hidden Markov models
*/

#include <vector>
#include <random>

typedef std::vector< std::vector<double> > matrix;

namespace knockoffs{
  class Knockoffs {
      int weighted_choice(double U, const std::vector<double> & weights);
  };

}

#endif
