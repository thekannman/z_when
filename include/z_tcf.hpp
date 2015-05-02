//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

// This class is for dealing with time-correlation functions (TCFs) between
// scalars or vectors.

#include <armadillo>

#ifndef _Z_TCF_HPP_
#define _Z_TCF_HPP_

template <class T>
class TCF {
 public:
  TCF(int length, int interval = 1, int zeros = 0)
      : length_(length), interval_(interval), zeros_(zeros) {
    correlation_function_.zeros(length_+zeros_);
  }

  // Calculations correlation between two vector time series assuming
  // time-symmetry around t=0.
  void Correlate(const arma::mat& vec1, const arma::mat& vec2);

  // Calculations correlation between two vector time series assuming
  // time-symmetry around t=0. Allows time series to start at arbitrary index
  // and wrap around end of vector.
  void Correlate(const arma::mat& vec1, const arma::mat& vec2, const int mod);

  // Calculations correlation between two vector time series without assuming
  // time-symmetry around t=0.
  void CorrelateOneDirection(const arma::mat& vec1, const arma::mat& vec2);

  // Calculations correlation between two vector time series without assuming
  // time-symmetry around t=0. Allows time series to start at arbitrary index
  // and wrap around end of vector.
  void CorrelateOneDirection(const arma::mat& vec1, const arma::mat& vec2,
                               const int mod);

  // Wrapper for Correlate to avoid ambiguity
  inline void Correlate(const arma::subview_cube<double>& vec1,
                        const arma::subview_cube<double>& vec2) {
    Correlate(arma::rowvec(vec1), arma::rowvec(vec2));
  }

  // Wrapper for Correlate to avoid ambiguity
  inline void Correlate(const arma::subview_cube<double>& vec1,
                        const arma::subview_cube<double>& vec2, const int mod) {
    Correlate(arma::rowvec(vec1), arma::rowvec(vec2), mod);
  }

  // Wrapper for CorrelateOneDirection to avoid ambiguity
  inline void CorrelateOneDirection(const arma::subview_cube<double>& vec1,
                                    const arma::subview_cube<double>& vec2) {
    CorrelateOneDirection(arma::rowvec(vec1), arma::rowvec(vec2));
  }

  // Wrapper for CorrelateOneDirection to avoid ambiguity
  inline void CorrelateOneDirection(const arma::subview_cube<double>& vec1,
                                    const arma::subview_cube<double>& vec2,
                                    const int mod) {
    CorrelateOneDirection(arma::rowvec(vec1), arma::rowvec(vec2), mod);
  }

  // Calculations correlation between two scalar time series assuming
  // time-symmetry around t=0.
  void Correlate(const arma::rowvec& vec1, const arma::rowvec& vec2);

  // Calculations correlation between two scalar time series assuming
  // time-symmetry around t=0. Allows time series to start at arbitrary index
  // and wrap around end of vector.
  void Correlate(const arma::rowvec& vec1, const arma::rowvec& vec2,
                    const int mod);

  // Calculations correlation between two scalar time series without assuming
  // time-symmetry around t=0.
  void CorrelateOneDirection(const arma::rowvec& vec1,
                                  const arma::rowvec& vec2);

  // Calculations correlation between two scalar time series without assuming
  // time-symmetry around t=0. Allows time series to start at arbitrary index
  // and wrap around end of vector.
  void CorrelateOneDirection(const arma::rowvec& vec1,
                                  const arma::rowvec& vec2, const int mod);

  // Wrapper for CorrelateOneDirection that allows vector self-correlation.
  inline void CorrelateOneDirection(const arma::mat& vec) {
    CorrelateOneDirection(vec, vec);
  }

  // Wrapper for CorrelateOneDirection that allows vector self-correlation and
  // allows time series to start at arbitrary index and wrap around end of
  // vector.
  inline void CorrelateOneDirection(const arma::mat& vec, const int mod) {
    CorrelateOneDirection(vec, vec, mod);
  }

  // Wrapper for CorrelateOneDirection that allows scalar self-correlation.
  inline void CorrelateOneDirection(const arma::rowvec& vec) {
    CorrelateOneDirection(vec, vec);
  }

  // Wrapper for CorrelateOneDirection that allows scalar self-correlation and
  // allows time series to start at arbitrary index and wrap around end of
  // vector.
  inline void CorrelateOneDirection(const arma::rowvec& vec,
                                         const int mod) {
    CorrelateOneDirection(vec, vec, mod);
  }

  //Accessors
  int length() const { return length_; }
  inline T tcf(int i) const { return correlation_function_(i); }

 private:
  arma::Row<T> correlation_function_;
  int length_, interval_, zeros_, number_, speclength_;
  int corr_int_, corr_, num_corr_;

};
#endif
