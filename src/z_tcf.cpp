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

// Implementation of TCF class. See include/z_tcf.hpp for
// more details about the class.

#include "z_tcf.hpp"

template <class T>
void TCF<T>::Correlate(const arma::mat& vec1, const arma::mat& vec2) {
  for (int i1=0; i1<num_corr_; i1++) {
    int i1corr = i1*corr_int_;
    correlation_function_(0) += 2.0*dot(vec1.row(i1corr),vec2.row(i1corr));
    for (int i2=1; i2<corr_; i2++) {
      int i1corri2 = i1corr + i2;
      correlation_function_(i2) += dot(vec1.row(i1corr),vec2.row(i1corri2));
      correlation_function_(i2) += dot(vec1.row(i1corri2),vec2.row(i1corr));
    }
  }
}

template <class T>
void TCF<T>::Correlate(const arma::mat& vec1, const arma::mat& vec2,
                       const int mod) {
  int corrmod = mod + corr_;
  for (int i2=mod; i2<corrmod; i2++) {
    int i2min = i2-mod;
    int i2mod = i2%corr_;
    correlation_function_(i2min) += dot(vec1.row(mod),vec2.row(i2mod));
    correlation_function_(i2min) += dot(vec1.row(i2mod),vec2.row(mod));
  }
}

template <class T>
void TCF<T>::CorrelateOneDirection(const arma::mat& vec1,
                                   const arma::mat& vec2) {
  for (int i1=0; i1<num_corr_; i1++) {
    int i1corr = i1*corr_int_;
    for (int i2=0; i2<corr_; i2++) {
      int i1corri2 = i1corr + i2;
      correlation_function_(i2) += dot(vec1.row(i1corr),vec2.row(i1corri2));
    }
  }
}

template <class T>
void TCF<T>::CorrelateOneDirection(const arma::mat& vec1, const arma::mat& vec2,
                                   const int mod) {
  int corrmod = mod + corr_;
  for (int i2=mod; i2<corrmod; i2++) {
    int i2min = i2-mod;
    int i2mod = i2%corr_;
    correlation_function_(i2min) += dot(vec1.row(mod),vec2.row(i2mod));
  }
}

template <class T>
void TCF<T>::Correlate(const arma::rowvec& vec1, const arma::rowvec& vec2) {
  for (int i1=0; i1<num_corr_; i1++) {
    int i1corr = i1*corr_int_;
    correlation_function_(0) += 2.0*vec1[i1corr]*vec2[i1corr];
    for (int i2=1; i2<corr_; i2++) {
      int i1corri2 = i1corr + i2;
      correlation_function_(i2) += vec1[i1corr]*vec2[i1corri2];
      correlation_function_(i2) += vec1[i1corri2]*vec2[i1corr];;
    }
  }
}

template <class T>
void TCF<T>::Correlate(const arma::rowvec& vec1, const arma::rowvec& vec2,
  const int mod) {
  int corrmod = mod + corr_;
  for (int i2=mod; i2<corrmod; i2++) {
    int i2min = i2-mod;
    int i2mod = i2%corr_;
    correlation_function_(i2min) += vec1[mod]*vec2[i2mod];
    correlation_function_(i2min) += vec1[i2mod]*vec2[mod];
  }
}

template <class T>
void TCF<T>::CorrelateOneDirection(const arma::rowvec& vec1,
                                   const arma::rowvec& vec2) {
  for (int i1=0; i1<num_corr_; i1++) {
    int i1corr = i1*corr_int_;
    for (int i2=0; i2<corr_; i2++) {
      int i1corri2 = i1corr + i2;
      correlation_function_(i2) += vec1[i1corr]*vec2[i1corri2];
    }
  }
}

template <class T>
void TCF<T>::CorrelateOneDirection(const arma::rowvec& vec1,
                                   const arma::rowvec& vec2, const int mod) {
  int corrmod = mod + corr_;
  for (int i2=mod; i2<corrmod; i2++) {
    int i2min = i2-mod;
    int i2mod = i2%corr_;
    correlation_function_(i2min) += vec1[mod]*vec2[i2mod];
  }
}

// Added to avoid linker error.
template class TCF<double>;
template class TCF<arma::cx_double>;
