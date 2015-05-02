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

// This class is used for interpolations along a 3-dimensional
// grid.

#include <armadillo>
#include "boost/multi_array.hpp"
#include "z_constants.hpp"
#include "z_vec.hpp"
#include "z_atom_group.hpp"

#ifndef _Z_GRID_HPP_
#define _Z_GRID_HPP_

enum MeshID {kLower, kUpper};

class Grid {
 public:
  Grid(const arma::rowvec& box, const std::string& filename,
       const double coarse_grain_length = 0.24, const double spacing = 0.1);

  // Calculates the particle density (rho) along the grid
  void CalcRho(const AtomGroup& atom_group);

  // Grabs mesh heights from file
  void ReadMesh();

  // Finds average height of a single mesh
  double AvgMesh();

  void FindNearestMeshPoint(const arma::rowvec& particle_position,
                            MeshID& mesh_id, int& min_x, int& min_y,
                            arma::rowvec& dx);

  void FindSurfaceNormal(const MeshID mesh_id, const int& x_point,
                         const int& y_point, arma::rowvec& surface_normal);

  inline void close_file() { return file_.close(); }

  // Accessors
  inline double spacing() const { return spacing_; }
  inline int size_x() const { return points_(0); }
  inline int size_y() const { return points_(1); }
  inline int size_z() const { return points_(2); }
  inline arma::irowvec size() const { return points_; }

  inline double upper_mesh(int x, int y, int z) const {
    return upper_mesh_(x, y, z);
  }

  inline arma::rowvec upper_mesh(int x, int y) const {
    return upper_mesh_.tube(x, y);
  }

  inline double lower_mesh(int x, int y, int z) const {
    return lower_mesh_(x, y, z);
  }

  inline arma::rowvec lower_mesh(int x, int y) const {
    return lower_mesh_.tube(x, y);
  }

  inline arma::rowvec mesh(int x, int y, const MeshID mesh_id) const {
    if (mesh_id == kUpper)
      return upper_mesh_.tube(x, y);
    else
      return lower_mesh_.tube(x, y);
  }

  inline double rho(int x, int y, int z) const {
    return rho(x, y, z);
  }

  inline arma::rowvec grid(int x, int y, int z) const {
    return VeclikeToRow(grid_[x][y][z], DIMS);
  }

  inline double grid(int x, int y, int z, int dim) const {
    return grid_[x][y][z][dim];
  }

  inline bool file_exists() const { return file_exists_; }

 private:
  // Resets the density (rho).
  inline void ZeroRho() {
      rho_ = arma::zeros<arma::cube>(points_(0),points_(1),points_(2));
  }

  inline int shift_x_down(const int x) const {
      return (x == 0) ? (points_(0) - 1) : (x - 1);
  }

  inline int shift_x_up(const int x) const {
      return (x == points_(0) - 1) ? 0 : (x + 1);
  }

  inline int shift_y_down(const int y) const {
      return (y == 0) ? (points_(1) - 1) : (y - 1);
  }

  inline int shift_y_up(const int y) const {
      return (y == points_(1) - 1) ? 0 : (y + 1);
  }

  inline int shift_z_down(const int z) const {
      return (z == 0) ? (points_(2) - 1) : (z - 1);
  }

  inline int shift_z_up(const int z) const {
      return (z == points_(2) - 1) ? 0 : (z + 1);
  }

  std::string filename_;
  double coarse_grain_length_;
  double coarse_grain_length_squared_;
  double cutoff_length_;
  double cutoff_length_squared_;
  double rho_pre_factor_;
  double rho_exp_factor_;
  double phi_rmax_;
  double spacing_;
  arma::rowvec length_;
  arma::irowvec points_;
  arma::cube rho_;
  hypercube grid_;
  // TODO(Zak): Make mesh a separate class
  arma::cube upper_mesh_;
  arma::cube lower_mesh_;
  std::fstream file_;

  bool file_exists_;

};

#endif
