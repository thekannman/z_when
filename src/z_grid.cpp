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

// Implementation of Grid class. See include/z_grid.hpp for
// more details about the class.

#include "z_grid.hpp"

Grid::Grid(const arma::rowvec& box, const std::string& filename,
           const double coarse_grain_length, const double spacing)
    : filename_(filename), coarse_grain_length_(coarse_grain_length),
      spacing_(spacing) {
  file_.open(filename_.c_str(),  std::fstream::in);
  if (!file_.is_open())
    file_.open(filename_.c_str(),  std::fstream::out);
  else
    file_exists_ = true;
  length_ = box;
  coarse_grain_length_squared_ = coarse_grain_length_*coarse_grain_length_;
  cutoff_length_ = 3.0*coarse_grain_length;
  cutoff_length_squared_ = cutoff_length_*cutoff_length_;

  rho_pre_factor_ = 1.0/(2.0*M_PI*coarse_grain_length_squared_);
  rho_pre_factor_ = sqrt(rho_pre_factor_*rho_pre_factor_*rho_pre_factor_);
  rho_exp_factor_ = -1.0/2.0/coarse_grain_length_squared_;
  phi_rmax_ = rho_pre_factor_*exp(-9.0/2.0);
  points_ = arma::zeros<arma::irowvec>(DIMS);
  for (int i = 0; i < DIMS; i++)
    points_(i) = length_(i)/spacing;
  upper_mesh_ = arma::zeros<arma::cube>(points_(0),points_(1),DIMS);
  for (int  i_x = 0; i_x < points_(0); i_x++) {
    for (int  i_y = 0; i_y < points_(0); i_y++) {
      upper_mesh_(i_x,i_y,0) = spacing_*(i_x+0.5);
      upper_mesh_(i_x,i_y,1) = spacing_*(i_y+0.5);
    }
  }
  lower_mesh_ = upper_mesh_;
  grid_.resize(boost::extents[points_(0)][points_(1)][points_(2)][DIMS]);
  for (int  i_x = 0; i_x < points_(0); i_x++) {
    for (int  i_y = 0; i_y < points_(1); i_y++) {
      for (int  i_z = 0; i_z < points_(2); i_z++) {
        grid_[i_x][i_y][i_z][0] = spacing_*(i_x+0.5);
        grid_[i_x][i_y][i_z][1] = spacing_*(i_y+0.5);
        grid_[i_x][i_y][i_z][2] = spacing_*(i_z+0.5);
      }
    }
  }
}

void Grid::ReadMesh() {
  for (int  i_x = 0; i_x < points_(0); i_x++) {
    for (int  i_y = 0; i_y < points_(1); i_y++) {
      file_ >> upper_mesh_(i_x,i_y,2);
      file_ >> lower_mesh_(i_x,i_y,2);
    }
  }
}

double Grid::AvgMesh() {
  // Only upper for now
  double avg = 0;
  for (int  i_x = 0; i_x < points_(0); i_x++) {
    for (int  i_y = 0; i_y < points_(1); i_y++) {
      avg += upper_mesh(i_x,i_y,2);
    }
  }
  avg /= (points_(0)*points_(1));
  return avg;
}

void Grid::FindNearestMeshPoint(const arma::rowvec& particle_position,
                                MeshID& mesh_id, int& min_x, int& min_y,
                                arma::rowvec& dx) {
  double rmin2 = std::numeric_limits<double>::max();
  for (int i_x = 0; i_x < points_(0); ++i_x) {
    for (int i_y = 0; i_y < points_(1); ++i_y) {
      FindDxNoShift(dx, particle_position, upper_mesh(i_x,i_y), length_);
      double r2 = arma::dot(dx,dx);
      if (r2<rmin2) {
        mesh_id = kUpper;
        min_x = i_x;
        min_y = i_y;
        rmin2 = r2;
      }
      FindDxNoShift(dx, particle_position, lower_mesh(i_x,i_y), length_);
      r2 = arma::dot(dx,dx);
      if (r2<rmin2) {
        mesh_id = kLower;
        min_x = i_x;
        min_y = i_y;
        rmin2 = r2;
      }
    }
  }
  FindDxNoShift(dx, particle_position, mesh(min_x, min_y, mesh_id), length_);
}

void Grid::FindSurfaceNormal(const MeshID mesh_id, const int& x_point,
                             const int& y_point, arma::rowvec& surface_normal) {
  arma::rowvec dx, dx2;
  FindDxNoShift(dx, mesh(shift_x_up(x_point),y_point, mesh_id),
                mesh(shift_x_down(x_point),y_point, mesh_id), length_);
  FindDxNoShift(dx2, mesh(x_point,shift_y_up(y_point), mesh_id),
                mesh(x_point,shift_y_down(y_point), mesh_id), length_);
  if (mesh_id == kUpper)
    surface_normal = arma::cross(dx,dx2);
  else
    surface_normal = arma::cross(dx2,dx);
  surface_normal = arma::normalise(surface_normal);
}

void Grid::CalcRho(const AtomGroup& atom_group) {
  ZeroRho();
  int grid_cutoff = cutoff_length_/spacing_ + 1;
  arma::rowvec dx;
  for (int i_atom = 0; i_atom < atom_group.size(); ++i_atom) {
    arma::irowvec closest_grid_point(DIMS);
    closest_grid_point = arma::conv_to<arma::irowvec>::from(
        floor(atom_group.position(i_atom)/spacing_) + points_);
    arma::irowvec min_grid_point = closest_grid_point - grid_cutoff;
    arma::irowvec max_grid_point = closest_grid_point + grid_cutoff;
    for (int i_x = min_grid_point(0); i_x < max_grid_point(0); ++i_x) {
      int i_x_shift = i_x % points_(0);
      for (int i_y = min_grid_point(1); i_y < max_grid_point(1); ++i_y) {
        int i_y_shift = i_y % points_(1);
        for (int i_z = min_grid_point(2); i_z < max_grid_point(2); ++i_z) {
          int i_z_shift = i_z % points_(2);
          FindDxNoShift(dx, grid(i_x_shift, i_y_shift, i_z_shift),
                         atom_group.position(i_atom), length_);
          double r2 = dot(dx,dx);
          if (r2 > cutoff_length_squared_) continue;
          rho_(i_x_shift,i_y_shift,i_z_shift) +=
              rho_pre_factor_*exp(r2*rho_exp_factor_) - phi_rmax_;
        }
      }
    }
  }

  for (int i_x = 0; i_x < points_(0); ++i_x) {
    for (int i_y = 0; i_y < points_(1); ++i_y) {
      bool FOUND_UPPER = false, FOUND_LOWER= false;
      for (int i_z = 0; i_z < points_(2); ++i_z) {
        double rhoCutoff = 16.0; // Typically set to half the bulk density.
        int i_z_shift = (i_z == 0) ? (points_(2)-1) : (i_z-1);
        if (!FOUND_UPPER && rho_(i_x,i_y,i_z) < rhoCutoff &&
            rho_(i_x,i_y,i_z_shift) > rhoCutoff) {
          upper_mesh_(i_x,i_y,2) = grid_[i_x][i_y][i_z][2] -
              (rhoCutoff-rho_(i_x,i_y,i_z))/
              (rho_(i_x,i_y,i_z_shift) - rho_(i_x,i_y,i_z))*
              spacing_;
          if (i_z == 0)
            lower_mesh_(i_x,i_y,2) += length_(2);
          FOUND_UPPER = true;
          if (FOUND_LOWER)
            break;
        }
        if (!FOUND_LOWER && rho_(i_x,i_y,i_z) > rhoCutoff &&
            rho_(i_x,i_y,i_z_shift) < rhoCutoff) {
          lower_mesh_(i_x,i_y,2) = grid_[i_x][i_y][i_z][2] -
              (rho_(i_x,i_y,i_z)-rhoCutoff)/
              (rho_(i_x,i_y,i_z) - rho_(i_x,i_y,i_z_shift))*
              spacing_;
          if (lower_mesh_(i_x,i_y,2) < 0.0)
            lower_mesh_(i_x,i_y,2) += length_(2);
          FOUND_LOWER = true;
          if (FOUND_UPPER)
            break;
        }
      }
      file_ << upper_mesh_(i_x,i_y,2) << " " <<
                   lower_mesh_(i_x,i_y,2) << std::endl;
    }
  }
}
