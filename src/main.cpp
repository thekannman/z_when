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

#include "z_sim_params.hpp"
#include "z_string.hpp"
#include "z_vec.hpp"
#include "z_conversions.hpp"
#include "z_molecule.hpp"
#include "z_atom_group.hpp"
#include "z_gromacs.hpp"
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"

namespace po = boost::program_options;
// Units are nm, ps.

int main (int argc, char *argv[])
{
  int st;
  SimParams params;
  int max_steps = std::numeric_limits<int>::max();

  double gdsTop = 3.2, gdsBottom = 8.75;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->default_value("He"),
     "Group for density/temperature profiles")
    ("liquid,l", po::value<std::string>()->default_value("OW"),
     "Group to use for calculation of surface")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("max_time,t",
     po::value<double>()->default_value(std::numeric_limits<double>::max()),
     "Maximum simulation time to use in calculations")
    ("split,s", "Use splitting from liquid as desorption event")
    ("rmin,r", po::value<double>()->default_value(0.48),
     "Cutoff to use for determination of first solvation shell");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }
  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());
  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                  params);
  AtomGroup all_atoms(vm["gro"].as<std::string>(), molecules);
  AtomGroup selected_group(vm["group"].as<std::string>(),
                           SelectGroup(groups, vm["group"].as<std::string>()),
                           all_atoms);
  AtomGroup liquid_group(vm["liquid"].as<std::string>(),
                         SelectGroup(groups, vm["liquid"].as<std::string>()),
                         all_atoms);
  bool split = vm.count("split") ? true : false;
  const double rmin2 = vm["rmin"].as<double>()*vm["rmin"].as<double>();

  arma::irowvec when = arma::zeros<arma::irowvec >(selected_group.size());
  arma::irowvec when_maybe = arma::zeros<arma::irowvec >(selected_group.size());
  arma::irowvec steps_in_vapor =
    arma::zeros<arma::irowvec >(selected_group.size());
  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  std::string trr_filename = "prod.trr";
  XDRFILE *xtc_file, *trr_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  trr_file = xdrfile_open(strdup(trr_filename.c_str()), "r");
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_max_time(vm["max_time"].as<double>());

  arma::rowvec dx = arma::zeros<arma::rowvec>(DIMS);
  const int split_delay = 5;
  arma::mat x_when = arma::zeros(selected_group.size(), DIMS);
  arma::mat v_when = arma::zeros(selected_group.size(), DIMS);
  arma::mat x_maybe = arma::zeros(selected_group.size(), DIMS);
  arma::mat v_maybe = arma::zeros(selected_group.size(), DIMS);
  rvec *v_in = NULL;
  float time, lambda, prec;
  for (int step=0, steps=0; step<max_steps; step++) {
    steps++;
    if(read_xtc(xtc_file,  params.num_atoms(), &st, &time, box_mat, x_in,
                &prec)) {
      break;
    }
    if(read_trr(trr_file, params.num_atoms(), &st, &time, &lambda, box_mat,
                NULL, v_in, NULL)) {
      break;
    }
    int i = 0;
    for (std::vector<int>::iterator i_atom = selected_group.begin();
         i_atom != selected_group.end(); ++i_atom, ++i) {
      selected_group.set_position(i, x_in[*i_atom]);
      selected_group.set_velocity(i, v_in[*i_atom]);
    }
    i = 0;
    for (std::vector<int>::iterator i_atom = liquid_group.begin();
         i_atom != liquid_group.end(); ++i_atom, ++i) {
      liquid_group.set_position(i, x_in[*i_atom]);
      liquid_group.set_velocity(i, v_in[*i_atom]);
    }
    //Start checking from here
    for (int i_sel=0; i_sel<selected_group.size(); i_sel++) {
      if (when(i_sel) != 0) continue;
      int is_in_vapor = 1;
      for (int i_liq = 0; i_liq < liquid_group.size(); i_liq++) {
        FindDxNoShift(dx, selected_group.position(i_sel),
                            liquid_group.position(i_liq), box);
        double r2 = arma::dot(dx,dx);
        if (r2<rmin2 && (selected_group.index_to_molecule(i_sel) !=
                         liquid_group.index_to_molecule(i_liq))) {
          is_in_vapor = 0;
          steps_in_vapor(i_sel) = 0;
          break;
        }
      }
      if(is_in_vapor) {
        if(steps_in_vapor(i_sel) == 0) {
          when_maybe(i_sel) = i;
          x_maybe(i_sel) = selected_group.position(i_sel,2);
          v_maybe(i_sel) = selected_group.velocity(i_sel,2);
        }
        steps_in_vapor(i_sel)++;
        if (steps_in_vapor(i_sel) == split_delay) {
          // In case the particle started outside the bulk
          if (i==split_delay) {
            when(i_sel) = -1;
            x_when(i_sel) = 0.0;
            v_when(i_sel) = 0.0;
          } else {
            when(i_sel) = when_maybe(i_sel);
            x_when(i_sel) = x_maybe(i_sel);
            v_when(i_sel) = v_maybe(i_sel);
          }
        }
      }
    }
  }

  std::string when_filename = split ? "whenSplit.dat" : "when.dat";
  std::ofstream when_file;
  when_file.open(when_filename.c_str());
  for (int i=0; i<selected_group.size(); i++) {
    when_file << when(i) << std::endl;
  }
  when_file.close();

  std::string x_when_filename = "x_whenSplit.dat";
  std::ofstream x_when_file;
  x_when_file.open(x_when_filename.c_str());
  for (int i=0; i<selected_group.size(); i++) {
    x_when_file << x_when(i) << std::endl;
  }
  x_when_file.close();

  std::string v_when_filename = "v_whenSplit.dat";
  std::ofstream v_when_file;
  v_when_file.open(v_when_filename.c_str());
  for (int i=0; i<selected_group.size(); i++) {
    v_when_file << v_when(i) << std::endl;
  }
  v_when_file.close();

} // main
