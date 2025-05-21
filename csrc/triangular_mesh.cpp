// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <math.h>

py::tuple get_mesh(long nx, long ny) {
  double delta_x = 1. / nx;
  double delta_y = 1. / ny;
  double area = delta_x * delta_y / 4.;
  long nv = nx * ny * 4;
  long ne = nx * ny * 12 - nx * 2 - ny * 2;
  long np = (nx + 1) * (ny + 1) + nx * ny;

  // std::cout << "nx: " << nx << std::endl;
  // std::cout << "ny: " << ny << std::endl;
  // std::cout << "nv: " << nv << std::endl;
  // std::cout << "ne: " << ne << std::endl;
  // std::cout << "np: " << np << std::endl;

  std::vector<long> src(ne, 0);
  std::vector<long> dst(ne, 0);
  std::vector<long> cells(nv * 3, 0);
  std::vector<double> points(np * 2, 0);
  std::vector<long> bcells(nx * 2 + ny * 2, 0);
  std::vector<long> bpoints((nx * 2 + ny * 2) * 2, 0);

  #pragma omp parallel for
  for (long j = 0; j < ny; j++) {
    for (long i = 0; i < nx; i++) {
      // std::cout << i << ',' << j << std::endl;
      long nrect = i + j * nx;
      long ncell = nrect * 4;
      long p0 = j * (nx + 1) + i;
      long p1 = (j + 1) * (nx + 1) + i;
      long p2 = (j + 1) * (nx + 1) + i + 1;
      long p3 = j * (nx + 1) + i + 1;
      long p4 = (nx + 1) * (ny + 1) + nx * j + i;

      points[p0 * 2] = i * delta_x;
      points[p0 * 2 + 1] = j * delta_y;
      points[p1 * 2] = i * delta_x;
      points[p1 * 2 + 1] = j * delta_y + delta_y;
      points[p2 * 2] = i * delta_x + delta_x;
      points[p2 * 2 + 1] = j * delta_y + delta_y;
      points[p3 * 2] = i * delta_x + delta_x;
      points[p3 * 2 + 1] = j * delta_y;
      points[p4 * 2] = i * delta_x + delta_x / 2;
      points[p4 * 2 + 1] = j * delta_y + delta_y / 2;

      cells[ncell * 3] = p0;
      cells[ncell * 3 + 1] = p3;
      cells[ncell * 3 + 2] = p4;

      cells[(ncell + 1) * 3] = p2;
      cells[(ncell + 1) * 3 + 1] = p4;
      cells[(ncell + 1) * 3 + 2] = p3;

      cells[(ncell + 2) * 3] = p2;
      cells[(ncell + 2) * 3 + 1] = p1;
      cells[(ncell + 2) * 3 + 2] = p4;

      cells[(ncell + 3) * 3] = p0;
      cells[(ncell + 3) * 3 + 1] = p4;
      cells[(ncell + 3) * 3 + 2] = p1;

      // internal edges
      long offset = nrect * 8;
      for (long k = 0; k < 4; k++) {
        long src_cell = ncell + k;
        long dst_cell = 0;
        if (k == 3) {
          dst_cell = ncell;
        }else{
          dst_cell = ncell + k + 1;
        }
        src[offset + k * 2] = src_cell;
        dst[offset + k * 2] = dst_cell;
        src[offset + k * 2 + 1] = dst_cell;
        dst[offset + k * 2 + 1] = src_cell;
      }

      // downward edges
      if (j != 0) {
        offset = 8 * nx * ny + nx * (j - 1) + i;
        src[offset] = ncell;
        dst[offset] = ncell - 4 * nx + 2;
      } else {
        bcells[i] = ncell;
        bpoints[i * 2] = p0;
        bpoints[i * 2 + 1] = p3;
      }

      // upward edges
      if (j != ny - 1) {
        offset = 8 * nx * ny + nx * (ny - 1) + nx * j + i;
        src[offset] = ncell + 2;
        dst[offset] = ncell + 4 * nx;
      } else {
        bcells[nx + i] = ncell + 2;
        bpoints[nx * 2 + i * 2] = p1;
        bpoints[nx * 2 + i * 2 + 1] = p2;
      }

      // leftward edges
      if (i != 0) {
        offset = 8 * nx * ny + 2 * nx * (ny - 1) + (nx - 1) * j + i - 1;
        src[offset] = ncell + 3;
        dst[offset] = ncell - 3;
      } else {
        bcells[nx * 2 + j] = ncell + 3;
        bpoints[nx * 2 * 2 + j * 2] = p0;
        bpoints[nx * 2 * 2 + j * 2 + 1] = p1;
      }

      // right edges
      if (i != nx - 1) {
        offset =
          8 * nx * ny + 2 * nx * (ny - 1) + (nx - 1) * ny + (nx - 1) * j + i;
        src[offset] = ncell + 1;
        dst[offset] = ncell + 7;
      } else {
        bcells[nx * 2 + ny + j] = ncell + 1;
        bpoints[nx * 2 * 2 + ny * 2 + j * 2] = p2;
        bpoints[nx * 2 * 2 + ny * 2 + j * 2 + 1] = p3;
      }
    }
  }

  auto opt = torch::TensorOptions().dtype(torch::kLong);
  torch::Tensor SRC = torch::tensor(at::ArrayRef<long>(src), opt);
  torch::Tensor DST = torch::tensor(at::ArrayRef<long>(dst), opt);
  torch::Tensor CELLS = torch::tensor(at::ArrayRef<long>(cells), opt);
  torch::Tensor BCELLS = torch::tensor(at::ArrayRef<long>(bcells), opt);
  torch::Tensor BPOINTS = torch::tensor(at::ArrayRef<long>(bpoints), opt);

  auto opt_double = torch::TensorOptions().dtype(torch::kDouble);
  torch::Tensor POINTS = torch::tensor(at::ArrayRef<double>(points), opt_double);

  return py::make_tuple(SRC, DST, CELLS, BCELLS, BPOINTS, POINTS);
}

void pybind_triangular_mesh(pybind11::module_ m) {
  m.def("generate_triangular_mesh", &get_mesh, "Get mesh");
}