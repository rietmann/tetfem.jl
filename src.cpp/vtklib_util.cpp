#include <stdio.h>
#include <stdlib.h>

#include "vtklib.h"

// #include <mpi.h>

std::ostringstream& VTKLIBAssert::Get(int line_number,std::string file) {

  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  os << "ASSERT FAILURE(" << file << ":" << line_number << ") ";
  return os;

}

VTKLIBAssert::VTKLIBAssert() {}

VTKLIBAssert::~VTKLIBAssert() {
  os << std::endl;
  fprintf(stderr, "%s", os.str().c_str());
  fflush(stderr);
  // Abort will be caught by GDB allowing us to debug our ASSERT
  abort();

}
