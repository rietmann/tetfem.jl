using namespace std;

#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolygon.h>
#include <vtkSmartPointer.h>
#include <vtkMath.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCleanPolyData.h>
#include <vtkDelaunay3D.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkXMLDataSetWriter.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGridReader.h>

#include "vtklib.h"

extern "C"
void freedbl_vtklib(double* ptr) {
  delete ptr;
}

extern "C"
void freeint_vtklib(int* ptr) {
  delete ptr;  
}

extern "C" void test_libvtk(int answer,double** vector) {

  printf("Loading file: %d\n",answer);
  printf("Vector: %p\n",vector);  
  *vector = new double[5];
  printf("Allocated vector: %p\n",*vector);
  // printf("Allocated vector\n");
  for(int i=0;i<5;i++) {
    (*vector)[i] = i;
  }
  printf("Finished test_libvtk\n");
}

extern "C"
void loadmesh_vtklib(char* filename,int* c_verts_per_element, int& number_elements,int& number_vertices,double** v_x,double** v_y,double** v_z, int** EToV) {

  vtkSmartPointer<vtkUnstructuredGridReader> reader =
    vtkSmartPointer<vtkUnstructuredGridReader>::New();
  reader->SetFileName(filename);
  reader->Update();

  vtkUnstructuredGrid* mesh_file = reader->GetOutput();

  // triangle or tetrahedra
  *c_verts_per_element = mesh_file->GetCell(0)->GetNumberOfPoints();
  int verts_per_element = *c_verts_per_element;
  
  // ------ Read Points -------  
  number_vertices = mesh_file->GetNumberOfPoints();  
  *v_x = new double[number_vertices];
  *v_y = new double[number_vertices];
  if(verts_per_element == 4) {*v_z = new double[number_vertices];}

  double xyz[3];
  for(int i=0;i<number_vertices;i++) {
    mesh_file->GetPoint(i,xyz);
    (*v_x)[i] = xyz[0];
    (*v_y)[i] = xyz[1];
    if(verts_per_element == 4) {(*v_z)[i] = xyz[2];}
  }

  // ------ Read Cells -------
  number_elements = mesh_file->GetNumberOfCells();
  
  *EToV = new int[verts_per_element*number_elements];
  for(int k=0;k<number_elements;k++) {
    // if(p_DIM == 2) {TETFEM_ASSERT(==3) << "Should be 3 points per triangle!";}
    // if(p_DIM == 3) {TETFEM_ASSERT(mesh_file->GetCell(k)->GetNumberOfPoints()==4) << "Should be 4 points per tetrahedra!";}
    VTKLIB_ASSERT(mesh_file->GetCell(k)->GetNumberOfPoints() == verts_per_element)
      << "Vertices per element should remain consistent through file";
    (*EToV)[k*verts_per_element + 0] = mesh_file->GetCell(k)->GetPointId(0);
    (*EToV)[k*verts_per_element + 1] = mesh_file->GetCell(k)->GetPointId(1);
    (*EToV)[k*verts_per_element + 2] = mesh_file->GetCell(k)->GetPointId(2);
    if(verts_per_element == 4) {(*EToV)[k*verts_per_element + 3] = mesh_file->GetCell(k)->GetPointId(3);}    
  }

}

