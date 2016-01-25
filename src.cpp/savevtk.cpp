#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <map>
using namespace std;

#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkProperty.h>
#include <vtkDataSet.h>
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
#include <vtkDelaunay2D.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkXMLDataSetWriter.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkGeometryFilter.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkFeatureEdges.h>
#include <vtkOutlineFilter.h>

vtkPoints* points;
vtkDoubleArray* data_array;
vtkPolyData* volume;
vtkDelaunay2D* del;
vtkCamera* camera;
vtkDataSetMapper* mapMesh;
vtkActor* actor;
vtkRenderer* ren;
vtkRenderWindow* renWin;
vtkRenderWindowInteractor* iren;

vtkPolyData* mesh2d;
vtkUnstructuredGrid* mesh3d;

extern "C"
void test(double* data,char* name,int length,void* tetval) {
  printf("%s[0 of %d] = %e\n",name,length,data[0]);
}

extern "C"
void initializeSavePoints2d(double* data,char* name,char* output_dir, int ndof,int K, double* x, double* y, void* EToV_local, int num_interior_elements, void* dofIndex, int(*index2)(void*,int,int)) {
  
  mesh2d = vtkPolyData::New();
  vtkPoints* points = vtkPoints::New();
  data_array = vtkDoubleArray::New();
  data_array->SetNumberOfValues(ndof);
  points->SetNumberOfPoints(ndof);
  mesh2d->Allocate(ndof,ndof);
  double xyz[3];

  for(int i=0;i<ndof;i++) {
    xyz[0] = x[i];
    xyz[1] = y[i];
    xyz[2] = 0;
    points->SetPoint(i,xyz);
    data_array->SetValue(i,data[i]);
  }
  
  int idx = index2(dofIndex,1,1);

  vtkIdType ids[3];
  // add each "element" of this element to the visualization "mesh"
  for(int k=0;k<K;k++) {
    for(int ki=0;ki<num_interior_elements;ki++) {
      int dofidx0 = index2(EToV_local,ki+1,1);
      int dofidx1 = index2(EToV_local,ki+1,2);
      int dofidx2 = index2(EToV_local,ki+1,3);
      ids[0] = index2(dofIndex,k+1,dofidx0)-1;
      ids[1] = index2(dofIndex,k+1,dofidx1)-1;
      ids[2] = index2(dofIndex,k+1,dofidx2)-1;
      mesh2d->InsertNextCell(VTK_TRIANGLE,3,ids);
    }
  } 
  
  data_array->SetName(name);
  mesh2d->SetPoints(points);
  mesh2d->GetPointData()->SetScalars(data_array);
  
  // setup output
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =
    vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetInput(mesh2d);
  
  // // int procid;
  // // MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  
  // make filename
  char filename[256];
  printf("Saving files to: %s/mesh_points_step_*******.vtu\n",output_dir);
  sprintf(filename,"%s/mesh_points_step_%07d.vtu",output_dir,0);
  
  // write file
  writer->SetFileName(filename);
  writer->Write();
  
}

extern "C"
void initializeSavePoints3d(double* data,char* name,char* output_dir, int ndof,int K, double* x, double* y, double* z, void* EToV_local, int num_interior_elements, void* dofIndex, int(*index2)(void*,int,int)) {
    
  mesh3d = vtkUnstructuredGrid::New();
  vtkPoints* points = vtkPoints::New();
  data_array = vtkDoubleArray::New();
  data_array->SetNumberOfValues(ndof);
  points->SetNumberOfPoints(ndof);
  mesh3d->Allocate(ndof,ndof);
  double xyz[3];

  for(int i=0;i<ndof;i++) {
    xyz[0] = x[i];
    xyz[1] = y[i];
    xyz[2] = z[i];
    points->SetPoint(i,xyz);
    data_array->SetValue(i,data[i]);
  }
  
  int idx = index2(dofIndex,1,1);

  vtkIdType ids[4];
  // add each "element" of this element to the visualization "mesh"
  for(int k=0;k<K;k++) {
    for(int ki=0;ki<num_interior_elements;ki++) {
      int dofidx0 = index2(EToV_local,ki+1,1);
      int dofidx1 = index2(EToV_local,ki+1,2);
      int dofidx2 = index2(EToV_local,ki+1,3);
      int dofidx3 = index2(EToV_local,ki+1,4);
      ids[0] = index2(dofIndex,k+1,dofidx0)-1;
      ids[1] = index2(dofIndex,k+1,dofidx1)-1;
      ids[2] = index2(dofIndex,k+1,dofidx2)-1;
      ids[3] = index2(dofIndex,k+1,dofidx3)-1;
      mesh3d->InsertNextCell(VTK_TETRA,4,ids);
    }
  } 
  
  data_array->SetName(name);
  mesh3d->SetPoints(points);
  mesh3d->GetPointData()->SetScalars(data_array);
  
  // setup output
  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
    vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetInput(mesh3d);
  
  // // int procid;
  // // MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  
  // make filename
  char filename[256];
  printf("Saving files to: %s/mesh_points_step_*******.vtu\n",output_dir);
  sprintf(filename,"%s/mesh_points_step_%07d.vtu",output_dir,0);
  
  // write file
  writer->SetFileName(filename);
  writer->Write();
  
}

extern "C"
void savePoints3d(double* data,int ndof, char* output_dir, int step) {

  for(int i=0;i<ndof;i++) {
    data_array->SetValue(i,data[i]);      
  }

  // setup output
  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer =
    vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetInput(mesh3d);

  // int procid;
  // MPI_Comm_rank(MPI_COMM_WORLD,&procid);
  
  // make filename
  char filename[256];
  sprintf(filename,"%s/mesh_points_step_%07d.vtu",output_dir,step);
  // write file
  writer->SetFileName(filename);
  writer->Write();

}

extern "C"
void savePoints2d(double* data,int ndof, char* output_dir, int step) {

  
  for(int i=0;i<ndof;i++) {
    data_array->SetValue(i,data[i]);      
  }

  // setup output
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =
    vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetInput(mesh2d);

  // make filename
  char filename[256];
  sprintf(filename,"%s/mesh_points_step_%07d.vtu",output_dir,step);

  // write file
  writer->SetFileName(filename);
  writer->Write();

}
