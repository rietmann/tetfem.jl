#ifndef VTKLIB_H
#define VTKLIB_H

#include <iostream>
#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>

class VTKLIBAssert
{
 public:
    VTKLIBAssert();
    virtual ~VTKLIBAssert();
    std::ostringstream& Get(int line_no,std::string file);
 protected:
    std::ostringstream os;
 private:
    VTKLIBAssert(const VTKLIBAssert&);
    VTKLIBAssert& operator =(const VTKLIBAssert&);
    int rank;
};

#define VTKLIB_ASSERT(bool_assertion)           \
    if(bool_assertion);                         \
    else VTKLIBAssert().Get(__LINE__,__FILE__)

#endif
