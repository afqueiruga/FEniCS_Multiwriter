
#ifndef __VTK_APPENDER_H
#define __VTK_APPENDER_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "dolfin/io/GenericFile.h"
#include "dolfin/io/VTKFile.h"

#include "dolfin/geometry/Point.h"
#include "dolfin/geometry/BoundingBoxTree3D.h"

namespace dolfin
{
  class VTKAppender : public VTKFile
  {
  public:
    VTKAppender(const std::string filename, std::string encoding);
    ~VTKAppender();
    void start_file(const Mesh &u, double time);
    void append_function(const Function & u);
    void close_file();
    void start_point_data();
    void push_point_data(const Function & u);
    void end_point_data();
    void start_cell_data();
    void push_cell_data(const Function &u);
    void end_cell_data();

    void write_point_data(const GenericFunction& u, const Mesh& mesh,
				       std::string vtu_filename, bool binary, bool compress) const;
    void write_cell_data(const Function& u, std::string filename, bool binary, bool compress);

    std::string ascii_cell_data(const Mesh& mesh,
				const std::vector<std::size_t>& offset,
				const std::vector<double>& values,
				std::size_t data_dim, std::size_t rank);
    
    std::string base64_cell_data(const Mesh& mesh,
				 const std::vector<std::size_t>& offset,
				 const std::vector<double>& values,
				 std::size_t data_dim, std::size_t rank,
				 bool compress);
    std::string vtu_filename;
  };



  class ProximityTree3D : public BoundingBoxTree3D
  {
  public:
    ProximityTree3D();
    ~ProximityTree3D();
    std::vector<unsigned int> compute_proximity_collisions(const Point& point, const Mesh & u, double radius);
  };


}




#endif
