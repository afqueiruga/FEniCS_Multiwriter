#include <ostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <boost/cstdint.hpp>
#include <boost/detail/endian.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>

#include "dolfin/io/Encoder.h"
#include "dolfin/io/VTKWriter.h"
#include "dolfin/io/VTKFile.h"

#include "multiwriter.h"

namespace dolfin {
  VTKAppender::VTKAppender(const std::string filename, std::string encoding)
    : VTKFile(filename, encoding)
  {
  }
  VTKAppender::~VTKAppender()
  {
    // Pass
  }
  void VTKAppender::start_file(const Mesh &mesh, double time) {
    vtu_filename =init(mesh, mesh.topology().dim());
    std::cout << counter << "  " << vtu_filename << "\n";
    VTKWriter::write_mesh(mesh,mesh.topology().dim(), vtu_filename, 0,0);
  }
  void VTKAppender::append_function(const Function & u) {
    // write_point_data(u,vtu_filename);
    results_write(u,vtu_filename);
  }

  void VTKAppender::start_point_data() {
      std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
      fp.precision(16);
      fp << "<PointData> " << std::endl;
  }
  void VTKAppender::push_point_data(const Function & u) {

      dolfin_assert(u.function_space()->mesh());
      const Mesh& mesh = *u.function_space()->mesh();

      write_point_data(u, mesh, vtu_filename, false,false);

  }
  void VTKAppender::end_point_data() {
      std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
      fp.precision(16);
      fp << "</PointData> " << std::endl;
  }
  void VTKAppender::start_cell_data() {
    std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
    fp.precision(16);
    fp << "<CellData> " << std::endl;
  }
  void VTKAppender::push_cell_data(const Function &u) {
 
    write_cell_data(u, vtu_filename, false, false);
  }
  void VTKAppender::end_cell_data() {
      std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
      fp.precision(16);
    fp << "</CellData> " << std::endl;
  }

  void VTKAppender::close_file() {
    vtk_header_close(vtu_filename);
    counter++;
    // finalize(vtu_filename,0.0);
  }




void VTKAppender::write_point_data(const GenericFunction& u, const Mesh& mesh,
				   std::string vtu_filename, bool binary, bool compress) const
{
  const std::size_t rank = u.value_rank();
  const std::size_t num_vertices = mesh.num_vertices();

  // Get number of components
  const std::size_t dim = u.value_size();

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Allocate memory for function values at vertices
  const std::size_t size = num_vertices*dim;
  std::vector<double> values(size);

  std::string encode_string;
  if (!binary)
    encode_string = "ascii";
  else
    encode_string = "binary";
  std::string _encoding = encode_string;

  // Get function values at vertices and zero any small values
  u.compute_vertex_values(values, mesh);
  dolfin_assert(values.size() == size);
  std::vector<double>::iterator it;
  for (it = values.begin(); it != values.end(); ++it)
  {
    if (std::abs(*it) < DOLFIN_EPS)
      *it = 0.0;
  }

  if (rank == 0)
  {
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  format=\""<< encode_string <<"\">";
  }
  else if (rank == 1)
  {
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">";
  }
  else if (rank == 2)
  {
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">";
  }

  if (_encoding == "ascii")
  {
    std::ostringstream ss;
    ss << std::scientific;
    ss << std::setprecision(16);
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    {
      if (rank == 1 && dim == 2)
      {
        // Append 0.0 to 2D vectors to make them 3D
        for(std::size_t i = 0; i < 2; i++)
          ss << values[vertex->index() + i*num_vertices] << " ";
        ss << 0.0 << "  ";
      }
      else if (rank == 2 && dim == 4)
      {
        // Pad 2D tensors with 0.0 to make them 3D
        for(std::size_t i = 0; i < 2; i++)
        {
          ss << values[vertex->index() + (2*i + 0)*num_vertices] << " ";
          ss << values[vertex->index() + (2*i + 1)*num_vertices] << " ";
          ss << 0.0 << " ";
        }
        ss << 0.0 << " ";
        ss << 0.0 << " ";
        ss << 0.0 << "  ";
      }
      else
      {
        // Write all components
        for(std::size_t i = 0; i < dim; i++)
          ss << values[vertex->index() + i*num_vertices] << " ";
        ss << " ";
      }
    }

    // Send to file
    fp << ss.str();
  }
  else if (_encoding == "base64" || _encoding == "compressed")
  {
    // Number of zero paddings per point
    std::size_t padding_per_point = 0;
    if (rank == 1 && dim == 2)
      padding_per_point = 1;
    else if (rank == 2 && dim == 4)
      padding_per_point = 5;

    // Number of data entries per point and total number
    const std::size_t num_data_per_point = dim + padding_per_point;
    const std::size_t num_total_data_points = num_vertices*num_data_per_point;

    std::vector<double> data(num_total_data_points, 0);
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    {
      const std::size_t index = vertex->index();
      for(std::size_t i = 0; i < dim; i++)
        data[index*num_data_per_point + i] = values[index + i*num_vertices];
    }

    // Create encoded stream
    fp << VTKWriter::encode_stream(data, compress) << std::endl;
  }

  fp << "</DataArray> " << std::endl;
}



void VTKAppender::write_cell_data(const Function& u, std::string filename,
                                bool binary, bool compress)
{
  // For brevity
  dolfin_assert(u.function_space()->mesh());
  dolfin_assert(u.function_space()->dofmap());
  const Mesh& mesh = *u.function_space()->mesh();
  const GenericDofMap& dofmap = *u.function_space()->dofmap();
  const std::size_t num_cells = mesh.num_cells();

  std::string encode_string;
  if (!binary)
    encode_string = "ascii";
  else
    encode_string = "binary";

  // Get rank of Function
  const std::size_t rank = u.value_rank();
  if(rank > 2)
  {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
  }

  // Get number of components
  const std::size_t data_dim = u.value_size();

  // Open file
  std::ofstream fp(filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Write headers
  if (rank == 0)
  {
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  format=\""
       << encode_string <<"\">";
  }
  else if (rank == 1)
  {
    if(!(data_dim == 2 || data_dim == 3))
    {
      dolfin_error("VTKWriter.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
    }
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">";
  }
  else if (rank == 2)
  {
    if(!(data_dim == 4 || data_dim == 9))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle tensor function with dimension other than 4 or 9");
    }
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">";
  }

  // Allocate memory for function values at cell centres
  const std::size_t size = num_cells*data_dim;

  // Build lists of dofs and create map
  std::vector<dolfin::la_index> dof_set;
  std::vector<std::size_t> offset(size + 1);
  std::vector<std::size_t>::iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Tabulate dofs
    const std::vector<dolfin::la_index>& dofs = dofmap.cell_dofs(cell->index());
    for(std::size_t i = 0; i < dofmap.cell_dimension(cell->index()); ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1) = *(cell_offset) + dofmap.cell_dimension(cell->index());
    ++cell_offset;
  }

  // Get  values
  std::vector<double> values(dof_set.size());
  dolfin_assert(u.vector());
  u.vector()->get_local(values.data(), dof_set.size(), dof_set.data());

  // Get cell data
  if (!binary)
    fp << ascii_cell_data(mesh, offset, values, data_dim, rank);
  else
  {
    fp << base64_cell_data(mesh, offset, values, data_dim, rank, compress)
       << std::endl;
  }
  fp << "</DataArray> " << std::endl;

}





std::string VTKAppender::ascii_cell_data(const Mesh& mesh,
                                       const std::vector<std::size_t>& offset,
                                       const std::vector<double>& values,
                                       std::size_t data_dim, std::size_t rank)
{
  std::ostringstream ss;
  ss << std::scientific;
  ss << std::setprecision(16);
  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (rank == 1 && data_dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      ss << values[*cell_offset] << "  " << values[*cell_offset + 1] << " "
         << 0.0;
    }
    else if (rank == 2 && data_dim == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for(std::size_t i = 0; i < 2; i++)
      {
        ss << values[*cell_offset + 2*i] << " ";
        ss << values[*cell_offset + 2*i + 1] << " ";
        ss << 0.0 << " ";
      }
      ss << 0.0 << " ";
      ss << 0.0 << " ";
      ss << 0.0;
    }
    else
    {
      // Write all components
      for (std::size_t i = 0; i < data_dim; i++)
        ss << values[*cell_offset + i] << " ";
    }
    ss << "  ";
    ++cell_offset;
  }

  return ss.str();
}

std::string VTKAppender::base64_cell_data(const Mesh& mesh,
                                        const std::vector<std::size_t>& offset,
                                        const std::vector<double>& values,
                                        std::size_t data_dim, std::size_t rank,
                                        bool compress)
{
  const std::size_t num_cells = mesh.num_cells();

  // Number of zero paddings per point
  std::size_t padding_per_point = 0;
  if (rank == 1 && data_dim == 2)
    padding_per_point = 1;
  else if (rank == 2 && data_dim == 4)
    padding_per_point = 5;

  // Number of data entries per point and total number
  const std::size_t num_data_per_point = data_dim + padding_per_point;
  const std::size_t num_total_data_points = num_cells*num_data_per_point;

  std::vector<std::size_t>::const_iterator cell_offset = offset.begin();
  std::vector<double> data(num_total_data_points, 0);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t index = cell->index();
    for(std::size_t i = 0; i < data_dim; i++)
      data[index*num_data_per_point + i] = values[*cell_offset + i];
    ++cell_offset;
  }

  return VTKWriter::encode_stream(data, compress);
}







  ProximityTree3D::ProximityTree3D()
  {
    // Do nothing
    // _tdim=0;
  }
  ProximityTree3D::~ProximityTree3D()
  {
    //do nothing
  }



  std::vector<unsigned int>
  ProximityTree3D::compute_proximity_collisions(const Point& point,
						const Mesh &u,
					      double radius) 
  {
    // Mesh * mesh = &u;
    // // Point in entity only implemented for cells. Consider extending this.
    // if (_tdim != mesh->topology().dim())
    //   {
    // 	dolfin_error("GenericBoundingBoxTree.cpp",
    // 		     "compute collision between point and mesh entities",
    // 		     "Point-in-entity is only implemented for cells");
    //   }
    
    // // Call recursive find function to compute bounding box candidates
    std::vector<unsigned int> entities;
    // _compute_collisions(*this, point, num_bboxes() - 1, entities, mesh);
    printf("%d\n",u.topology().dim());
    return entities;
  }

}
