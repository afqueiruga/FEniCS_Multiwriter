#!/usr/bin/python

"""
Write multiple fields to a single vtk file.
"""

from dolfin import *
import numpy as np
import os

srcdir = str(os.path.dirname(os.path.realpath(__file__))+"/src/")
header_file = open(srcdir+"/multiwriter.h", "r")
code = header_file.read()
header_file.close()
compiled_module = compile_extension_module(
    code=code, source_directory=srcdir, sources=["multiwriter.cpp"],
    include_dirs=[".",os.path.abspath(srcdir)],
    additional_declarations='%feature("notabstract") ProximityTree3D;')

def writedata(self,i,point_data,cell_data):
    self.start_file(point_data[0].function_space().mesh(),i)
    self.start_point_data()
    for u in point_data:
        self.push_point_data(u)
    self.end_point_data()
    self.start_cell_data()
    for u in cell_data:
        self.push_cell_data(u)
    self.end_cell_data()
    self.close_file()
compiled_module.VTKAppender.write = writedata

VTKAppender = compiled_module.VTKAppender

if __name__=="__main__":
    me = UnitSquareMesh(2,2);
    V = FunctionSpace(me,"CG",1);
    u = Function(V);
    u.interpolate(Expression("x[0]"));
    y = Function(V);
    y.interpolate(Expression("x[0]*x[1]"));
    Vvec = VectorFunctionSpace(me,"DG",0);
    v = Function(Vvec)
    v.interpolate(Expression(("x[0]","x[1]")))
    vf = compiled_module.VTKAppender("foo.pvd","ascii");
    vf.write(0,[u,y],[v])
    # vf.start_file(me,0)
    # vf.start_point_data()
    # vf.push_point_data(u)
    # vf.push_point_data(y)
    # vf.end_point_data()
    # vf.start_cell_data()
    # vf.push_cell_data(v)
    # vf.end_cell_data()
    # vf.close_file()
