// Minimal, ROS-free <urdf/model.h> shim for tracikpy.
//
// tracikpy's trac_ik.cpp does:  urdf::Model robot_model; robot_model.initString(xml);
// urdf::Model (with initString/initFile) lives in the ROS `urdf` package, NOT in plain urdfdom --
// urdfdom only exposes the free function urdf::parseURDF(). This shim provides the tiny wrapper
// class: a urdf::ModelInterface subclass whose initString()/initFile() delegate to urdfdom's
// parser and copy the result into itself. Depends only on urdfdom.
#ifndef URDF_MODEL_SHIM_H_
#define URDF_MODEL_SHIM_H_

#include <iostream>  // trac_ik.cpp uses std::cerr and relied on the ROS urdf header pulling this in
#include <string>

#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

namespace urdf
{
class Model : public ModelInterface
{
public:
  /// Parse a URDF XML string into this model. Returns false on parse failure.
  bool initString(const std::string & xml)
  {
    ModelInterfaceSharedPtr parsed = parseURDF(xml);
    if (!parsed) {
      return false;
    }
    *static_cast<ModelInterface *>(this) = *parsed;
    return true;
  }

  /// Parse a URDF file on disk into this model. Returns false on parse failure.
  bool initFile(const std::string & filename)
  {
    ModelInterfaceSharedPtr parsed = parseURDFFile(filename);
    if (!parsed) {
      return false;
    }
    *static_cast<ModelInterface *>(this) = *parsed;
    return true;
  }
};
}  // namespace urdf

#endif  // URDF_MODEL_SHIM_H_
