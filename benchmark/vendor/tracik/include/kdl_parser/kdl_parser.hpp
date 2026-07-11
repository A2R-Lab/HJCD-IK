// Minimal, ROS-free kdl_parser shim for tracikpy.
//
// Upstream kdl_parser is a ROS package; on a bare (non-ROS) Linux box its header is absent and
// tracikpy fails to build (fatal error: kdl_parser/kdl_parser.hpp: No such file or directory).
// tracikpy only ever calls treeFromUrdfModel() (it parses the URDF itself via urdfdom), so this
// shim provides exactly that plus the string/file convenience overloads, built on urdfdom + KDL
// only -- no ROS, no tinyxml, no console_bridge.
//
// Derived from ros/kdl_parser (noetic-devel, BSD-3-Clause, (c) 2008 Willow Garage).
#ifndef KDL_PARSER_KDL_PARSER_HPP_
#define KDL_PARSER_KDL_PARSER_HPP_

#include <string>

#include <kdl/tree.hpp>
#include <urdf_model/model.h>

namespace kdl_parser
{
/// Build a KDL::Tree from a URDF file on disk.
bool treeFromFile(const std::string & file, KDL::Tree & tree);

/// Build a KDL::Tree from a URDF XML string.
bool treeFromString(const std::string & xml, KDL::Tree & tree);

/// Build a KDL::Tree from an already-parsed urdfdom model. (The overload tracikpy uses.)
bool treeFromUrdfModel(const urdf::ModelInterface & robot_model, KDL::Tree & tree);
}  // namespace kdl_parser

#endif  // KDL_PARSER_KDL_PARSER_HPP_
