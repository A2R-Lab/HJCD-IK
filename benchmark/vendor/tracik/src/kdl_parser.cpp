// Minimal, ROS-free kdl_parser implementation for tracikpy. See kdl_parser.hpp for rationale.
// Derived from ros/kdl_parser (noetic-devel, BSD-3-Clause, (c) 2008 Willow Garage, author Wim Meeussen),
// stripped of ROS logging / tinyxml / urdf-compat paths. Depends only on urdfdom + orocos-kdl.

#include "kdl_parser/kdl_parser.hpp"

#include <cstdio>
#include <string>
#include <vector>

#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <kdl/frames_io.hpp>

// Upstream forwarded these to ROS logging; here they go straight to stderr.
#define KDLP_WARN(...) fprintf(stderr, __VA_ARGS__)

namespace kdl_parser
{
// construct vector
static KDL::Vector toKdl(urdf::Vector3 v)
{
  return KDL::Vector(v.x, v.y, v.z);
}

// construct rotation
static KDL::Rotation toKdl(urdf::Rotation r)
{
  return KDL::Rotation::Quaternion(r.x, r.y, r.z, r.w);
}

// construct pose
static KDL::Frame toKdl(urdf::Pose p)
{
  return KDL::Frame(toKdl(p.rotation), toKdl(p.position));
}

// construct joint
static KDL::Joint toKdl(urdf::JointSharedPtr jnt)
{
  KDL::Frame F_parent_jnt = toKdl(jnt->parent_to_joint_origin_transform);

  switch (jnt->type) {
    case urdf::Joint::FIXED: {
        return KDL::Joint(jnt->name, KDL::Joint::None);
      }
    case urdf::Joint::REVOLUTE: {
        KDL::Vector axis = toKdl(jnt->axis);
        return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::RotAxis);
      }
    case urdf::Joint::CONTINUOUS: {
        KDL::Vector axis = toKdl(jnt->axis);
        return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::RotAxis);
      }
    case urdf::Joint::PRISMATIC: {
        KDL::Vector axis = toKdl(jnt->axis);
        return KDL::Joint(jnt->name, F_parent_jnt.p, F_parent_jnt.M * axis, KDL::Joint::TransAxis);
      }
    default: {
        KDLP_WARN("Converting unknown joint type of joint '%s' into a fixed joint\n",
          jnt->name.c_str());
        return KDL::Joint(jnt->name, KDL::Joint::None);
      }
  }
  return KDL::Joint();
}

// construct inertia
static KDL::RigidBodyInertia toKdl(urdf::InertialSharedPtr i)
{
  KDL::Frame origin = toKdl(i->origin);

  // the mass is frame independent
  double kdl_mass = i->mass;

  // kdl and urdf both specify the com position in the reference frame of the link
  KDL::Vector kdl_com = origin.p;

  // kdl specifies the inertia matrix in the reference frame of the link,
  // while the urdf specifies the inertia matrix in the inertia reference frame
  KDL::RotationalInertia urdf_inertia =
    KDL::RotationalInertia(i->ixx, i->iyy, i->izz, i->ixy, i->ixz, i->iyz);

  // Rotation operators are not defined for rotational inertia,
  // so we use the RigidBodyInertia operators (with com = 0) as a workaround
  KDL::RigidBodyInertia kdl_inertia_wrt_com_workaround =
    origin.M * KDL::RigidBodyInertia(0, KDL::Vector::Zero(), urdf_inertia);

  // Note that the RigidBodyInertia constructor takes the 3d inertia wrt the com
  // while the getRotationalInertia method returns the 3d inertia wrt the frame origin
  // (but having com = Vector::Zero() in kdl_inertia_wrt_com_workaround they match)
  KDL::RotationalInertia kdl_inertia_wrt_com =
    kdl_inertia_wrt_com_workaround.getRotationalInertia();

  return KDL::RigidBodyInertia(kdl_mass, kdl_com, kdl_inertia_wrt_com);
}

// recursive function to walk through tree
static bool addChildrenToTree(urdf::LinkConstSharedPtr root, KDL::Tree & tree)
{
  std::vector<urdf::LinkSharedPtr> children = root->child_links;

  // constructs the optional inertia
  KDL::RigidBodyInertia inert(0);
  if (root->inertial) {
    inert = toKdl(root->inertial);
  }

  // constructs the kdl joint
  KDL::Joint jnt = toKdl(root->parent_joint);

  // construct the kdl segment
  KDL::Segment sgm(root->name, jnt, toKdl(
      root->parent_joint->parent_to_joint_origin_transform), inert);

  // add segment to tree
  tree.addSegment(sgm, root->parent_joint->parent_link_name);

  // recursively add all children
  for (size_t i = 0; i < children.size(); i++) {
    if (!addChildrenToTree(children[i], tree)) {
      return false;
    }
  }
  return true;
}

bool treeFromUrdfModel(const urdf::ModelInterface & robot_model, KDL::Tree & tree)
{
  if (!robot_model.getRoot()) {
    return false;
  }

  tree = KDL::Tree(robot_model.getRoot()->name);

  // warn if root link has inertia. KDL does not support this
  if (robot_model.getRoot()->inertial) {
    KDLP_WARN("The root link %s has an inertia specified in the URDF, but KDL does not "
      "support a root link with an inertia.  As a workaround, you can add an extra "
      "dummy link to your URDF.\n", robot_model.getRoot()->name.c_str());
  }

  // add all children
  for (size_t i = 0; i < robot_model.getRoot()->child_links.size(); i++) {
    if (!addChildrenToTree(robot_model.getRoot()->child_links[i], tree)) {
      return false;
    }
  }

  return true;
}

bool treeFromString(const std::string & xml, KDL::Tree & tree)
{
  urdf::ModelInterfaceSharedPtr robot_model = urdf::parseURDF(xml);
  if (!robot_model) {
    KDLP_WARN("Could not generate robot model\n");
    return false;
  }
  return treeFromUrdfModel(*robot_model, tree);
}

bool treeFromFile(const std::string & file, KDL::Tree & tree)
{
  urdf::ModelInterfaceSharedPtr robot_model = urdf::parseURDFFile(file);
  if (!robot_model) {
    KDLP_WARN("Could not generate robot model from file %s\n", file.c_str());
    return false;
  }
  return treeFromUrdfModel(*robot_model, tree);
}
}  // namespace kdl_parser
