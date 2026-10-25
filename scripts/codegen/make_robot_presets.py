#!/usr/bin/env python3
"""Reproduce the Z1/UR5e fixtures from pinned, locally supplied upstream checkouts.

Dependencies: hjcdik[build,codegen], scipy, pycollada, xacro==2.1.1.
No downloading; upstream meshes are read for spherization and fidelity, not bundled.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

SOURCES = {
    "z1": ("267182b8521c8d6a631bab1fe63836873237a525", "https://github.com/unitreerobotics/unitree_ros"),
    "ur5e": ("22f055da2fa7e2158254426107d1f257fd56aebb", "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(name, source):
    if name == "z1":
        package = source / "robots/z1_description"
        root = ET.parse(package / "xacro/z1.urdf").getroot()
        # Keep only URDF data, not simulator/control configuration.
        for node in list(root):
            if node.tag not in ("link", "joint", "material"):
                root.remove(node)
        # The upstream primitive model omits link01 and link05. Use the complete
        # upstream collision STL set, in its original link coordinates, for all links.
        for i in range(7):
            link = root.find(f"link[@name='link{i:02d}']")
            for collision in link.findall("collision"):
                link.remove(collision)
            col = ET.SubElement(link,"collision")
            ET.SubElement(col,"origin",xyz="0 0 0",rpy="0 0 0")
            ET.SubElement(ET.SubElement(col,"geometry"),"mesh",
                          filename=f"package://z1_description/meshes/collision/z1_Link{i:02d}.STL")
        ET.SubElement(root,"link",name="hjcd_tool")
        tool = ET.SubElement(root,"joint",name="hjcd_tool_joint",type="fixed")
        ET.SubElement(tool,"parent",link="link06")
        ET.SubElement(tool,"child",link="hjcd_tool")
        ET.SubElement(tool,"origin",xyz="0 0 0",rpy="0 0 0")
        target = "hjcd_tool_joint"
        edits = "Simulator/transmission tags removed; all link00..06 collision primitives replaced by upstream collision STLs; identity fixed tool at link06, no gripper assumed."
        args = None
    else:
        import xacro
        import xacro.substitution_args
        package = source
        def find_package(pkg):
            if pkg != "ur_description":
                raise ValueError(f"unexpected xacro package {pkg}")
            return str(package)
        xacro.substitution_args._eval_find = find_package
        args = dict(name="ur5e",ur_type="ur5e",tf_prefix="",safety_limits="false",force_abs_paths="false")
        root = ET.fromstring(xacro.process_file(str(package/"urdf/ur.urdf.xacro"),mappings=args).toxml())
        target = "wrist_3_link-tool0_fixed_joint"
        if root.find(f"joint[@name='{target}']") is None:
            candidates = [j.get("name") for j in root.findall("joint") if j.find("child").get("link")=="tool0"]
            if len(candidates)!=1: raise ValueError(f"unexpected tool0 mapping: {candidates}")
            target = candidates[0]
        edits = "Expanded stock ur5e xacro with default nominal calibration, no kinematic edits. Tool target is the fixed joint whose child is tool0."
    root.set("name",name)
    return root, package, target, edits, args


def generate(name, source, output):
    from grid_codegen.spherize import fidelity_report
    from grid_codegen.algorithms._spherize import spherize_urdf
    from grid_codegen.algorithms._collision import validate_sphere_model
    revision,url = SOURCES[name]
    actual = subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if actual != revision: raise ValueError(f"{name}: expected {revision}, found {actual}")
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():
        raise ValueError("upstream checkout must be clean")
    root,package,target,edits,args = build_model(name,source)
    resolved = copy.deepcopy(root)
    meshes = {}
    for elem in resolved.findall(".//mesh"):
        uri = elem.get("filename")
        if not uri.startswith("package://"): raise ValueError(f"unexpected mesh URI: {uri}")
        _,rel = uri[len("package://"):].split("/",1)
        path = package/rel
        if not path.is_file(): raise FileNotFoundError(path)
        meshes[uri] = sha(path)
        elem.set("filename",str(path))
    provenance = dict(source=url,commit=revision,license="BSD-3-Clause",ee_joint=target,
                      changes=edits,xacro_arguments=args,meshes=meshes,
                      generation=dict(grid_rbd="0.6.0",mesh_mode="bounded-bulge",resolution_m=.03))
    with tempfile.TemporaryDirectory(prefix="hjcd-preset-") as tmp:
        original = Path(tmp)/"source.urdf"
        sphere_path = Path(tmp)/"spheres.urdf"
        ET.ElementTree(resolved).write(original)
        spherize_urdf(original,.03,sphere_path,mesh_mode="bounded-bulge")
        validate_sphere_model(original,sphere_path)
        reports = {geometry:fidelity_report(original,sphere_path,geometry=geometry)
                   for geometry in ("collision","visual")}
        if any("unresolved" in row for rows in reports.values() for row in rows):
            raise ValueError("unresolved geometry in fidelity report")
        spheres = ET.parse(sphere_path).getroot()
        # Restore portable package URIs after evaluating against actual local files.
        for elem in spheres.findall(".//mesh"):
            elem.set("filename", "package://"+("z1_description" if name=="z1" else "ur_description")+
                     "/"+str(Path(elem.get("filename")).relative_to(package)))
    provenance["fidelity"] = reports
    comment = (f"Generated by scripts/codegen/make_robot_presets.py. {url} at {revision}. "
               f"BSD-3-Clause; see ../licenses/{name}.txt. {edits} "
               "Meshes remain upstream package URIs; HJCD uses the shipped sphere model, not unresolved meshes. "
               "Full mesh hashes and sampled fidelity (not a coverage certificate) are in ../collision/"+name+"_provenance.json.")
    for directory,filename,tree in (("urdf",name+".urdf",root),("collision",name+"_spheres.urdf",spheres)):
        path = output/directory/filename; path.parent.mkdir(parents=True,exist_ok=True)
        tree.insert(0,ET.Comment(comment))
        ET.indent(tree,space="  ")
        ET.ElementTree(tree).write(path,encoding="utf-8",xml_declaration=True)
    (output/"collision"/(name+"_provenance.json")).write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    license_path = output/"licenses"/(name+".txt")
    license_path.parent.mkdir(parents=True,exist_ok=True)
    license_path.write_text(source.joinpath("LICENSE").read_text())
    print(name,target,len(meshes),reports,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name",choices=SOURCES)
    parser.add_argument("source",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    generate(args.name,args.source.resolve(),args.output.resolve())


if __name__ == "__main__": main()
