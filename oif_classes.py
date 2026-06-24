# Copyright (C) 2010-2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import random
import math
import espressomd
from espressomd.interactions import OifLocalForces, OifGlobalForces, OifOutDirection
from .oif_utils import (
    large_number, small_epsilon, discard_epsilon, custom_str, norm,
    vec_distance, get_triangle_normal, area_triangle, angle_btw_triangles, angle_btw_vectors,
    oif_calc_stretching_force, oif_calc_bending_force,
    oif_calc_local_area_force, oif_calc_global_area_force, oif_calc_volume_force, output_vtk_lines
)

import object_in_fluid as oif


class FixedPoint:

    """
    Represents mesh points, not connected to any ESPResSo particle.

    """

    def __init__(self, pos, id):
        if not isinstance(id, int):
            raise TypeError("Id must be integer.")
        if not ((len(pos) == 3) and isinstance(pos[0], float) and isinstance(
                pos[1], float) and isinstance(pos[2], float)):
            raise TypeError("Pos must be a list of three floats.")

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.id = id
        self.neighbour_ids = []

    def get_pos(self):
        return [self.x, self.y, self.z]

    def get_id(self):
        return self.id


class PartPoint:

    """
    Represents mesh points, connected to ESPResSo particle.

    """

    # part is physical ESPResSo particle corresponding to that particular point

    def __init__(self, part, id, part_id):
        if not (isinstance(part, espressomd.particle_data.ParticleHandle)
                and isinstance(id, int) and isinstance(part_id, int)):
            raise TypeError("Arguments to PartPoint are incorrect.")
        self.part = part
        self.part_id = part_id  # because in adding bonds to the particles in OifCell
        # one needs to know the global id of the particle.
        self.id = id
        self.neighbour_ids = []

    def get_pos(self):
        return self.part.pos

    def get_part_id(self):
        return self.part_id

    def get_vel(self):
        return self.part.v

    def get_mass(self):
        return self.part.mass

    def get_type(self):
        return self.part.type

    def get_force(self):
        return self.part.f

    def set_pos(self, pos):
        self.part.pos = pos

    def set_vel(self, vel):
        self.part.v = vel

    def set_force(self, force):
        self.part.ext_force = force

    def fix(self):
        self.part.fix = [1, 1, 1]

    def unfix(self):
        self.part.fix = [0, 0, 0]


class Edge:

    """
    Represents edges in a mesh.

    """

    def __init__(self, A, B):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B]):
            TypeError("Arguments to Edge must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B

    def length(self):
        return vec_distance(self.A.get_pos(), self.B.get_pos())


class Triangle:

    """
    Represents triangles in a mesh.

    """

    def __init__(self, A, B, C):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B, C]):
            TypeError("Arguments to Triangle must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C

    def area(self):
        area = area_triangle(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos())
        return area


class Angle:

    """
    Represents angles in a mesh.

    """

    def __init__(self, A, B, C, D):
        if not all(isinstance(x, (PartPoint, FixedPoint))
                   for x in [A, B, C, D]):
            TypeError("Arguments to Angle must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def size(self):
        angle_size = angle_btw_triangles(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos(), self.D.get_pos())
        return angle_size


class ThreeNeighbors:

    """
    Represents three best spatially distributed neighbors of a point in a mesh.

    """

    def __init__(self, A, B, C):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B, C]):
            TypeError(
                "Arguments to ThreeNeighbors must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C

    def outer_normal(self):
        outer_normal = get_triangle_normal(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos())
        return outer_normal


class Mesh:

    """
    Represents a triangular mesh.

    """

    def __init__(
            self, nodes_file=None, triangles_file=None, system=None, resize=(1.0, 1.0, 1.0),
            particle_type=-1, particle_mass=1.0, normal=False, check_orientation=True):
        if (system is None) or (not isinstance(system, espressomd.System)):
            raise Exception(
                "Mesh: No system provided or wrong type given. Quitting.")

        self.system = system
        self.normal = normal
        self.nodes_file = nodes_file
        self.triangles_file = triangles_file

        self.points = []
        self.edges = []
        self.triangles = []
        self.angles = []
        self.neighbors = []
        self.ids_extremal_points = [0, 0, 0, 0, 0, 0, 0]

        if not ((nodes_file is None) or (triangles_file is None)):
            if not (isinstance(nodes_file, str)
                    and isinstance(triangles_file, str)):
                raise TypeError("Mesh: Filenames must be strings.")
            if not ((len(resize) == 3) and isinstance(resize[0], float) and isinstance(
                    resize[1], float) and isinstance(resize[2], float)):
                raise TypeError("Mesh: Pos must be a list of three floats.")
            if not isinstance(particle_type, int):
                raise TypeError("Mesh: particle_type must be integer.")
            if not isinstance(particle_mass, float):
                raise TypeError("Mesh: particle_mass must be float.")
            if not isinstance(normal, bool):
                raise TypeError("Mesh: normal must be bool.")
            if not isinstance(check_orientation, bool):
                raise TypeError("Mesh: check_orientation must be bool.")
            # reading the mesh point positions from file
            in_file = open(nodes_file, "r")
            nodes_coord = in_file.read().split("\n")
            in_file.close()
            # removes a blank line at the end of the file if there is any:
            nodes_coord = filter(None, nodes_coord)
            # here we have list of lines with triplets of strings
            for line in nodes_coord:  # extracts coordinates from the string line
                line = np.array([float(x) for x in line.split()])
                coords = np.array(resize) * line
                tmp_fixed_point = FixedPoint(coords, len(self.points))
                self.points.append(tmp_fixed_point)

            # searching for extremal points IDs
            x_min = large_number
            x_max = -large_number
            y_min = large_number
            y_max = -large_number
            z_min = large_number
            z_max = -large_number
            for tmp_fixed_point in self.points:
                coords = tmp_fixed_point.get_pos()
                if coords[0] < x_min:
                    x_min = coords[0]
                    self.ids_extremal_points[0] = tmp_fixed_point.get_id()
                if coords[0] > x_max:
                    x_max = coords[0]
                    self.ids_extremal_points[1] = tmp_fixed_point.get_id()
                if coords[1] < y_min:
                    y_min = coords[1]
                    self.ids_extremal_points[2] = tmp_fixed_point.get_id()
                if coords[1] > y_max:
                    y_max = coords[1]
                    self.ids_extremal_points[3] = tmp_fixed_point.get_id()
                if coords[2] < z_min:
                    z_min = coords[2]
                    self.ids_extremal_points[4] = tmp_fixed_point.get_id()
                if coords[2] > z_max:
                    z_max = coords[2]
                    self.ids_extremal_points[5] = tmp_fixed_point.get_id()

            # reading the triangle incidences from file
            in_file = open(triangles_file, "r")
            triangles_incid = in_file.read().split("\n")
            in_file.close()
            # removes a blank line at the end of the file if there is any:
            triangles_incid = filter(None, triangles_incid)
            for line in triangles_incid:  # extracts incidences from the string line
                incid = np.array([int(x) for x in line.split()])
                tmp_triangle = Triangle(
                    self.points[incid[0]], self.points[incid[1]], self.points[incid[2]])
                self.triangles.append(tmp_triangle)

            if check_orientation is True:
                # check whether all triangles in file had the same orientation;
                # if not, correct the orientation
                self.check_orientation()

            # creating list of edge incidences from triangle incidences
            # using temporary list of edge incidences
            tmp_edge_incidences = []
            for triangle in self.triangles:
                pa = triangle.A.id
                pb = triangle.B.id
                pc = triangle.C.id
                if ([pa, pb] not in tmp_edge_incidences) and (
                        [pb, pa] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pa, pb])
                if ([pb, pc] not in tmp_edge_incidences) and (
                        [pc, pb] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pb, pc])
                if ([pa, pc] not in tmp_edge_incidences) and (
                        [pc, pa] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pa, pc])
            for tmp_incid in tmp_edge_incidences:
                tmp_edge = Edge(
                    self.points[tmp_incid[0]], self.points[tmp_incid[1]])
                self.edges.append(tmp_edge)

            # creating list angles (former bending incidences) from triangle
            # incidences
            for edge in self.edges:
                pa = edge.A.id
                pb = edge.B.id
                pc = -1
                pd = -1
                detected = 0
                # detected = number of detected triangles with current edge common
                # Algorithm is as follows: we run over all triangles and check
                # whether two vertices are those from current edge. If we find such triangle,
                # we put the ID of the third vertex to pc and we check if the orientation pa, pb, pc is the same as
                # was in the triangle list (meaning, that we found one of the following three triples
                # in the triangle list: pa, pb, pc or pb, pc, pa or pc, pa, pb).
                # If we have the same orientation, we set orient = 1, otherwise orient = -1.
                # Then we go further looking for the second triangle.
                # The second triangle should have the opposite orientation.
                # The normal of the first triangle will be P1P2 x P1P3, of the
                # second triangle will be P2P4 x P2P3
                orient = 0
                for triangle in self.triangles:
                    # Run over all triangles and determine the two triangles
                    # with the common current edge
                    if (pa == triangle.A.id) and (pb == triangle.B.id):
                        if detected == 0:
                            # if no triangle with such edge was detected before
                            pc = triangle.C.id
                            detected = 1
                            orient = 1
                        else:
                            # if this is the second triangle with this edge,
                            # then also quit the for-loop over triangles
                            pd = triangle.C.id
                            break
                    if (pa == triangle.B.id) and (pb == triangle.C.id):
                        if detected == 0:
                            pc = triangle.A.id
                            detected = 1
                            orient = 1
                        else:
                            pd = triangle.A.id
                            break
                    if (pa == triangle.C.id) and (pb == triangle.A.id):
                        if detected == 0:
                            pc = triangle.B.id
                            detected = 1
                            orient = 1
                        else:
                            pd = triangle.B.id
                            break
                    if (pa == triangle.B.id) and (pb == triangle.A.id):
                        if detected == 0:
                            pc = triangle.C.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.C.id
                            break
                    if (pa == triangle.C.id) and (pb == triangle.B.id):
                        if detected == 0:
                            pc = triangle.A.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.A.id
                            break
                    if (pa == triangle.A.id) and (pb == triangle.C.id):
                        if detected == 0:
                            pc = triangle.B.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.B.id
                            break
                if orient == 1:
                    tmp = pd
                    pd = pc
                    pc = tmp
                tmp_angle = Angle(
                    self.points[pc], self.points[pa], self.points[pb], self.points[pd])
                self.angles.append(tmp_angle)

            # fills the list of neighbours for each mesh point
            for point in self.points:
                for edge in self.edges:
                    if edge.A.id == point.id:
                        point.neighbour_ids.append(edge.B.id)
                    if edge.B.id == point.id:
                        point.neighbour_ids.append(edge.A.id)

            # creating list of three neighbors for membrane collision
            if normal is True:
                for point in self.points:
                    tmp_neighbors = []
                    # cycle through edges and select those that contain point
                    for edge in self.edges:
                        # take an edge and copy the nodes of the edge to pa, pb
                        if edge.A.id == point.id:
                            tmp_neighbors.append(edge.B)
                        if edge.B.id == point.id:
                            tmp_neighbors.append(edge.A)
                    # create vectors to all neighbors and normalize them
                    tmp_vectors_to_neighbors = []
                    p_coords = np.array(point.get_pos())
                    for neighbor in tmp_neighbors:
                        tmp_vector = neighbor.get_pos() - p_coords
                        tmp_length = norm(tmp_vector)
                        if tmp_length < small_epsilon:
                            raise Exception("Mesh: Degenerate edge. Quitting.")
                        tmp_vector /= tmp_length
                        tmp_vectors_to_neighbors.append(tmp_vector)
                    # check all triplets of neighbors and select the one that is best spatially distributed
                    # by adding the corresponding three normalized vectors
                    # and selecting the one with smallest resultant vector
                    n_neighbors = len(tmp_neighbors)
                    min_length = large_number
                    best_neighbors = [
                        tmp_neighbors[0], tmp_neighbors[1], tmp_neighbors[2]]
                    for i in range(0, n_neighbors):
                        for j in range(i + 1, n_neighbors):
                            for k in range(j + 1, n_neighbors):
                                tmp_result_vector = tmp_vectors_to_neighbors[i] + tmp_vectors_to_neighbors[j] + \
                                    tmp_vectors_to_neighbors[k]
                                tmp_result_vector_length = norm(
                                    tmp_result_vector)
                                if tmp_result_vector_length < min_length:
                                    min_length = tmp_result_vector_length
                                    best_neighbors = [
                                        tmp_neighbors[i], tmp_neighbors[j], tmp_neighbors[k]]
                    # find one triangle that contains this point and compute
                    # its normal vector
                    for triangle in self.triangles:
                        if triangle.A.id == point.id or triangle.B.id == point.id or triangle.C.id == point.id:
                            tmp_normal_triangle = get_triangle_normal(
                                triangle.A.get_pos(), triangle.B.get_pos(),
                                triangle.C.get_pos())
                            break
                    # properly orient selected neighbors and save them to the
                    # list of neighbors
                    tmp_normal_neighbors = get_triangle_normal(
                        best_neighbors[
                            0].get_pos(), best_neighbors[1].get_pos(),
                        best_neighbors[2].get_pos())
                    tmp_length_normal_triangle = norm(tmp_normal_triangle)
                    tmp_length_normal_neighbors = norm(tmp_normal_neighbors)
                    tmp_product = np.dot(tmp_normal_triangle, tmp_normal_neighbors) / \
                        (tmp_length_normal_triangle *
                         tmp_length_normal_neighbors)
                    tmp_angle = np.arccos(tmp_product)
                    if tmp_angle > np.pi / 2.0:
                        selected_neighbors = ThreeNeighbors(
                            best_neighbors[0], best_neighbors[1], best_neighbors[2])
                    else:
                        selected_neighbors = ThreeNeighbors(
                            best_neighbors[0], best_neighbors[2], best_neighbors[1])
                    self.neighbors.append(selected_neighbors)
            else:
                for point in self.points:
                    selected_neighbors = ThreeNeighbors(point, point, point)
                    self.neighbors.append(selected_neighbors)

    def copy(self, origin=None, particle_type=-1, particle_mass=1.0,
             rotate=None):
        mesh = Mesh(system=self.system)
        mesh.ids_extremal_points = self.ids_extremal_points
        rotation = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])

        if rotate is not None:
            # variables for rotation
            ca = np.cos(rotate[0])
            sa = np.sin(rotate[0])
            cb = np.cos(rotate[1])
            sb = np.sin(rotate[1])
            cc = np.cos(rotate[2])
            sc = np.sin(rotate[2])
            rotation = np.array(
                [[cb * cc, sa * sb * cc - ca * sc, sc * sa + cc * sb * ca],
                 [cb * sc, ca * cc + sa * sb *
                  sc, sc * sb * ca - cc * sa],
                 [-sb, cb * sa, ca * cb]])
        for point in self.points:
            # PartPoints are created
            tmp_pos = point.get_pos()
            tmp_rotate_pos = np.array(point.get_pos())
            # rotation of nodes
            if rotate is not None:
                tmp_pos = rotation.dot(tmp_rotate_pos)
                tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                    tmp_pos[1]), discard_epsilon(tmp_pos[2])]
            if origin is not None:
                tmp_pos += np.array(origin)
            # to remember the global id of the ESPResSo particle
            new_part_id = len(self.system.part)
            self.system.part.add(
                pos=tmp_pos, type=particle_type, mass=particle_mass, mol_id=particle_type)
            new_part = self.system.part[new_part_id]
            new_part_point = PartPoint(new_part, len(mesh.points), new_part_id)
            new_part_point.neighbour_ids = point.neighbour_ids
            mesh.points.append(new_part_point)
        for edge in self.edges:
            new_edge = Edge(mesh.points[edge.A.id], mesh.points[edge.B.id])
            mesh.edges.append(new_edge)
        for triangle in self.triangles:
            new_triangle = Triangle(
                mesh.points[triangle.A.id], mesh.points[triangle.B.id], mesh.points[triangle.C.id])
            mesh.triangles.append(new_triangle)
        for angle in self.angles:
            new_angle = Angle(
                mesh.points[angle.A.id], mesh.points[
                    angle.B.id], mesh.points[angle.C.id],
                mesh.points[angle.D.id])
            mesh.angles.append(new_angle)
        for neighbors in self.neighbors:
            new_neighbors = ThreeNeighbors(
                mesh.points[neighbors.A.id], mesh.points[neighbors.B.id],
                mesh.points[neighbors.C.id])
            mesh.neighbors.append(new_neighbors)
        return mesh

    def check_orientation(self):
        tmp_triangle_list = []
        tmp_triangle_list_ok = []
        t_ok = None
        corrected_triangle = None
        for triangle in self.triangles:
            tmp_triangle_list.append(triangle)

        # move the first triangle to the checked and corrected list
        tmp_triangle_list_ok.append(tmp_triangle_list[0])
        tmp_triangle_list.pop(0)

        while tmp_triangle_list:
            i = 0
            while i < len(tmp_triangle_list):
                tmp_triangle = tmp_triangle_list[i]
                for correct_triangle in tmp_triangle_list_ok:
                    # check if triangles have a common edge, if so, check
                    # orientation
                    are_neighbors = True
                    if tmp_triangle.A.id == correct_triangle.A.id:
                        if tmp_triangle.B.id == correct_triangle.B.id:
                            t_ok = False  # this is situation 123 and 124
                            corrected_triangle = Triangle(
                                tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                        else:
                            if tmp_triangle.B.id == correct_triangle.C.id:
                                t_ok = True  # this is situation 123 and 142
                            else:
                                if tmp_triangle.C.id == correct_triangle.B.id:
                                    t_ok = True  # this is situation 123 and 134
                                else:
                                    if tmp_triangle.C.id == correct_triangle.C.id:
                                        t_ok = False  # this is situation 123 and 143
                                        corrected_triangle = Triangle(
                                            tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                    else:
                                        are_neighbors = False
                    else:
                        if tmp_triangle.A.id == correct_triangle.B.id:
                            if tmp_triangle.B.id == correct_triangle.C.id:
                                t_ok = False  # this is situation 123 and 412
                                corrected_triangle = Triangle(
                                    tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                            else:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    t_ok = True  # this is situation 123 and 214
                                else:
                                    if tmp_triangle.C.id == correct_triangle.C.id:
                                        t_ok = True  # this is situation 123 and 413
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.A.id:
                                            t_ok = False  # this is situation 123 and 314
                                            corrected_triangle = Triangle(
                                                tmp_triangle.A, tmp_triangle.C,
                                                tmp_triangle.B)
                                        else:
                                            are_neighbors = False
                        else:
                            if tmp_triangle.A.id == correct_triangle.C.id:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    t_ok = False  # this is situation 123 and 241
                                    corrected_triangle = Triangle(
                                        tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                else:
                                    if tmp_triangle.B.id == correct_triangle.B.id:
                                        t_ok = True  # this is situation 123 and 421
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.A.id:
                                            t_ok = True  # this is situation 123 and 341
                                        else:
                                            if tmp_triangle.C.id == correct_triangle.B.id:
                                                t_ok = False  # this is situation 123 and 431
                                                corrected_triangle = Triangle(
                                                    tmp_triangle.A, tmp_triangle.C,
                                                    tmp_triangle.B)
                                            else:
                                                are_neighbors = False
                            else:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    if tmp_triangle.C.id == correct_triangle.B.id:
                                        t_ok = False  # this is situation 123 and 234
                                        corrected_triangle = Triangle(
                                            tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.C.id:
                                            t_ok = True  # this is situation 123 and 243
                                        else:
                                            are_neighbors = False
                                else:
                                    if tmp_triangle.B.id == correct_triangle.B.id:
                                        if tmp_triangle.C.id == correct_triangle.C.id:
                                            t_ok = False  # this is situation 123 and 423
                                            corrected_triangle = Triangle(
                                                tmp_triangle.A, tmp_triangle.C,
                                                tmp_triangle.B)
                                        else:
                                            if tmp_triangle.C.id == correct_triangle.A.id:
                                                t_ok = True  # this is situation 123 and 324
                                            else:
                                                are_neighbors = False
                                    else:
                                        if tmp_triangle.B.id == correct_triangle.C.id:
                                            if tmp_triangle.C.id == correct_triangle.A.id:
                                                t_ok = False  # this is situation 123 and 342
                                                corrected_triangle = Triangle(
                                                    tmp_triangle.A, tmp_triangle.C,
                                                    tmp_triangle.B)
                                            else:
                                                if tmp_triangle.C.id == correct_triangle.B.id:
                                                    t_ok = True  # this is situation 123 and 432
                                                else:
                                                    are_neighbors = False
                                        else:
                                            are_neighbors = False
                    if are_neighbors:
                        # move the tmp_triangle to the checked and corrected
                        # list
                        if t_ok:
                            tmp_triangle_list_ok.append(tmp_triangle)
                        else:
                            tmp_triangle_list_ok.append(corrected_triangle)
                        tmp_triangle_list.pop(i)
                        break
                i += 1
        # replace triangles with checked triangles
        i = 0
        for tmp_triangle in tmp_triangle_list_ok:
            self.triangles[i] = Triangle(
                tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
            i += 1
        # all triangles now have the same orientation, check if it is correct
        tmp_volume = self.volume()
        if tmp_volume < 0:
            # opposite orientation, flip all triangles
            i = 0
            for tmp_triangle in self.triangles:
                self.triangles[i] = Triangle(
                    tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                i += 1
        return 0

    def min_edge_length(self):
        min_length = large_number
        i = 0
        for edge in self.edges:
            i = i + 1
            l = edge.length()
            if l < min_length:
                min_length = l
        if i == 0:
            raise Exception("Mesh, min_edge_length: No edges. Quitting.")
        return min_length

    def total_fluid_force(self, lbfluid, friction):
        total_force = np.array([0.0, 0.0, 0.0])
        for p in self.points:
            vel_point = p.get_vel()
            pos_point = p.get_pos()
            vel_fluid = lbfluid.get_interpolated_velocity(pos_point)
            total_force += - friction * (vel_point - vel_fluid)
        return total_force

    def max_edge_length(self):
        max_length = - large_number
        i = 0
        for edge in self.edges:
            i = i + 1
            l = edge.length()
            if l > max_length:
                max_length = l
        if i == 0:
            raise Exception("Mesh, max_edge_length: No edges. Quitting.")
        return max_length

    def gen_new_mesh(self, filename):
        output_file = open(filename, "w")
        # orig = self.get_origin()
        c0 = 0.207161
        c1 = 2.002558
        c2 = -1.122762
        r0 = 3.91
        for p in self.points:
            pos = p.get_pos()
            w = (pos[0] * pos[0] + pos[2] * pos[2]) / (1.0 * r0 * r0)
            if w > 1:
                w = 1.0
            new_z = 0.5 * r0 * np.sqrt(1 - w) * (c0 + c1 * w + c2 * w * w)
            if pos[1] < 0:
                new_z = - 1.0 * new_z
            output_file.write(str(pos[0]) + " " +
                              str(new_z) + " " + str(pos[2]) + "\n")
        output_file.close()
        return 1

    def aver_edge_length(self):
        edge_aver = 0.0
        i = 0
        for edge in self.edges:
            i = i + 1
            edge_aver = edge_aver + edge.length()
        if i == 0:
            raise Exception("Mesh, aver_edge_length: No edges. Quitting.")
        edge_aver = edge_aver / (1.0 * i)
        return edge_aver

    def check_if_spherical(self):
        radius = oif.vec_distance(self.points[0].get_pos(), [0.0, 0.0, 0.0])
        epsilon = 0.001
        spherical = True
        for point in self.points:
            if np.abs(oif.vec_distance(point.get_pos(), [0.0, 0.0, 0.0]) - radius) > epsilon:
                spherical = False
                break
        return spherical

    def stdev_edge_length(self):
        edge_aver = self.aver_edge_length()
        stdev = 0
        i = 0
        for edge in self.edges:
            stdev = stdev + (edge_aver - edge.length()) * \
                (edge_aver - edge.length())
            i = i + 1
        if i == 0:
            raise Exception("Mesh, stdev_edge_length: No edges. Quitting.")
        stdev = stdev / (1.0 * (i - 1))
        stdev = np.sqrt(stdev)
        return stdev

    def print_analysis(self):
        print("\t n_nodes: " + str(self.get_n_nodes()))
        print("\t n_triangles: " + str(self.get_n_triangles()))
        print("\t n_edges: " + str(self.get_n_edges()))
        print("\t aver_edge_length: " + str(self.aver_edge_length()))
        print("\t min_edge_length: " + str(self.min_edge_length()))
        print("\t max_edge_length: " + str(self.max_edge_length()))
        print("\t stdev_edge_length: " + str(self.stdev_edge_length()))

    def surface(self):
        surface = 0.0
        for triangle in self.triangles:
            surface += triangle.area()
        return surface

    def volume(self):
        volume = 0.0
        for triangle in self.triangles:
            tmp_normal = get_triangle_normal(
                triangle.A.get_pos(), triangle.B.get_pos(), triangle.C.get_pos())
            tmp_normal_length = norm(tmp_normal)
            tmp_sum_z_coords = 1.0 / 3.0 * \
                (triangle.A.get_pos()[2] + triangle.B.get_pos()[2] +
                 triangle.C.get_pos()[2])
            volume -= (triangle.area() * tmp_normal[2] / tmp_normal_length *
                       tmp_sum_z_coords)
        return volume

    def get_n_nodes(self):
        return len(self.points)

    def get_n_triangles(self):
        return len(self.triangles)

    def get_n_edges(self):
        return len(self.edges)

    def output_mesh_triangles(self, triangles_file=None):
        # this is useful after the mesh correction
        # output of mesh nodes can be done from OifCell (this is because their
        # position may change)
        if triangles_file is None:
            raise Exception(
                "OifMesh: No file_name provided for triangles. Quitting.")
        output_file = open(triangles_file, "w")
        for t in self.triangles:
            output_file.write(
                str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()
        return 0

    def mirror(self, mirror_x=0, mirror_y=0, mirror_z=0, out_file_name=""):
        if out_file_name == "":
            raise Exception(
                "Cell.Mirror: output meshnodes file for new mesh is missing. Quitting.")
        if mirror_x not in (0, 1) or mirror_y not in (
                0, 1) or mirror_z not in (0, 1):
            raise Exception(
                "Mesh.Mirror: for mirroring only values 0 or 1 are accepted. 1 indicates that the corresponding coordinate will be flipped.  Exiting.")
        if mirror_x + mirror_y + mirror_z > 1:
            raise Exception(
                "Mesh.Mirror: flipping allowed only for one axis. Exiting.")
        if mirror_x + mirror_y + mirror_z == 1:
            out_file = open(out_file_name, "w")
            for p in self.points:
                coor = p.get_pos()
                if mirror_x == 1:
                    coor[0] *= -1.0
                if mirror_y == 1:
                    coor[1] *= -1.0
                if mirror_z == 1:
                    coor[2] *= -1.0
                out_file.write(custom_str(coor[0]) + " " + custom_str(
                    coor[1]) + " " + custom_str(coor[2]) + "\n")
            out_file.close()
        return 0

    def exclusions_cut_off(self, cut_off):
        for p in self.points:
            for r in self.points:
                if r.id > p.id:
                    ppos = p.get_pos()
                    rpos = r.get_pos()
                    dist = np.sqrt((ppos[0] - rpos[0])*(ppos[0] - rpos[0]) + (ppos[1] - rpos[1])*(
                        ppos[1] - rpos[1]) + (ppos[2] - rpos[2])*(ppos[2] - rpos[2]))
                    if dist < cut_off:
                        self.system.part[p.id].add_exclusion(r.id)


class OifCellType:  # analogous to oif_template

    """
    Represents a template for creating elastic objects.

    """

    def __init__(
            self, nodes_file="", triangles_file="", system=None, resize=(1.0, 1.0, 1.0), ks=0.0, kslin=0.0,
            kb=0.0, kal=0.0, kag=0.0, kv=0.0, kvisc=0.0, normal=False, check_orientation=True, cell_types=None):
        if (system is None) or (not isinstance(system, espressomd.System)):
            raise Exception(
                "OifCellType: No system provided or wrong type. Quitting.")
        if (nodes_file == "") or (triangles_file == ""):
            raise Exception(
                "OifCellType: One of nodesfile or trianglesfile is missing. Quitting.")
        if not (isinstance(nodes_file, str)
                and isinstance(triangles_file, str)):
            raise TypeError("OifCellType: Filenames must be strings.")
        if not ((len(resize) == 3) and isinstance(resize[0], float) and isinstance(
                resize[1], float) and isinstance(resize[2], float)):
            raise TypeError(
                "OifCellType: Resize must be a list of three floats.")
        if not (isinstance(ks, float) and isinstance(kslin, float) and isinstance(kb, float) and isinstance(
                kal, float) and isinstance(kag, float) and isinstance(kv, float) and isinstance(kvisc, float)):
            raise TypeError("OifCellType: Elastic parameters must be floats.")
        if not isinstance(normal, bool):
            raise TypeError("OifCellType: normal must be bool.")
        if not isinstance(check_orientation, bool):
            raise TypeError("OifCellType: check_orientation must be bool.")
        if (ks != 0.0) and (kslin != 0.0):
            raise Exception(
                "OifCellType: Cannot use linear and nonlinear stretching at the same time. Quitting.")
        cell_type_exists = False
        if cell_types != None:
            for i in range(len(cell_types)):
                if ((cell_types[i].mesh.nodes_file.split('/nodes_files/')[1] == nodes_file.split('/nodes_files/')[1]) and
                    (cell_types[i].mesh.triangles_file.split('/triangles_files/')[1] == triangles_file.split('/triangles_files/')[1]) and
                    (cell_types[i].ks == ks) and
                    (cell_types[i].kslin == kslin) and
                    (cell_types[i].kb == kb) and
                    (cell_types[i].kal == kal) and
                    (cell_types[i].kag == kag) and
                    (cell_types[i].kv == kv) and
                    (cell_types[i].kvisc == kvisc) and
                    (cell_types[i].mesh.normal == normal) and
                        (cell_types[i].resize == resize)):
                    self.__dict__.update(cell_types[i].__dict__)
                    cell_type_exists = True

        if cell_type_exists is False:
            self.system = system
            self.mesh = Mesh(
                nodes_file=nodes_file, triangles_file=triangles_file, system=system, resize=resize,
                normal=normal, check_orientation=check_orientation)
            self.local_force_interactions = []
            self.resize = resize
            self.ks = ks
            self.kslin = kslin
            self.kb = kb
            self.kal = kal
            self.kag = kag
            self.kv = kv
            self.kvisc = kvisc
            self.normal = normal
            r_cut_global = 0.0
            if (ks != 0.0) or (kslin != 0.0) or (kb != 0.0) or (kal != 0.0) or (kvisc != 0.0):
                for angle in self.mesh.angles:
                    r0 = vec_distance(angle.B.get_pos(), angle.C.get_pos())
                    if r0 > r_cut_global:
                        r_cut_global = r0
                    phi = angle_btw_triangles(
                        angle.A.get_pos(), angle.B.get_pos(), angle.C.get_pos(), angle.D.get_pos())
                    area1 = area_triangle(
                        angle.A.get_pos(), angle.B.get_pos(), angle.C.get_pos())
                    area2 = area_triangle(
                        angle.D.get_pos(), angle.B.get_pos(), angle.C.get_pos())
                    tmp_local_force_inter = OifLocalForces(
                        r0=r0, ks=ks, kslin=kslin, phi0=phi, kb=kb, A01=area1, A02=area2,
                        kal=kal, kvisc=kvisc)
                    self.local_force_interactions.append(
                        [tmp_local_force_inter, [angle.A, angle.B, angle.C, angle.D]])
                    self.system.bonded_inter.add(tmp_local_force_inter)
            r_cut_global *= 5.0
            if (kag != 0.0) or (kv != 0.0):
                surface = self.mesh.surface()
                volume = self.mesh.volume()
                self.global_force_interaction = OifGlobalForces(
                    A0_g=surface, ka_g=kag, V0=volume, kv=kv, r_cut=r_cut_global)
                self.system.bonded_inter.add(self.global_force_interaction)

            if cell_types is not None:
                cell_types.append(self)

    def print_info(self):
        print("\nThe following OifCellType was created: ")
        print("\t nodes_file: " + self.mesh.nodes_file)
        print("\t triangles_file: " + self.mesh.triangles_file)
        print("\t n_nodes: " + str(self.mesh.get_n_nodes()))
        print("\t n_triangles: " + str(self.mesh.get_n_triangles()))
        print("\t n_edges: " + str(self.mesh.get_n_edges()))
        print("\t ks: " + custom_str(self.ks))
        print("\t kslin: " + custom_str(self.kslin))
        print("\t kb: " + custom_str(self.kb))
        print("\t kal: " + custom_str(self.kal))
        print("\t kag: " + custom_str(self.kag))
        print("\t kv: " + custom_str(self.kv))
        print("\t kvisc: " + custom_str(self.kvisc))
        print("\t normal: " + str(self.normal))
        print("\t resize: " + str(self.resize))
        print(" ")

    def suggest_LBgamma(self, visc=None, dens=None):
        if not (isinstance(visc, float) and isinstance(dens, float)):
            raise Exception(
                "OifCellType: viscosity or density must be real numbers in suggest_LBgamma. Quitting.")
        noNodes = self.mesh.get_n_nodes()
        surface = self.mesh.surface()

        LBgamma = (393.0 / (1.0 * noNodes)) * np.sqrt(surface / 201.0619) * (
            (5.6 - 1.82) / (5.853658537 - 1.5) * (visc - 1.5) + (10 - 1.82) / (6 - 1.025) * (
                dens - 1.025) + 1.82)
        return LBgamma


class OifCell:

    """
    Represents a concrete elastic object.

    """

    def __init__(self, cell_type=None, origin=None, particle_type=None,
                 particle_mass=1.0, rotate=None, exclusion_neighbours=True, inner_particles=False, rotation_ids=()):
        if (cell_type is None) or (not isinstance(cell_type, OifCellType)):
            raise Exception(
                "OifCell: No cellType provided or wrong type. Quitting.")
        if (origin is None) or \
                (not ((len(origin) == 3) and isinstance(origin[0], float) and isinstance(origin[1], float) and isinstance(origin[2], float))):
            raise TypeError(
                "OifCell: origin must be list of three floats. Quitting.")
        if (particle_type is None) or (not isinstance(particle_type, int)):
            raise Exception(
                "OifCell: No particle_type specified or wrong type. Quitting.")
        if not isinstance(particle_mass, float):
            raise Exception("OifCell: particle_mass must be float.")
        if (rotate is not None) and not ((len(rotate) == 3) and isinstance(
                rotate[0], float) and isinstance(rotate[1], float) and isinstance(rotate[2], float)):
            raise TypeError("OifCell: rotate must be list of three floats.")
        if not isinstance(exclusion_neighbours, bool):
            raise Exception("OifCell: exclusion_neighbours must be bool.")

        self.cell_type = cell_type
        self.cell_type.system.max_oif_objects = self.cell_type.system.max_oif_objects + 1
        self.mesh = cell_type.mesh.copy(
            origin=origin, particle_type=particle_type, particle_mass=particle_mass, rotate=rotate)
        self.particle_mass = particle_mass
        self.particle_type = particle_type
        self.origin = origin
        self.rotate = rotate
        self.rot_ids = []
        self.rot_init_pos = []
        rot_ids_ok = 1

        if inner_particles:
            self.inner_particles = OifInnerParticles(self)

        for it in rotation_ids:
            if not isinstance(it, int):
                rot_ids_ok = 0
            it = int(it)
            if it < 0 or it >= len(self.mesh.points):
                rot_ids_ok = 0
        if rot_ids_ok == 0:
            raise Exception(
                "OifCell: rotation_ids must be a tuple of integer valued ids between 0 and the number of mesh points.")
        self.set_rotation(rotation_ids)

        for inter in self.cell_type.local_force_interactions:
            esp_inter = inter[0]
            points = inter[1]
            n_points = len(points)
            if n_points == 2:
                p0 = self.mesh.points[
                    points[0].id]  # Getting PartPoints from id's of FixedPoints
                p1 = self.mesh.points[points[1].id]
                p0.part.add_bond((esp_inter, p1.part_id))
            if n_points == 3:
                p0 = self.mesh.points[points[0].id]
                p1 = self.mesh.points[points[1].id]
                p2 = self.mesh.points[points[2].id]
                p0.part.add_bond((esp_inter, p1.part_id, p2.part_id))
            if n_points == 4:
                p0 = self.mesh.points[points[0].id]
                p1 = self.mesh.points[points[1].id]
                p2 = self.mesh.points[points[2].id]
                p3 = self.mesh.points[points[3].id]
                p1.part.add_bond(
                    (esp_inter, p0.part_id, p2.part_id, p3.part_id))

        if (self.cell_type.kag != 0.0) or (self.cell_type.kv != 0.0):
            for triangle in self.mesh.triangles:
                triangle.A.part.add_bond(
                    (self.cell_type.global_force_interaction, triangle.B.part_id,
                     triangle.C.part_id))

        # setting the out_direction interaction for membrane collision
        if self.cell_type.mesh.normal is True:
            tmp_out_direction_interaction = OifOutDirection()
            # this interaction could be just one for all objects, but here it
            # is created multiple times
            self.cell_type.system.bonded_inter.add(
                tmp_out_direction_interaction)
            for p in self.mesh.points:
                p.part.add_bond(
                    (tmp_out_direction_interaction, self.mesh.neighbors[
                        p.id].A.part_id,
                     self.mesh.neighbors[p.id].B.part_id, self.mesh.neighbors[p.id].C.part_id))

        if exclusion_neighbours is True:
            for point in self.mesh.points:
                excl = []
                for id in point.neighbour_ids:
                    if id > point.id:
                        if id not in excl:
                            excl.append(id)
                    for idd in self.mesh.points[id].neighbour_ids:
                        if idd > point.id:
                            if idd not in excl:
                                excl.append(idd)

                for exclusion in excl:
                    self.cell_type.system.part[point.part_id].add_exclusion(
                        self.mesh.points[exclusion].part_id)

                excl = []
                for id in point.neighbour_ids:
                    if id != point.id:
                        if id not in excl:
                            excl.append(id)
                    for idd in self.mesh.points[id].neighbour_ids:
                        if idd != point.id:
                            if idd not in excl:
                                excl.append(idd)
                point.exclusions = excl

    def check_inner_particles_distance(self):
        if len(self.inner_particles.particles) > 0:
            for part in self.inner_particles.particles:
                if self.diameter() > vec_distance(part.pos, self.get_origin()):
                    return True
                else:
                    return False

    @classmethod
    def load_cell(cls, directory, system, rotate=None, exclusion_neighbours=True, inner_particles=True, cell_types=None):
        import json
        f = open(directory + '/data.json', )
        data = json.load(f)

        json_cell_type = data['cell_type']
        type_of_cell = oif.OifCellType(nodes_file=directory + '/nodes_files/' + json_cell_type['nodes_file'],
                                       triangles_file=directory + '/triangles_files/' +
                                       json_cell_type['triangles_file'],
                                       check_orientation=False,
                                       system=system,
                                       ks=json_cell_type['ks'],
                                       kslin=json_cell_type['kslin'],
                                       kb=json_cell_type['kb'],
                                       kal=json_cell_type['kal'],
                                       kag=json_cell_type['kag'],
                                       kv=json_cell_type['kv'],
                                       kvisc=json_cell_type['kvisc'],
                                       normal=json_cell_type['normal'],
                                       resize=json_cell_type['resize'],
                                       cell_types=cell_types)

        cell = cls(cell_type=type_of_cell,
                   particle_type=data['particle_type'],
                   origin=data['origin'],
                   particle_mass=data['particle_mass'],
                   rotate=rotate,
                   exclusion_neighbours=exclusion_neighbours,
                   inner_particles=inner_particles)

        positions = data['positions']

        for i, point in enumerate(cell.mesh.points):
            point.set_pos([cell.origin[0] + positions[i][0], cell.origin[1] +
                          positions[i][1], cell.origin[2] + positions[i][2]])

        return cell

    def set_rotation(self, ids=[]):
        if not ids:
            self.rot_ids = [-1, -1, -1, -1, -1, -1]
            # searching for extremal points IDs
            x_min = large_number
            x_max = -large_number
            y_min = large_number
            y_max = -large_number
            z_min = large_number
            z_max = -large_number
            for tmp_part_point in self.cell_type.mesh.points:
                coords = tmp_part_point.get_pos()
                if coords[0] < x_min:
                    x_min = coords[0]
                    self.rot_ids[0] = tmp_part_point.get_id()
                if coords[0] > x_max:
                    x_max = coords[0]
                    self.rot_ids[1] = tmp_part_point.get_id()
                if coords[1] < y_min:
                    y_min = coords[1]
                    self.rot_ids[2] = tmp_part_point.get_id()
                if coords[1] > y_max:
                    y_max = coords[1]
                    self.rot_ids[3] = tmp_part_point.get_id()
                if coords[2] < z_min:
                    z_min = coords[2]
                    self.rot_ids[4] = tmp_part_point.get_id()
                if coords[2] > z_max:
                    z_max = coords[2]
                    self.rot_ids[5] = tmp_part_point.get_id()
        else:
            ok = 1
            for it in ids:
                if not isinstance(it, int):
                    ok = 0
                it = int(it)
                if it < 0 or it >= len(self.mesh.points):
                    ok = 0
            if ok == 0:
                raise Exception(
                    "OifCell: set_rotation: rotation_ids must be a tuple of integer valued ids between 0 and the number of mesh points.")
            self.rot_ids = ids

        self.rot_init_pos = []
        orig = self.get_origin()
        for it in self.rot_ids:
            pos = list(self.mesh.points[it].get_pos())
            for ii in range(0, 3):
                pos[ii] = pos[ii] - orig[ii]
            self.rot_init_pos.append(pos)

    def get_rotation_angles(self):
        i = 0
        ang = []
        orig = self.get_origin()
        for it in self.rot_ids:
            vec_cur = list(self.mesh.points[it].get_pos())
            vec_init = self.rot_init_pos[i]
            for ii in range(0, 3):
                vec_cur[ii] = vec_cur[ii] - orig[ii]
            ang.append(angle_btw_vectors(vec_init, vec_cur))
            i = i + 1
        return ang

    def get_rotation_positions(self):
        pos = []
        orig = self.get_origin()
        for it in self.rot_ids:
            tmp = list(self.mesh.points[it].get_pos())
            for ii in range(0, 3):
                tmp[ii] = tmp[ii] - orig[ii]
            pos.append(tmp)
        return pos

    def rotate_cell(self, rotate=(0.0, 0.0, 0.0)):  # EXPERIMENTAL FUNCTION ADDED BY MICHAL
        # move to (0,0,0)
        origin = np.array(self.get_origin())
        self.set_origin()

        # create rotation matrix
        ca = np.cos(rotate[0])
        sa = np.sin(rotate[0])
        cb = np.cos(rotate[1])
        sb = np.sin(rotate[1])
        cc = np.cos(rotate[2])
        sc = np.sin(rotate[2])
        rotation = np.array(
            [[cb * cc, sa * sb * cc - ca * sc, sc * sa + cc * sb * ca],
             [cb * sc, ca * cc + sa * sb *
              sc, sc * sb * ca - cc * sa],
             [-sb, cb * sa, ca * cb]])

        # rotate
        for point in self.mesh.points:
            tmp_rotate_pos = np.array(point.get_pos())
            tmp_pos = rotation.dot(tmp_rotate_pos)
            tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                tmp_pos[1]), discard_epsilon(tmp_pos[2])]
            point.set_pos(tmp_pos)

        if self.inner_particles:
            for point in self.inner_particles.particles:
                tmp_rotate_pos = np.array(point.pos - origin)
                tmp_pos = rotation.dot(tmp_rotate_pos)
                tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                    tmp_pos[1]), discard_epsilon(tmp_pos[2])]
                point.pos = tmp_pos + origin

        # move back
        self.set_origin(origin)

    def get_origin(self):
        center = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            center += p.get_pos()
        return center / len(self.mesh.points)

    def set_origin(self, new_origin=(0.0, 0.0, 0.0)):
        old_origin = self.get_origin()
        for p in self.mesh.points:
            new_position = p.get_pos() - old_origin + new_origin
            p.set_pos(new_position)

    def get_approx_origin(self):
        approx_center = np.array([0.0, 0.0, 0.0])
        for id in self.mesh.ids_extremal_points:
            approx_center += self.mesh.points[id].get_pos()
        return approx_center / len(self.mesh.ids_extremal_points)

    def get_origin_folded(self):
        origin = self.get_origin()
        return np.mod(origin, self.cell_type.system.box_l)

    def get_velocity(self):
        velocity = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            velocity += p.get_vel()
        return velocity / len(self.mesh.points)

    def set_velocity(self, new_velocity=(0.0, 0.0, 0.0)):
        for p in self.mesh.points:
            p.set_vel(new_velocity)

    def pos_bounds(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for p in self.mesh.points:
            coords = p.get_pos()
            if coords[0] < x_min:
                x_min = coords[0]
            if coords[0] > x_max:
                x_max = coords[0]
            if coords[1] < y_min:
                y_min = coords[1]
            if coords[1] > y_max:
                y_max = coords[1]
            if coords[2] < z_min:
                z_min = coords[2]
            if coords[2] > z_max:
                z_max = coords[2]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def point_bound(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for p in self.mesh.points:
            coords = p.get_pos()
            if coords[0] < x_min:
                x_min = coords[0]
                x_min_point = p
            if coords[0] > x_max:
                x_max = coords[0]
                x_max_point = p
            if coords[1] < y_min:
                y_min = coords[1]
                y_min_point = p
            if coords[1] > y_max:
                y_max = coords[1]
                y_max_point = p
            if coords[2] < z_min:
                z_min = coords[2]
                z_min_point = p
            if coords[2] > z_max:
                z_max = coords[2]
                z_max_point = p
        return [x_min_point, x_max_point, y_min_point, y_max_point, z_min_point, z_max_point]

    def surface(self):
        return self.mesh.surface()

    def print_mesh_analysis(self):
        return self.mesh.print_analysis()

    def min_edge_length(self):
        return self.mesh.min_edge_length()

    def total_fluid_force(self, lbfluid, friction):
        return self.mesh.total_fluid_force(lbfluid, friction)

    def max_edge_length(self):
        return self.mesh.max_edge_length()

    def aver_edge_length(self):
        return self.mesh.aver_edge_length()

    def stdev_edge_length(self):
        return self.mesh.stdev_edge_length()

    def volume(self):
        return self.mesh.volume()

    def suggest_LBgamma(self, visc=None, dens=None):
        return self.cell_type.suggest_LBgamma(visc=visc, dens=dens)

    def diameter(self):
        max_distance = 0.0
        n_points = len(self.mesh.points)
        for i in range(0, n_points):
            for j in range(i + 1, n_points):
                p1 = self.mesh.points[i].get_pos()
                p2 = self.mesh.points[j].get_pos()
                tmp_dist = vec_distance(p1, p2)
                if tmp_dist > max_distance:
                    max_distance = tmp_dist
        return max_distance

    def get_n_nodes(self):
        return self.mesh.get_n_nodes()

    def set_force(self, new_force=(0.0, 0.0, 0.0)):
        for p in self.mesh.points:
            p.set_force(new_force)

    def fix(self):
        for p in self.mesh.points:
            p.fix()

    def unfix(self):
        for p in self.mesh.points:
            p.unfix()

    def output_vtk_pos(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting")
        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)
        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos()
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()

    def output_vtk_point_data(self, file_name=None, point_id=None, data_type=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting")
        if point_id is None:
            raise Exception(
                "OifCell: No point_id provided for vtk output. Quitting")
        if point_id < 0 or point_id >= len(self.mesh.points):
            raise Exception(
                "OifCell: point_id is negative or larger than number of mesh points. Quitting")
        if data_type != "neighbours" and data_type != "exclusions":
            raise Exception(
                "OifCell: Wrong data_type provided, exclusions or neighbours allowed. Quitting")

        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)
        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos()
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.write("POINT_DATA " + str(n_points) + "\n")
        output_file.write("SCALARS neighbours float 1\n")
        output_file.write("LOOKUP_TABLE default\n")
        tmp_point = self.mesh.points[point_id]
        for p in self.mesh.points:
            if p.id == point_id:
                output_file.write("1.5\n")
            else:
                if p.id in tmp_point.neighbour_ids:
                    output_file.write("1.0\n")
                else:
                    if (p.id in tmp_point.exclusions) and (data_type == "exclusions"):
                        output_file.write("0.5\n")
                    else:
                        output_file.write("0.0\n")
        output_file.close()

    def output_vtk_pos_folded(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting.")
        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)

        # get coordinates of the origin
        center = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            center += p.get_pos()
        center /= len(self.mesh.points)
        center_folded = np.floor(center / self.cell_type.system.box_l)
        # this gives how many times the origin is folded in all three
        # directions

        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos() - center_folded * self.cell_type.system.box_l
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()

    def append_point_data_to_vtk(self, file_name=None, data_name=None,
                                 data=None, first_append=None):
        if file_name is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No file_name provided. Quitting.")
        if data is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No data provided. Quitting.")
        if data_name is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No data_name provided. Quitting.")
        if first_append is None:
            raise Exception("OifCell: append_point_data_to_vtk: Need to know whether this is the first data list to be "
                            "appended for this file. Quitting.")
        n_points = self.get_n_nodes()
        if len(data) != n_points:
            raise Exception(
                "OifCell: append_point_data_to_vtk: Number of data points does not match number of mesh points. Quitting.")
        output_file = open(file_name, "a")
        if first_append is True:
            output_file.write("POINT_DATA " + str(n_points) + "\n")
        output_file.write("SCALARS " + data_name + " float 1\n")
        output_file.write("LOOKUP_TABLE default\n")
        for p in self.mesh.points:
            output_file.write(str(data[p.id]) + "\n")
        output_file.close()

    def output_raw_data(self, file_name=None, data=None):
        if file_name is None:
            raise Exception(
                "OifCell: output_raw_data: No file_name provided. Quitting.")
        if data is None:
            raise Exception(
                "OifCell: output_raw_data: No data provided. Quitting.")
        n_points = self.get_n_nodes()
        if len(data) != n_points:
            raise Exception(
                "OifCell: output_raw_data: Number of data points does not match number of mesh points. Quitting.")
        output_file = open(file_name, "w")
        for p in self.mesh.points:
            output_file.write(" ".join(map(str, data[p.id])) + "\n")
        output_file.close()

    def output_mesh_points(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for mesh nodes output. Quitting.")
        output_file = open(file_name, "w")
        center = self.get_origin()
        for p in self.mesh.points:
            coords = p.get_pos() - center
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.close()

    def set_mesh_points(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for set_mesh_points. Quitting.")
        center = self.get_origin()
        n_points = self.get_n_nodes()

        in_file = open(file_name, "r")
        nodes_coord = in_file.read().split("\n")
        in_file.close()
        # removes a blank line at the end of the file if there is any:
        nodes_coord = list(filter(None, nodes_coord))
        # here we have list of lines with triplets of strings
        if len(nodes_coord) != n_points:
            raise Exception("OifCell: Mesh nodes not set to new positions: "
                            "number of lines in the file does not equal number of Cell nodes. Quitting.")
        else:
            i = 0
            for line in nodes_coord:  # extracts coordinates from the string line
                line = line.split()
                new_position = np.array(line).astype(np.float) + center
                self.mesh.points[i].set_pos(new_position)
                i += 1

    def print_info(self):
        print("\nThe following OifCell was created: ")
        print("\t particle_mass: " + custom_str(self.particle_mass))
        print("\t particle_type: " + str(self.particle_type))
        print("\t rotate: " + str(self.rotate))
        print("\t origin: " + str(self.origin[0]) + " " + str(
            self.origin[1]) + " " + str(self.origin[2]))

    def get_mesh_particles_range(self):
        return str(self.mesh.points[0].get_part_id()) + " - " + str(self.mesh.points[-1].get_part_id())

    def get_inner_particles_range(self):
        if (len(self.inner_particles.particles) == 0):
            return str(0) + " - " + str(0)
        else:
            return str(self.inner_particles.particles[0].id) + " - " + str(self.inner_particles.particles[-1].id)

    def elastic_forces(
            self, el_forces=(0, 0, 0, 0, 0, 0), f_metric=(0, 0, 0, 0, 0, 0), vtk_file=None,
            raw_data_file=None):
        # the order of parameters in elastic_forces and in f_metric is as follows (ks, kb, kal, kag, kv, total)
        # vtk_file means that a vtk file for visualisation of elastic forces will be written
        # raw_data_file means that just the elastic forces will be written into
        # the output file

        stretching_forces_list = []
        bending_forces_list = []
        local_area_forces_list = []
        global_area_forces_list = []
        volume_forces_list = []
        elastic_forces_list = []
        stretching_forces_norms_list = []
        bending_forces_norms_list = []
        local_area_forces_norms_list = []
        global_area_forces_norms_list = []
        volume_forces_norms_list = []
        elastic_forces_norms_list = []
        ks_f_metric = 0.0
        kb_f_metric = 0.0
        kal_f_metric = 0.0
        kag_f_metric = 0.0
        kv_f_metric = 0.0
        total_f_metric = 0.0

        for i in range(0, 6):
            if (el_forces[i] != 0) and (el_forces[i] != 1):
                raise Exception(
                    "OifCell: elastic_forces: Incorrect argument. el_forces has to be a sixtuple of 0s and 1s, "
                    "specifying which elastic forces will be calculated. The order in the sixtuple is (ks, kb, "
                    "kal, kag, kv, total).")
        for i in range(0, 6):
            if (f_metric[i] != 0) and (f_metric[i] != 1):
                raise Exception(
                    "OifCell: elastic_forces: Incorrect argument. f_metric has to be a sixtuple of 0s and 1s, "
                    "specifying which f_metric will be calculated. The order in the sixtuple is (ks, kb, kal, "
                    "kag, kv, total)")
        # calculation of stretching forces and f_metric
        if (el_forces[0] == 1) or (el_forces[5] == 1) or (
                f_metric[0] == 1) or (f_metric[5] == 1):
            # initialize list
            stretching_forces_list = []
            for p in self.mesh.points:
                stretching_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses edges, but results are stored for nodes
            for e in self.mesh.edges:
                a_current_pos = e.A.get_pos()
                b_current_pos = e.B.get_pos()
                a_orig_pos = self.cell_type.mesh.points[e.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[e.B.id].get_pos()
                current_dist = e.length()
                orig_dist = vec_distance(a_orig_pos, b_orig_pos)
                tmp_stretching_force = oif_calc_stretching_force(
                    self.cell_type.ks, a_current_pos, b_current_pos,
                    orig_dist, current_dist)
                stretching_forces_list[e.A.id] += tmp_stretching_force
                stretching_forces_list[e.B.id] -= tmp_stretching_force
            # calculation of stretching f_metric, if needed
            if f_metric[0] == 1:
                ks_f_metric = 0.0
                for p in self.mesh.points:
                    ks_f_metric += norm(stretching_forces_list[p.id])

        # calculation of bending forces and f_metric
        if (el_forces[1] == 1) or (el_forces[5] == 1) or (
                f_metric[1] == 1) or (f_metric[5] == 1):
            # initialize list
            bending_forces_list = []
            for p in self.mesh.points:
                bending_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses bending incidences, but results are stored for
            # nodes
            for angle in self.mesh.angles:
                a_current_pos = angle.A.get_pos()
                b_current_pos = angle.B.get_pos()
                c_current_pos = angle.C.get_pos()
                d_current_pos = angle.D.get_pos()
                a_orig_pos = self.cell_type.mesh.points[angle.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[angle.B.id].get_pos()
                c_orig_pos = self.cell_type.mesh.points[angle.C.id].get_pos()
                d_orig_pos = self.cell_type.mesh.points[angle.D.id].get_pos()
                current_angle = angle.size()
                orig_angle = angle_btw_triangles(
                    a_orig_pos, b_orig_pos, c_orig_pos, d_orig_pos)
                tmp_bending_forces = oif_calc_bending_force(
                    self.cell_type.kb, a_current_pos, b_current_pos, c_current_pos,
                    d_current_pos, orig_angle, current_angle)
                tmp_bending_force1 = np.array(
                    [tmp_bending_forces[0], tmp_bending_forces[1], tmp_bending_forces[2]])
                tmp_bending_force2 = np.array(
                    [tmp_bending_forces[3], tmp_bending_forces[4], tmp_bending_forces[5]])
                bending_forces_list[angle.A.id] += tmp_bending_force1
                bending_forces_list[angle.B.id] -= 0.5 * \
                    tmp_bending_force1 + 0.5 * tmp_bending_force2
                bending_forces_list[angle.C.id] -= 0.5 * \
                    tmp_bending_force1 + 0.5 * tmp_bending_force2
                bending_forces_list[angle.D.id] += tmp_bending_force2
            # calculation of bending f_metric, if needed
            if f_metric[1] == 1:
                kb_f_metric = 0.0
                for p in self.mesh.points:
                    kb_f_metric += norm(bending_forces_list[p.id])

        # calculation of local area forces and f_metric
        if (el_forces[2] == 1) or (el_forces[5] == 1) or (
                f_metric[2] == 1) or (f_metric[5] == 1):
            # initialize list
            local_area_forces_list = []
            for p in self.mesh.points:
                local_area_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                a_orig_pos = self.cell_type.mesh.points[t.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[t.B.id].get_pos()
                c_orig_pos = self.cell_type.mesh.points[t.C.id].get_pos()
                current_area = t.area()
                orig_area = area_triangle(a_orig_pos, b_orig_pos, c_orig_pos)
                tmp_local_area_forces = oif_calc_local_area_force(
                    self.cell_type.kal, a_current_pos, b_current_pos,
                    c_current_pos, orig_area, current_area)
                local_area_forces_list[t.A.id] += np.array(
                    [tmp_local_area_forces[0], tmp_local_area_forces[1],
                     tmp_local_area_forces[2]])
                local_area_forces_list[t.B.id] += np.array(
                    [tmp_local_area_forces[3], tmp_local_area_forces[4],
                     tmp_local_area_forces[5]])
                local_area_forces_list[t.C.id] += np.array(
                    [tmp_local_area_forces[6], tmp_local_area_forces[7],
                     tmp_local_area_forces[8]])

            # calculation of local area f_metric, if needed
            if f_metric[2] == 1:
                kal_f_metric = 0.0
                for p in self.mesh.points:
                    kal_f_metric += norm(local_area_forces_list[p.id])

        # calculation of global area forces and f_metric
        if (el_forces[3] == 1) or (el_forces[5] == 1) or (
                f_metric[3] == 1) or (f_metric[5] == 1):
            # initialize list
            global_area_forces_list = []
            for p in self.mesh.points:
                global_area_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                current_surface = self.mesh.surface()
                orig_surface = self.cell_type.mesh.surface()
                tmp_global_area_forces = oif_calc_global_area_force(
                    self.cell_type.kag, a_current_pos, b_current_pos,
                    c_current_pos, orig_surface, current_surface)
                global_area_forces_list[t.A.id] += np.array(
                    [tmp_global_area_forces[0], tmp_global_area_forces[1],
                     tmp_global_area_forces[2]])
                global_area_forces_list[t.B.id] += np.array(
                    [tmp_global_area_forces[3], tmp_global_area_forces[4],
                     tmp_global_area_forces[5]])
                global_area_forces_list[t.C.id] += np.array(
                    [tmp_global_area_forces[6], tmp_global_area_forces[7],
                     tmp_global_area_forces[8]])
            # calculation of global area f_metric, if needed
            if f_metric[3] == 1:
                kag_f_metric = 0.0
                for p in self.mesh.points:
                    kag_f_metric += norm(global_area_forces_list[p.id])

        # calculation of volume forces and f_metric
        if (el_forces[4] == 1) or (el_forces[5] == 1) or (
                f_metric[4] == 1) or (f_metric[5] == 1):
            # initialize list
            volume_forces_list = []
            for p in self.mesh.points:
                volume_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                current_volume = self.mesh.volume()
                orig_volume = self.cell_type.mesh.volume()
                tmp_volume_force = oif_calc_volume_force(
                    self.cell_type.kv, a_current_pos, b_current_pos, c_current_pos,
                    orig_volume, current_volume)
                volume_forces_list[t.A.id] += tmp_volume_force
                volume_forces_list[t.B.id] += tmp_volume_force
                volume_forces_list[t.C.id] += tmp_volume_force
            # calculation of volume f_metric, if needed
            if f_metric[4] == 1:
                kv_f_metric = 0.0
                for p in self.mesh.points:
                    kv_f_metric += norm(volume_forces_list[p.id])

        # calculation of total elastic forces and f_metric
        if (el_forces[5] == 1) or (f_metric[5] == 1):
            elastic_forces_list = []
            for p in self.mesh.points:
                total_elastic_forces = stretching_forces_list[p.id] + bending_forces_list[p.id] + \
                    local_area_forces_list[p.id] + global_area_forces_list[p.id] + \
                    volume_forces_list[p.id]
                elastic_forces_list.append(total_elastic_forces)
            # calculation of total f_metric, if needed
            if f_metric[5] == 1:
                total_f_metric = 0.0
                for p in self.mesh.points:
                    total_f_metric += norm(elastic_forces_list[p.id])

        # calculate norms of resulting forces
        if sum(el_forces) != 0:
            if el_forces[0] == 1:
                stretching_forces_norms_list = []
                for p in self.mesh.points:
                    stretching_forces_norms_list.append(
                        norm(stretching_forces_list[p.id]))
            if el_forces[1] == 1:
                bending_forces_norms_list = []
                for p in self.mesh.points:
                    bending_forces_norms_list.append(
                        norm(bending_forces_list[p.id]))
            if el_forces[2] == 1:
                local_area_forces_norms_list = []
                for p in self.mesh.points:
                    local_area_forces_norms_list.append(
                        norm(local_area_forces_list[p.id]))
            if el_forces[3] == 1:
                global_area_forces_norms_list = []
                for p in self.mesh.points:
                    global_area_forces_norms_list.append(
                        norm(global_area_forces_list[p.id]))
            if el_forces[4] == 1:
                volume_forces_norms_list = []
                for p in self.mesh.points:
                    volume_forces_norms_list.append(
                        norm(volume_forces_list[p.id]))
            if el_forces[5] == 1:
                elastic_forces_norms_list = []
                for p in self.mesh.points:
                    elastic_forces_norms_list.append(
                        norm(elastic_forces_list[p.id]))

        # output vtk (folded)
        if vtk_file is not None:
            if el_forces == (0, 0, 0, 0, 0, 0):
                raise Exception("OifCell: elastic_forces: The option elastic_forces was not used. "
                                "Nothing to output to vtk file.")
            self.output_vtk_pos_folded(vtk_file)
            first = True
            if el_forces[0] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="ks_f_metric",
                    data=stretching_forces_norms_list, first_append=first)
                first = False
            if el_forces[1] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kb_f_metric",
                    data=bending_forces_norms_list, first_append=first)
                first = False
            if el_forces[2] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kal_f_metric",
                    data=local_area_forces_norms_list, first_append=first)
                first = False
            if el_forces[3] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kag_f_metric",
                    data=global_area_forces_norms_list, first_append=first)
                first = False
            if el_forces[4] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kav_f_metric",
                    data=volume_forces_norms_list, first_append=first)
                first = False
            if el_forces[5] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="total_f_metric",
                    data=elastic_forces_norms_list, first_append=first)
                first = False

        # output raw data
        if raw_data_file is not None:
            if sum(el_forces) != 1:
                raise Exception("OifCell: elastic_forces: Only one type of elastic forces can be written into one "
                                "raw_data_file. If you need several, please call OifCell.elastic_forces multiple times - "
                                "once per elastic force.")
            if el_forces[0] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=stretching_forces_list)
            if el_forces[1] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=bending_forces_list)
            if el_forces[2] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=local_area_forces_list)
            if el_forces[3] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=global_area_forces_list)
            if el_forces[4] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=volume_forces_list)
            if el_forces[5] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=elastic_forces_list)

        # return f_metric
        if f_metric[0] + f_metric[1] + f_metric[2] + \
                f_metric[3] + f_metric[4] + f_metric[5] > 0:
            results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if f_metric[0] == 1:
                results[0] = ks_f_metric
            if f_metric[1] == 1:
                results[1] = kb_f_metric
            if f_metric[2] == 1:
                results[2] = kal_f_metric
            if f_metric[3] == 1:
                results[3] = kag_f_metric
            if f_metric[4] == 1:
                results[4] = kv_f_metric
            if f_metric[5] == 1:
                results[5] = total_f_metric
            return results
        else:
            return 0

    def edge_length_analysis(self):
        """
        Compares the lengths of current vs relaxed edge lengths, returning minimal ratio, averaged ratio and maximal ratio.
        """
        min_val = 100000
        max_val = 0
        i = 0
        aver_val = 0
        for e in self.mesh.edges:
            a_orig_pos = self.cell_type.mesh.points[e.A.id].get_pos()
            b_orig_pos = self.cell_type.mesh.points[e.B.id].get_pos()
            current_dist = e.length()
            orig_dist = vec_distance(a_orig_pos, b_orig_pos)
            val = (1.0*current_dist)/(1.0*orig_dist)
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            aver_val += val
            i += 1
        aver_val = aver_val/(1.0*i)
        result = np.array([min_val, aver_val, max_val])
        return result

    def bending_angle_analysis(self):
        """
        Compares the angles btw triangles of current vs relaxed angles, returning minimal ratio, averaged ratio and maximal ratio.
        """
        min_val = 100000
        max_val = 0
        i = 0
        aver_val = 0
        for angle in self.mesh.angles:
            a_orig_pos = self.cell_type.mesh.points[angle.A.id].get_pos()
            b_orig_pos = self.cell_type.mesh.points[angle.B.id].get_pos()
            c_orig_pos = self.cell_type.mesh.points[angle.C.id].get_pos()
            d_orig_pos = self.cell_type.mesh.points[angle.D.id].get_pos()
            current_angle = angle.size()
            orig_angle = angle_btw_triangles(
                a_orig_pos, b_orig_pos, c_orig_pos, d_orig_pos)
            val = (1.0*current_angle)/(1.0*orig_angle)
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            aver_val += val
            i += 1
        aver_val = aver_val/(1.0*i)
        result = np.array([min_val, aver_val, max_val])
        return result

    def create_neighbours(self):
        excl = []
        for p in self.mesh.points:
            neigh = []
            for e in self.mesh.edges:
                if e.A.id == p.id:
                    neigh.append(e.B.id)
                if e.B.id == p.id:
                    neigh.append(e.A.id)
            print("point " + str(p.id) + " neigh: " + str(neigh))
            for m in neigh:
                for n in neigh:
                    if m < n:
                        included = 0
                        for ex in excl:
                            if m == ex[0] and n == ex[1]:
                                included = 1
                        if included == 0:
                            excl.append([m, n])
        print("n exclusions: " + str(len(excl)))
        print(str(excl))
        for ex in excl:
            self.cell_type.system.part[ex[0]].add_exclusion(ex[1])

    def save_cell(self, directory, save_interactions=True, save_json=True):
        import os
        import json
        import shutil
        os.makedirs(directory)

        # first for loop
        cellType = self.cell_type
        origin = self.origin

        # cell
        json_cell = {}
        # index of array in cell_types json
        json_cell['origin'] = origin
        json_cell['particle_type'] = self.particle_type
        json_cell['particle_mass'] = self.particle_mass

        positions = []
        for point in self.mesh.points:
            pos = point.get_pos()
            positions.append([pos[0] - origin[0], pos[1] -
                             origin[1], pos[2] - origin[2]])
        json_cell['positions'] = positions

        # end first loop

        # saving cell type
        json_cell_type = {}

        new_nodes_file_name = os.path.basename(cellType.mesh.nodes_file)
        new_triangles_file_name = os.path.basename(
            cellType.mesh.triangles_file)

        # mesh
        json_cell_type['nodes_file'] = new_nodes_file_name
        json_cell_type['triangles_file'] = new_triangles_file_name

        # cell type variables
        json_cell_type['resize'] = cellType.resize
        json_cell_type['ks'] = cellType.ks
        json_cell_type['kslin'] = cellType.kslin
        json_cell_type['kb'] = cellType.kb
        json_cell_type['kal'] = cellType.kal
        json_cell_type['kag'] = cellType.kag
        json_cell_type['kv'] = cellType.kv
        json_cell_type['kvisc'] = cellType.kvisc
        json_cell_type['normal'] = cellType.normal

        json_cell['cell_type'] = json_cell_type

        with open(directory + '/data.json', 'w') as outfile:
            json.dump(json_cell, outfile)

        os.makedirs(directory + "/nodes_files")
        os.makedirs(directory + "/triangles_files")

        shutil.copyfile(cellType.mesh.nodes_file, directory +
                        "/nodes_files/" + os.path.basename(new_nodes_file_name))
        shutil.copyfile(cellType.mesh.triangles_file, directory +
                        "/triangles_files/" + os.path.basename(new_triangles_file_name))

        return json_cell


class OifInnerParticles:

    def __init__(self, cell):
        self.cell = cell
        self.particles = []
        self.particle_type = -1
        self.particle_r = 0

        # particle-membrane soft-sphere interactions
        self.soft_a = None
        self.soft_n = None
        self.soft_cutoff = None

        # particle-particle hat interactions
        self.hat_fmax = None

        # particle-particle dpd interactions
        self.dpd_gamma = None
        self.dpd_cutoff = None

    def get_origin(self):
        center = np.array([0.0, 0.0, 0.0])
        for p in self.particles:
            center += p.pos
        return center / len(self.particles)

    def set_origin(self, new_origin=(0.0, 0.0, 0.0)):
        old_origin = self.get_origin()
        for particle in self.particles:
            new_position = particle.pos - old_origin + new_origin
            particle.pos = new_position

    def load(self, directory, json_filename, load_interactions=True, particle_type=None):
        import json

        f = open(directory + "/" + json_filename)

        data = json.load(f)
        f.close()

        seed_file = directory + "/" + data["particle"]["file"]

        if particle_type == None:
            self.particle_type = data["particle_type"]
        else:
            self.particle_type = particle_type

        positions = []

        if data["particle"]["r"] * 2 < self.cell.cell_type.mesh.aver_edge_length():
            raise Exception(
                "OifInnerParticles: Particle radius is too small for current mesh.")

        origin = self.cell.get_origin()
        f = open(seed_file, "r")
        for line in f:
            split = line.split(" ")
            split = [float(split[0]), float(split[1]), float(split[2])]
            positions.append(origin + split)
        f.close()

        system_npart = len(self.cell.cell_type.system.part)
        for i in range(len(positions)):
            self.particles.append(self.cell.cell_type.system.part.add(pos=positions[i],
                                                                      mass=data["particle"]["particle_mass"],
                                                                      id=system_npart + i,
                                                                      type=self.particle_type))

        particle_volume = len(positions) * 4 * np.pi * \
            data["particle"]["r"] ** 3 / 3.0
        cell_volume = self.cell.volume()

        if particle_volume >= 0.5 * cell_volume:
            raise Exception(
                "OifInnerParticles: Too large volume of inner particles.")

        if load_interactions:
            self.set_interactions(dpd_gamma=data["interactions"]["dpd"]["dpd_gamma"],
                                  hat_fmax=data["interactions"]["hat"]["hat_fmax"],
                                  soft_a=data["interactions"]["soft"]["soft_a"],
                                  soft_n=data["interactions"]["soft"]["soft_n"],
                                  dpd_cutoff=data["interactions"]["dpd"]["dpd_cutoff"],
                                  soft_cutoff=data["interactions"]["soft"]["soft_cutoff"])

    def seed_sphere(self, particle_r, n, particle_type, particle_mass=None, center=None, seed=-1):
        if not particle_mass:
            p_mass = self.cell.particle_mass
        else:
            p_mass = particle_mass

        # check radius with respect to the mesh
        # if 2 * particle_r < self.cell.cell_type.mesh.aver_edge_length():
        #     raise Exception(
        #         "OifInnerParticles: Particle radius is too small for current mesh.")

        self.particle_r = particle_r
        min_r = 100.0

        if not center:
            center = self.cell.get_origin()
        [xo, yo, zo] = center

        # finding the shortest distance from center to mesh
        for part in self.cell.mesh.points:
            dist = oif.vec_distance(part.get_pos(), center)
            if dist < min_r:
                min_r = dist

        # shortest distance must be shortened by radius of inner_part
        min_r -= self.cell.mesh.aver_edge_length()

        if seed != -1:
            random.seed(seed)

        # array of positions inside cell
        positions = []
        if not n == 0:
            max_attempts = 100000
            for i in range(max_attempts):
                # random position in box [2*min_r,2*min_r,2*min_r] is generated
                new_pos = [random.uniform(xo - min_r, xo + min_r),
                           random.uniform(yo - min_r, yo + min_r),
                           random.uniform(zo - min_r, zo + min_r)]
                if oif.vec_distance(new_pos, center) < min_r:
                    positions.append(new_pos)
                    # if we have enough particles, stop
                    if len(positions) == n:
                        break

        self.create_particles(particle_type, positions, p_mass)

    def seed_box(self, particle_r, n, a, b, c, particle_type, particle_mass=None, seed=-1):
        if not particle_mass:
            p_mass = self.cell.particle_mass
        else:
            p_mass = particle_mass

        if 2 * particle_r < self.cell.cell_type.mesh.aver_edge_length():
            raise Exception(
                "OifInnerParticles: Particle radius is too small for current mesh.")

        self.particle_r = particle_r

        if seed != -1:
            random.seed(seed)

        positions = []
        [xo, yo, zo] = self.cell.get_origin()

        for i in range(n):
            positions.append([random.uniform(xo - a / 2.0, xo + a / 2.0),
                              random.uniform(yo - b / 2.0, yo + b / 2.0),
                              random.uniform(zo - c / 2.0, zo + c / 2.0)])

        self.create_particles(particle_type, positions, p_mass)

    def create_particles(self, particle_type, positions, particle_r=None, particle_mass=None):
        if not particle_mass:
            p_mass = self.cell.particle_mass
        else:
            p_mass = particle_mass

        if particle_r:
            self.particle_r = particle_r

        self.particle_type = particle_type

        system_npart = len(self.cell.cell_type.system.part)

        # creating particles
        for i, position in enumerate(positions):
            self.particles.append(self.cell.cell_type.system.part.add(pos=position,
                                                                      mass=p_mass,
                                                                      id=system_npart + i,
                                                                      type=particle_type))

    def output_particles_to_dat(self, directory):
        part_file_name = "inner" + \
            str(len(self.particles)) + "_type" + \
            str(self.particle_type) + ".dat"
        f = open(directory + "/" + part_file_name, "w")
        cell_origin = self.cell.get_origin()
        for par in self.particles:
            position = par.pos - cell_origin
            f.write(str(position[0]) + " " +
                    str(position[1]) + " " + str(position[2]))
            f.write("\n")
        f.close()
        return part_file_name

    def save(self, directory, save_interactions=True):
        import os
        import json

        if not os.path.isdir(directory):
            os.makedirs(directory)

        # saving particles as dat file
        dat_file_name = self.output_particles_to_dat(directory)

        data = {}

        # saving information about inner particles
        particle = {}
        particle['file'] = dat_file_name
        particle['r'] = self.particle_r
        particle['particle_mass'] = self.particles[0].mass

        data['particle'] = particle
        data['particle_type'] = self.particle_type

        # saving interactions
        if save_interactions:
            interactions = {}

            # particle-membrane soft-sphere
            if self.soft_a and self.soft_n and self.soft_cutoff:
                soft = {}
                soft['soft_a'] = self.soft_a
                soft['soft_n'] = self.soft_n
                soft['soft_cutoff'] = self.soft_cutoff

                interactions['soft'] = soft

            # particle-particle dpd
            if self.dpd_gamma and self.dpd_cutoff:
                dpd = {}
                dpd['dpd_gamma'] = self.dpd_gamma
                dpd['dpd_cutoff'] = self.dpd_cutoff

                interactions['dpd'] = dpd

            # particle-particle hat
            if self.hat_fmax:
                hat = {}
                hat['hat_fmax'] = self.hat_fmax

                interactions['hat'] = hat

            data['interactions'] = interactions

        with open(directory + '/data.json', 'w') as outfile:
            json.dump(data, outfile)

    def output_vtk_pos_folded(self, num, output_directory=None):
        if output_directory is None:
            raise Exception(
                "OifInnerParticles: No output_directory provided for vtk output. Quitting.")
        file_name = output_directory + "/innerParticles_type" + \
            str(self.particle_type) + "_" + str(num) + ".vtk"
        self.cell.cell_type.system.part.writevtk(
            file_name, types=[self.particle_type])

    def set_interactions(self, dpd_gamma=3.0, hat_fmax=0.3, soft_a=0.256, soft_n=1.5, dpd_cutoff=None, soft_cutoff=None):

        # particle-membrane interactions
        self.soft_a = soft_a
        self.soft_n = soft_n
        if not soft_cutoff:
            self.soft_cutoff = 2 * self.particle_r
        else:
            self.soft_cutoff = soft_cutoff
        # if self.soft_cutoff < 0.6 * self.cell.aver_edge_length():
        #     raise Exception(
        #         "OifInnerParticles: Particle interactions cutoff too small with respect to mesh.")
        self.cell.cell_type.system.non_bonded_inter[self.particle_type, self.cell.particle_type].soft_sphere.set_params(
            a=soft_a,
            n=soft_n,
            cutoff=self.soft_cutoff,
            offset=0.0)

        # particle-particle interaction
        if not dpd_cutoff:
            self.dpd_cutoff = 2 * self.particle_r
        else:
            self.dpd_cutoff = dpd_cutoff
        self.dpd_gamma = dpd_gamma
        self.cell.cell_type.system.non_bonded_inter[self.particle_type, self.particle_type].dpd.set_params(
            weight_function=0,
            gamma=dpd_gamma,
            r_cut=self.dpd_cutoff,
            trans_weight_function=0,
            trans_gamma=dpd_gamma,
            trans_r_cut=self.dpd_cutoff)
        self.hat_fmax = hat_fmax
        self.cell.cell_type.system.non_bonded_inter[self.particle_type, self.particle_type].hat.set_params(
            F_max=hat_fmax,
            cutoff=self.dpd_cutoff)


class OifCluster:

    def __init__(self, name, cells):
        self.name = name
        self.cells = cells
        self.initial_positions = []
        self.n_contact_areas = -1

        if not cells:
            self.cell_radius = 0.0
            self.cell_type = None
        else:
            radius = cells[0].cell_type.resize[0]
            for cell in cells:
                if not cell.cell_type.mesh.check_if_spherical():
                    raise TypeError("OifCluster: Cells are not spherical.")
                if not cell.cell_type.resize[0] == radius:
                    raise TypeError(
                        "OifCluster: Cells are not of the same size.")
            self.cell_radius = self.cells[0].cell_type.resize[0]
            # Note: this assumes that all the cells have the same cell type.
            # If we want to assert this, we need equals() method for class CellType.
            self.cell_type = self.cells[0].cell_type

        # LJ interactions
        self.lj_eps = None
        self.lj_rmin = None
        self.lj_cutoff = None
        self.lj_shift = None

        # morse interactions
        self.m_eps = None
        self.m_alpha = None
        self.m_cutoff = None
        self.m_rmin = None

        # soft-sphere interactions
        self.soft_a = None
        self.soft_n = None
        self.soft_cutoff = None
        self.soft_offset = None

        # "self-cell" soft-sphere interactions
        self.sc_a = None
        self.sc_n = None
        self.sc_cutoff = None
        self.sc_offset = None

        # membrane collision interactions
        self.mc_a = None
        self.mc_n = None
        self.mc_cutoff = None
        self.mc_offset = None

    def _create_cells(self):
        start_particle_id = len(self.cell_type.system.part)
        for i, position in enumerate(self.initial_positions):
            self.add_cell(OifCell(cell_type=self.cell_type,
                                  particle_type=start_particle_id + i,
                                  origin=position,
                                  particle_mass=0.5,
                                  exclusion_neighbours=False))

    def get_origin(self):
        center = np.array([0.0, 0.0, 0.0])
        for cell in self.cells:
            center += cell.get_origin()
        return center / len(self.cells)

    def set_origin(self, new_origin=(0.0, 0.0, 0.0)):
        old_origin = self.get_origin()
        for cell in self.cells:
            new_position = cell.get_origin() - old_origin + new_origin
            cell.set_origin(new_position)

    def get_n_cells(self):
        return len(self.cells)

    def get_name(self):
        return self.name

    def set_n_contact_areas(self, n):
        self.n_contact_areas = n

    def get_n_nodes_cluster(self):
        num_points = 0
        for cell in self.cells:
            num_points += cell.get_n_nodes()
        return num_points

    def output_vtk_cluster(self, output_directory, cluster_num, vtk_time, particles=False):  # TO DOOOOOOOO
        for i in range(len(self.cells)):
            self.cells[i].output_vtk_pos_folded(file_name=output_directory + "cluster" + str(
                cluster_num) + "_cell" + str(i) + "_" + str(vtk_time) + ".vtk")
            if particles:
                self.cells[i].inner_particles.output_vtk_pos_folded(
                    output_directory=output_directory, num=vtk_time)

    def output_vtk_cell_bonds(self, output_directory, num):
        pairs = []
        positions = []
        if self.lj_cutoff:
            max_distance = self.lj_cutoff
        elif self.m_cutoff:
            max_distance = self.m_cutoff
        else:
            raise ValueError(
                'OifCluster: Cannot output cell bonds since there are no cell adhesion interactions (Lennard-Jones or Morse) defined.')

        for cell in self.cells:
            point_position = []
            for point in cell.mesh.points:
                point_position.append(point.get_pos())
            positions.append(point_position)

        for k, cell_k in enumerate(self.cells):
            for kk, part_k in enumerate(cell_k.mesh.points):
                kpos = positions[k][kk]
                for m, cell_m in enumerate(self.cells[k+1:len(self.cells)]):
                    for mm, part_m in enumerate(cell_m.mesh.points):
                        mpos = positions[k+m+1][mm]
                        d = oif.vec_distance(kpos, mpos)

                        if d <= max_distance:
                            line = [kpos[0], kpos[1], kpos[2],
                                    mpos[0], mpos[1], mpos[2]]
                            pairs.append(line)
        output_vtk_lines(lines=pairs, out_file=output_directory +
                         "/" + str(self.name) + "_lines_" + str(num) + ".vtk")

        return pairs

    def count_current_contact_areas(self):
        if self.lj_cutoff:
            max_distance = self.lj_cutoff
        elif self.m_cutoff:
            max_distance = self.m_cutoff
        else:
            raise ValueError(
                'OifCluster: Cannot count contact areas. Missing cell adhesion interactions (Lennard-Jones or Morse).')

        contacts = 0
        positions = []
        for cell in self.cells:
            point_position = []
            for point in cell.mesh.points:
                point_position.append(point.get_pos())
            positions.append(point_position)

        for k, cell_k in enumerate(self.cells):
            for m, cell_m in enumerate(self.cells[k + 1:len(self.cells)]):
                for kk, part_k in enumerate(cell_k.mesh.points):
                    kpos = positions[k][kk]
                    for mm, part_m in enumerate(cell_m.mesh.points):
                        mpos = positions[k+m+1][mm]
                        if oif.vec_distance(kpos, mpos) <= max_distance:
                            contacts += 1
                            break
                    if oif.vec_distance(kpos, mpos) <= max_distance:
                        break

        return contacts

    def add_cells(self, cells):
        self.cells.extend(cells)

    def add_cell(self, cell):
        self.cells.append(cell)

    def set_velocity(self, new_velocity=(0.0, 0.0, 0.0)):
        for cell in self.cells:
            cell.set_velocity(new_velocity)

    def set_velocities(self, velocities):
        for i in range(0, len(self.cells)):
            self.cells[i].set_velocity(velocities[i])

    def get_velocities(self):
        velocities = []
        for cell in self.cells:
            vel = cell.get_velocity()
            velocities.append((vel[0], vel[1], vel[2]))
        return velocities

    def get_mean_velocity(self):
        velocity = np.array([0.0, 0.0, 0.0])
        for cell in self.cells:
            velocity += cell.get_velocity()
        return velocity / len(self.cells)

    def rotate(self, rotate=(0.0, 0.0, 0.0)):
        # move to (0,0,0)
        origin = np.array(self.get_origin())
        self.set_origin()

        # create rotation matrix
        ca = np.cos(rotate[0])
        sa = np.sin(rotate[0])
        cb = np.cos(rotate[1])
        sb = np.sin(rotate[1])
        cc = np.cos(rotate[2])
        sc = np.sin(rotate[2])
        rotation = np.array(
            [[cb * cc, sa * sb * cc - ca * sc, sc * sa + cc * sb * ca],
             [cb * sc, ca * cc + sa * sb *
              sc, sc * sb * ca - cc * sa],
             [-sb, cb * sa, ca * cb]])

        # rotate
        for cell in self.cells:
            for point in cell.mesh.points:
                tmp_rotate_pos = np.array(point.get_pos())
                tmp_pos = rotation.dot(tmp_rotate_pos)
                tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                    tmp_pos[1]), discard_epsilon(tmp_pos[2])]
                point.set_pos(tmp_pos)

            if cell.inner_particles:
                for point in cell.inner_particles.particles:
                    tmp_rotate_pos = np.array(point.pos - origin)
                    tmp_pos = rotation.dot(tmp_rotate_pos)
                    tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                        tmp_pos[1]), discard_epsilon(tmp_pos[2])]
                    point.pos = tmp_pos + origin

        # move back
        self.set_origin(origin)

    def pos_bounds(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for cell in self.cells:
            cell_pos_bounds = cell.pos_bounds()

            if cell_pos_bounds[0] < x_min:
                x_min = cell_pos_bounds[0]
            if cell_pos_bounds[1] > x_max:
                x_max = cell_pos_bounds[1]
            if cell_pos_bounds[2] < y_min:
                y_min = cell_pos_bounds[2]
            if cell_pos_bounds[3] > y_max:
                y_max = cell_pos_bounds[3]
            if cell_pos_bounds[4] < z_min:
                z_min = cell_pos_bounds[4]
            if cell_pos_bounds[5] > z_max:
                z_max = cell_pos_bounds[5]

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def set_cell_boundary_interactions(self, boundary_particle_type, soft_a=0.00022, soft_n=0.5, soft_cutoff=0.5, soft_offset=0):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')
        system = self.cells[0].cell_type.system
        for cell in self.cells:
            system.non_bonded_inter[cell.particle_type, boundary_particle_type].soft_sphere.set_params(a=soft_a,
                                                                                                       n=soft_n,
                                                                                                       cutoff=soft_cutoff,
                                                                                                       offset=soft_offset)

    def set_soft_sphere_interactions(self, soft_a=0.00022, soft_n=0.5, soft_cutoff=0.5, soft_offset=0):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')

        system = self.cells[0].cell_type.system
        self.soft_a = soft_a
        self.soft_n = soft_n
        self.soft_cutoff = soft_cutoff
        self.soft_offset = soft_offset

        for i in range(len(self.cells)):
            for j in range(i+1, len(self.cells)):
                system.non_bonded_inter[i, j].soft_sphere.set_params(a=soft_a,
                                                                     n=soft_n,
                                                                     cutoff=soft_cutoff,
                                                                     offset=soft_offset)

    def set_morse_interactions(self, m_eps=0.0145, m_alpha=1.0, m_cutoff=0.7, m_rmin=0.5):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')

        system = self.cells[0].cell_type.system
        self.m_eps = m_eps
        self.m_alpha = m_alpha
        self.m_cutoff = m_cutoff
        self.m_rmin = m_rmin

        for i in range(len(self.cells)):
            for j in range(i+1, len(self.cells)):
                system.non_bonded_inter[i, j].morse.set_params(eps=m_eps,
                                                               alpha=m_alpha,
                                                               cutoff=m_cutoff,
                                                               rmin=m_rmin)

    def set_lennard_jones_interactions(self, lj_eps=0.005, lj_rmin=0.15, lj_cutoff=0.3, lj_shift=0.0):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')

        system = self.cells[0].cell_type.system
        self.lj_eps = lj_eps
        self.lj_rmin = lj_rmin
        self.lj_cutoff = lj_cutoff
        self.lj_shift = lj_shift

        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=lj_eps,
                                                                       sigma=lj_rmin /
                                                                       (2 ** (1 / 6)),
                                                                       cutoff=lj_cutoff,
                                                                       shift=lj_shift)

    def set_membrane_collision_interactions(self, mc_a=0.00022, mc_n=1.2, mc_cutoff=0.1, mc_offset=0.0):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')

        system = self.cells[0].cell_type.system
        self.mc_a = mc_a
        self.mc_n = mc_n
        self.mc_cutoff = mc_cutoff
        self.mc_offset = mc_offset

        for i in range(0, len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                system.non_bonded_inter[i, j].membrane_collision.set_params(a=mc_a,
                                                                            n=mc_n,
                                                                            cutoff=mc_cutoff,
                                                                            offset=mc_offset)

    def set_self_cell_soft_sphere_interactions(self, sc_a=0.0005, sc_n=1.2, sc_cutoff=0.1, sc_offset=0.0):
        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, interactions cannot be set.')

        system = self.cells[0].cell_type.system
        self.sc_a = sc_a
        self.sc_n = sc_n
        self.sc_cutoff = sc_cutoff
        self.sc_offset = sc_offset

        for i in range(len(self.cells)):
            system.non_bonded_inter[i, i].soft_sphere.set_params(a=sc_a,
                                                                 n=sc_n,
                                                                 cutoff=sc_cutoff,
                                                                 offset=sc_offset)

    def deform(self, n_cycles_deform=100, n_cycles_relax=10, vtk_directory="", force=0.5):
        # applies external force to all cells towards cluster centroid
        # runs for 100*n_cycles_deform integration steps
        # saves output to vtk_directory (if left empty, no vtk output is saved)
        # relaxation at the end for 100*n_cycles_relax

        if not self.cells:
            raise ValueError(
                'OifCluster: Cluster does not have any cells, deform() cannot be run.')
        if self.n_contact_areas == -1:
            raise ValueError(
                'OifCluster: n_contact_areas not set, deform() cannot be run.')

        system = self.cells[0].cell_type.system

        # set force
        centroid = self.get_origin()
        for cell in self.cells:
            cell_origin = cell.get_origin()
            distance = oif.vec_distance(centroid, cell_origin)
            if distance > small_epsilon:
                direction = (centroid - cell_origin) / distance
                cell.set_force(force / cell.get_n_nodes() * direction)

        # deformation
        pairs = []
        steps = 100
        time = steps

        for i in range(n_cycles_deform):

            if vtk_directory != "":
                self.output_vtk_cluster(output_directory=vtk_directory, num=i)
                # update positions of cell-cell bonds less frequently
                if (i % 20 == 0 and i != 0) or i == n_cycles_deform - 1:
                    pairs = self.output_vtk_cell_bonds(
                        output_directory=vtk_directory, num=i)
                else:
                    oif.output_vtk_lines(
                        lines=pairs, out_file=vtk_directory + "/" + str(self.name) + "_lines_" + str(i) + ".vtk")

            print("(deformation) time: " + str(time))
            system.integrator.run(steps=steps)
            time += steps

        # check if enough cells touch
        if self.n_contact_areas != self.count_current_contact_areas():
            raise RuntimeError(
                'OifCluster: Deformation time too short, not all cells bonded.')

        # remove force
        for cell in self.cells:
            cell.set_force([0.0, 0.0, 0.0])

        # relaxation
        for i in range(n_cycles_relax):

            if vtk_directory != "":
                self.output_vtk_cluster(
                    output_directory=vtk_directory, num=n_cycles_deform+i)
                # update positions of cell-cell bonds less frequently
                if (i % 20 == 0 and i != 0) or i == n_cycles_relax - 1:
                    pairs = self.output_vtk_cell_bonds(
                        output_directory=vtk_directory, num=n_cycles_deform+i)
                else:
                    oif.output_vtk_lines(lines=pairs, out_file=vtk_directory + "/" + str(
                        self.name) + "_lines_" + str(n_cycles_deform+i) + ".vtk")

            print("(relaxation) time: " + str(time))
            system.integrator.run(steps=steps)
            time += steps

    def color_contact_areas_vtk(self, output_directory, num):
        # for now only for clusters with the same cell radii
        # returns a list of contact radii
        # outputs vtks of cells with contact areas colored

        if self.lj_cutoff:
            max_distance = self.lj_cutoff
        elif self.m_cutoff:
            max_distance = self.m_cutoff
        else:
            raise ValueError('OifCluster: Cannot color contact areas.'
                             'Missing cell adhesion interactions (Lennard-Jones or Morse).')

        ncells = self.get_n_cells()
        contact_radii = []
        n_particles_in_contacts = []

        points_to_color = [
            [0.0 for i in range(self.cells[j].get_n_nodes())] for j in range(ncells)]
        for k in range(ncells):
            for m in range(ncells):
                if k != m:
                    n_particles_in_contact = 0
                    for idk, this_point in enumerate(self.cells[k].mesh.points):
                        this_point_position = this_point.get_pos()
                        for idm, other_cell_point in enumerate(self.cells[m].mesh.points):
                            dist = oif.vec_distance(
                                this_point_position, other_cell_point.get_pos())
                            if dist < max_distance:
                                points_to_color[k][idk] = 1.0
                                points_to_color[m][idm] = 1.0
                                n_particles_in_contact += 1
                                break  # if point is in adhesion interaction, we stop checking,
                                # so that we do not include it more than once (this also makes the check faster)
                    contact_area = 1.0 * n_particles_in_contact * \
                        self.cells[k].surface() / self.cells[k].get_n_nodes()
                    contact_radii.append(np.sqrt(contact_area / np.pi))
                    n_particles_in_contacts.append(n_particles_in_contact)

        # output vtk and color contact areas
        self.output_vtk_cluster(output_directory, num)
        for i in range(ncells):
            vtk_file_for_coloring = open(
                output_directory + "/" + str(self.name) + "_cell" + str(i) + "_" + str(num) + ".vtk", "a")
            vtk_file_for_coloring.write(
                "POINT_DATA " + str(self.cells[i].get_n_nodes()) + "\n")
            vtk_file_for_coloring.write("SCALARS contact float 1\n")
            vtk_file_for_coloring.write("LOOKUP_TABLE default\n")
            for point in points_to_color[i]:
                vtk_file_for_coloring.write(str(point) + "\n")
            vtk_file_for_coloring.close()

        return contact_radii, n_particles_in_contacts

    def save_cluster(self, directory, save_interactions=True, save_inner_particles=False):
        import os
        import json
        import shutil

        if not os.path.isdir(directory):
            os.makedirs(directory)

        nodes_files = []
        triangles_files = []
        cell_types = []

        origin = self.get_origin()

        data = {}
        data['name'] = self.name
        data['n_contact_areas'] = self.n_contact_areas
        data['origin'] = origin.tolist()
        data['cells'] = []

        for cell in self.cells:
            cellType = cell.cell_type

            # only compares the pointer, not object variable values
            if cellType not in cell_types:
                cell_types.append(cellType)

            mesh = cell.mesh

            # cell
            json_cell = {}
            # index of array in cell_types json
            json_cell['cell_type'] = cell_types.index(cellType)
            json_cell['origin'] = cell.origin
            json_cell['particle_type'] = cell.particle_type
            json_cell['particle_mass'] = cell.particle_mass

            positions = []
            for point in mesh.points:
                pos = point.get_pos()
                positions.append(
                    [pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]])
            json_cell['positions'] = positions

            data['cells'].append(json_cell)

        # saving cell type
        json_cell_types = {}
        for i, cell_type in enumerate(cell_types):

            # every celltype gets its own nodes.dat and triangles.dat file, even
            # when they had same name while saving, to avoid confusing different files with same names
            new_nodes_file_name = str(
                i) + "_" + os.path.basename(cell_type.mesh.nodes_file)
            new_triangles_file_name = str(
                i) + "_" + os.path.basename(cell_type.mesh.triangles_file)

            nodes_files.append(cell_type.mesh.nodes_file)
            triangles_files.append(cell_type.mesh.triangles_file)

            json_cell_type = {}
            # mesh
            json_cell_type['nodes_file'] = new_nodes_file_name
            json_cell_type['triangles_file'] = new_triangles_file_name

            # cell type variables
            json_cell_type['resize'] = cell_type.resize
            json_cell_type['ks'] = cell_type.ks
            json_cell_type['kslin'] = cell_type.kslin
            json_cell_type['kb'] = cell_type.kb
            json_cell_type['kal'] = cell_type.kal
            json_cell_type['kag'] = cell_type.kag
            json_cell_type['kv'] = cell_type.kv
            json_cell_type['kvisc'] = cell_type.kvisc
            json_cell_type['normal'] = cell_type.normal

            json_cell_types[i] = json_cell_type

        data['cell_types'] = json_cell_types

        # saving interactions
        if save_interactions:
            interactions = {}

            # LJ
            if self.lj_eps:
                lj = {}
                lj['lj_eps'] = self.lj_eps
                lj['lj_rmin'] = self.lj_rmin
                lj['lj_cutoff'] = self.lj_cutoff
                lj['lj_shift'] = self.lj_shift

                interactions['lj'] = lj

            # morse
            if self.m_eps:
                morse = {}
                morse['m_eps'] = self.m_eps
                morse['m_alpha'] = self.m_alpha
                morse['m_cutoff'] = self.m_cutoff
                morse['m_rmin'] = self.m_rmin

                interactions['morse'] = morse

            # soft-sphere
            if self.soft_a:
                soft_sphere = {}
                soft_sphere['soft_a'] = self.soft_a
                soft_sphere['soft_n'] = self.soft_n
                soft_sphere['soft_cutoff'] = self.soft_cutoff
                soft_sphere['soft_offset'] = self.soft_offset

                interactions['soft_sphere'] = soft_sphere

            # self-cell soft-sphere
            if self.sc_a:
                self_cell = {}
                self_cell['sc_a'] = self.sc_a
                self_cell['sc_n'] = self.sc_n
                self_cell['sc_cutoff'] = self.sc_cutoff
                self_cell['sc_offset'] = self.sc_offset

                interactions['self_cell'] = self_cell

            # membrane collision
            if self.mc_a:
                membrane_collision = {}
                membrane_collision['mc_a'] = self.mc_a
                membrane_collision['mc_n'] = self.mc_n
                membrane_collision['mc_cutoff'] = self.mc_cutoff
                membrane_collision['mc_offset'] = self.mc_offset

                interactions['membrane_collision'] = membrane_collision

            data['interactions'] = interactions

        with open(directory + '/data.json', 'w') as outfile:
            json.dump(data, outfile)

        os.makedirs(directory + "/nodes_files")
        os.makedirs(directory + "/triangles_files")

        for i, file in enumerate(nodes_files):
            shutil.copyfile(file, directory + "/nodes_files/" +
                            str(i) + "_" + os.path.basename(file))
        for i, file in enumerate(triangles_files):
            shutil.copyfile(file, directory + "/triangles_files/" +
                            str(i) + "_" + os.path.basename(file))

        # puts cluster origin to its original position
        self.set_origin(origin)

        # save inner particles
        index = 0
        if save_inner_particles:
            for cell in self.cells:
                cell.inner_particles.save(
                    directory=f"{directory}/inner_particles/{index}", save_interactions=True)
                index += 1

        # write cluster origin to file - for load_cluster method
        data_origin = {}
        data_origin['origin'] = self.get_origin().tolist()
        with open(directory + '/origin.json', 'w') as outfile:
            json.dump(data_origin, outfile)

    def load_cluster(self, system, directory, load_inner_particles=False, origin=None, load_interactions=True, array_particle_types=None,
                     cell_types=None):
        import json
        f = open(directory + '/data.json',)
        data = json.load(f)

        if (self.name is None):
            self.name = data['name']

        if ('n_contact_areas' in data):
            self.n_contact_areas = data['n_contact_areas']
        else:
            self.n_contact_areas = -1  # niekde je v sone contact areas a niekde nie

        json_cell_types = data['cell_types']
        temp_cell_types = [None] * len(json_cell_types)

        for type_key, cell_type in json_cell_types.items():
            temp_cell_types[int(type_key)] = oif.OifCellType(nodes_file=directory + '/nodes_files/' + cell_type['nodes_file'],
                                                             triangles_file=directory + '/triangles_files/' +
                                                             cell_type['triangles_file'],
                                                             check_orientation=False,
                                                             system=system,
                                                             ks=cell_type['ks'],
                                                             kslin=cell_type['kslin'],
                                                             kb=cell_type['kb'],
                                                             kal=cell_type['kal'],
                                                             kag=cell_type['kag'],
                                                             kv=cell_type['kv'],
                                                             kvisc=cell_type['kvisc'],
                                                             normal=cell_type['normal'],
                                                             resize=cell_type['resize'],
                                                             cell_types=cell_types)

        json_cells = data['cells']
        cells = []

        for index, cell in enumerate(json_cells):
            cell_type_key = cell['cell_type']
            type_of_cell = temp_cell_types[int(cell_type_key)]

            if (array_particle_types is None):
                type_of_particle = cell['particle_type']
            else:
                type_of_particle = array_particle_types[index]

            print("Type of particle: " + str(type_of_particle))
            new_cell = oif.OifCell(cell_type=type_of_cell,
                                   particle_type=type_of_particle,
                                   origin=cell['origin'],
                                   particle_mass=cell['particle_mass'],
                                   inner_particles=True,
                                   exclusion_neighbours=False)

            positions = cell['positions']

            for i, point in enumerate(new_cell.mesh.points):
                point.set_pos(positions[i])

            cells.append(new_cell)

        self.add_cells(cells)

        if origin is not None:
            self.set_origin(origin)
        else:
            self.set_origin(data['origin'])
            # duplicita

        print('Cluster je loadnuty na pozicii :' + str(origin))

        if load_interactions:
            json_interactions = data['interactions']

            if 'lj' in json_interactions:
                lj = json_interactions['lj']
                self.set_lennard_jones_interactions(lj_eps=lj['lj_eps'],
                                                    lj_rmin=lj['lj_rmin'],
                                                    lj_cutoff=lj['lj_cutoff'],
                                                    lj_shift=lj['lj_shift'])

            if 'morse' in json_interactions:
                morse = json_interactions['morse']
                self.set_morse_interactions(m_eps=morse['m_eps'],
                                            m_alpha=morse['m_alpha'],
                                            m_cutoff=morse['m_cutoff'],
                                            m_rmin=morse['m_rmin'])

            if 'soft_sphere' in json_interactions:
                soft_sphere = json_interactions['soft_sphere']
                self.set_soft_sphere_interactions(soft_a=soft_sphere['soft_a'],
                                                  soft_n=soft_sphere['soft_n'],
                                                  soft_cutoff=soft_sphere['soft_cutoff'],
                                                  soft_offset=soft_sphere['soft_offset'])

            if 'self_cell' in json_interactions:
                self_cell = json_interactions['self_cell']
                self.set_self_cell_soft_sphere_interactions(sc_a=self_cell['sc_a'],
                                                            sc_n=self_cell['sc_n'],
                                                            sc_cutoff=self_cell['sc_cutoff'],
                                                            sc_offset=self_cell['sc_offset'])

            if 'membrane_collision' in json_interactions:
                membrane_collision = json_interactions['membrane_collision']
                self.set_membrane_collision_interactions(mc_a=membrane_collision['mc_a'],
                                                         mc_n=membrane_collision['mc_n'],
                                                         mc_cutoff=membrane_collision['mc_cutoff'],
                                                         mc_offset=membrane_collision['mc_offset'])

        # load inner particles
        if load_inner_particles:
            index = 0
            for cell in self.cells:
                cell.inner_particles.load(
                    directory=f"{directory}/inner_particles/{index}", json_filename="data.json", load_interactions=True)
                index += 1

    def check_cell_separation(self):
        if (len(self.cells) >= 2):
            cell_separated = False
            for cell in self.cells:
                if (oif.vec_distance(cell.get_origin(), self.get_origin()) > 2 * cell.diameter()):
                    cell_separated = True
            return cell_separated
        else:
            print("You have only one cell in the cluster! There is nothing to check.")


class OifBiCluster(OifCluster):
    def __init__(self, name, cell_type, centroid=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifBiCluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(1)
        self.__create_positions(centroid, space)
        super()._create_cells()

    def __create_positions(self, centroid, space):
        [x, y, z] = centroid
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y, z])
        self.initial_positions.append(
            [x + (self.cell_radius + (space / 2.0)), y, z])


class OifL3Cluster(OifCluster):
    def __init__(self, name, cell_type, position=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifL3Cluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(2)
        self.__create_positions(position, space)
        super()._create_cells()

    def __create_positions(self, position, space):
        [x, y, z] = position
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y + (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x + (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])


class OifL4Cluster(OifCluster):
    def __init__(self, name, cell_type, position=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifL4Cluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(3)
        self.__create_positions(position, space)
        super()._create_cells()

    def __create_positions(self, position, space):
        [x, y, z] = position
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y + (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append([x - (self.cell_radius + (space / 2.0)), y - (
            self.cell_radius + (space / 2.0)), z - (2 * self.cell_radius + space)])
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x + (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])


class OifSquareCluster(OifCluster):
    def __init__(self, name, cell_type, centroid=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifSquareCluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(4)
        self.__create_positions(centroid, space)
        super()._create_cells()

    def __create_positions(self, centroid, space):
        [x, y, z] = centroid
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y + (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x + (self.cell_radius + (space / 2.0)), y + (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x - (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])
        self.initial_positions.append(
            [x + (self.cell_radius + (space / 2.0)), y - (self.cell_radius + (space / 2.0)), z])


class OifTetraCluster(OifCluster):
    def __init__(self, name, cell_type, centroid=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifTetraCluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(6)
        self.__create_positions(centroid, space)
        super()._create_cells()

    def __create_positions(self, centroid, space):
        [x, y, z] = centroid
        scale = (2*self.cell_radius + space) * np.sqrt(3.0/8.0)
        self.initial_positions.append(
            [x + scale * np.sqrt(8.0/9.0), y, z - scale/3.0])
        self.initial_positions.append(
            [x - scale * np.sqrt(2.0/9.0), y + scale * np.sqrt(2.0/3.0), z - scale/3.0])
        self.initial_positions.append(
            [x - scale * np.sqrt(2.0/9.0), y - scale * np.sqrt(2.0/3.0), z - scale/3.0])
        self.initial_positions.append([x, y, z + scale])


class OifStarCluster(OifCluster):
    def __init__(self, name, cell_type, centroid=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifStarCluster, self).__init__(name, cells=[])
        self.cell_radius = cell_type.resize[0]
        self.cell_type = cell_type
        self.set_n_contact_areas(12)
        self.__create_positions(centroid, space)
        super()._create_cells()

    def __create_positions(self, centroid, space):
        [x, y, z] = centroid
        self.initial_positions.append([x, y, z])
        self.initial_positions.append([x, y + 2 * self.cell_radius + space, z])
        self.initial_positions.append([x, y - 2 * self.cell_radius - space, z])
        self.initial_positions.append(
            [x - 2 * self.cell_radius, y + self.cell_radius + space/2.0, z])
        self.initial_positions.append(
            [x - 2 * self.cell_radius, y - self.cell_radius - space/2.0, z])
        self.initial_positions.append(
            [x + 2 * self.cell_radius, y + self.cell_radius + space/2.0, z])
        self.initial_positions.append(
            [x + 2 * self.cell_radius, y - self.cell_radius - space/2.0, z])


class OifDiamondCluster(OifCluster):
    def __init__(self, name, cell_type, centroid=(0.0, 0.0, 0.0), space=1.0):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifDiamondCluster, self).__init__(name, cells=[])
        self.cell_type = cell_type
        self.cell_radius = cell_type.resize[0]
        self.set_n_contact_areas(9)
        self.__create_positions(centroid, space)
        super()._create_cells()

    def __create_positions(self, centroid, space=1.0):
        [x, y, z] = centroid
        median_length = ((self.cell_radius * 2 + space) ** 2.0 -
                         (self.cell_radius + space / 2.0) ** 2.0) ** (1 / 2)
        centroid_distance_from_base = median_length * (1.0 / 3.0)
        centroid_distance_from_top = median_length * (2.0 / 3.0)
        median_length_from_center = (
            (self.cell_radius * 2.0 + space) ** 2.0 - centroid_distance_from_top ** 2.0) ** (1 / 2)
        self.initial_positions.append(
            [x - centroid_distance_from_base, y + (self.cell_radius + space / 2.0), z])
        self.initial_positions.append(
            [x - centroid_distance_from_base, y - (self.cell_radius + space / 2.0), z])
        self.initial_positions.append([x + centroid_distance_from_top, y, z])
        self.initial_positions.append([x, y, z + median_length_from_center])
        self.initial_positions.append([x, y, z - median_length_from_center])


class OifChainCluster(OifCluster):
    def __init__(self, name, cell_type, n_cells=5, angle=15, space=1.0, cluster_start=(0.0, 0.0, 0.0)):
        if not cell_type.mesh.check_if_spherical():
            raise TypeError("OifCluster: Cells are not spherical.")
        super(OifChainCluster, self).__init__(name, cells=[])
        self.cell_type = cell_type
        self.cell_radius = cell_type.resize[0]
        self.angle = angle
        self.set_n_contact_areas(n_cells - 1)
        self.__create_positions(cluster_start, angle, n_cells, space)
        super()._create_cells()

    def __create_positions(self, cluster_start, angle, n_cells, space=1.0):
        angle_surface1 = 0
        angle_surface2 = 0
        advance = 2 * self.cell_radius + space
        [x, y, z] = cluster_start
        self.initial_positions.append([x, y, z])

        for i in range(n_cells - 1):
            alpha = random.uniform(-angle, angle)
            beta = random.uniform(-angle, angle)
            angle_surface1 += alpha
            angle_surface2 += beta

            x = x + math.cos(math.radians(angle_surface1)) * \
                math.cos(math.radians(angle_surface2)) * advance
            y = y + math.sin(math.radians(angle_surface1)) * \
                math.cos(math.radians(angle_surface2)) * advance
            z = z + math.sin(math.radians(angle_surface2)) * advance

            self.initial_positions.append([x, y, z])

    def deform(self, n_cycles_deform=100, n_cycles_relax=10, vtk_directory="", force=5.0):
        counter = 0
        steps = 100  # integration steps
        time = steps
        pairs = []
        space = 0.8
        system = self.cell_type.system

        angle_surface1 = 0
        angle_surface2 = 0
        advance = 2 * self.cell_radius + space

        for m in range(len(self.cells) - 1):
            # move m+1st cell
            alpha = random.uniform(-self.angle, self.angle)
            beta = random.uniform(-self.angle, self.angle)
            angle_surface1 += alpha
            angle_surface2 += beta
            origin = self.cells[m].get_origin()
            x = origin[0] + math.cos(math.radians(angle_surface1)) * \
                math.cos(math.radians(angle_surface2)) * advance
            y = origin[1] + math.sin(math.radians(angle_surface1)) * \
                math.cos(math.radians(angle_surface2)) * advance
            z = origin[2] + math.sin(math.radians(angle_surface2)) * advance

            self.cells[m + 1].set_origin([x, y, z])

            cell_distance = oif.vec_distance(origin, [x, y, z])
            direction = (origin - [x, y, z])/cell_distance
            self.cells[m].set_force(-force /
                                    self.cells[m].get_n_nodes() * direction)
            self.cells[m+1].set_force(force /
                                      self.cells[m+1].get_n_nodes() * direction)

            for i in range(n_cycles_deform):
                if vtk_directory != "":
                    self.output_vtk_cluster(
                        output_directory=vtk_directory, num=counter)
                    # update positions of cell-cell bonds less frequently
                    if (i % 20 == 0 and i != 0) or i == n_cycles_deform - 1:
                        pairs = self.output_vtk_cell_bonds(
                            output_directory=vtk_directory, num=counter)
                    else:
                        oif.output_vtk_lines(lines=pairs, out_file=vtk_directory + "/" + str(
                            self.name) + "_lines_" + str(counter) + ".vtk")

                print("(deformation) time: " + str(time))
                system.integrator.run(steps=steps)
                time += steps
                counter += 1

            self.cells[m].set_force([0.0, 0.0, 0.0])
            self.cells[m + 1].set_force([0.0, 0.0, 0.0])

            # relaxation
            for i in range(n_cycles_relax):
                if vtk_directory != "":
                    self.output_vtk_cluster(
                        output_directory=vtk_directory, num=counter)
                    # update positions of cell-cell bonds less frequently
                    if (i % 20 == 0 and i != 0) or i == n_cycles_relax - 1:
                        pairs = self.output_vtk_cell_bonds(
                            output_directory=vtk_directory, num=counter)
                    else:
                        oif.output_vtk_lines(lines=pairs, out_file=vtk_directory + "/" + str(
                            self.name) + "_lines_" + str(counter) + ".vtk")

                print("(relaxation) time: " + str(time))
                system.integrator.run(steps=steps)
                time += steps
                counter += 1


class OifVarChainCluster(OifCluster):
    def __init__(self, name, cell_types, space=1.0, cluster_start=(0.0, 0.0, 0.0)):
        self.cell_types = cell_types
        self.cell_radii = []
        self.check_radii()
        self.set_n_contact_areas(len(self.cell_radii)-1)
        super(OifVarChainCluster, self).__init__(name, cells=[])
        self.__create_positions(cluster_start, space)
        self.__create_cells()

    def check_radii(self):
        for ctype in self.cell_types:
            if not ctype.mesh.check_if_spherical():
                raise TypeError("OifVarChainCluster: Cells are not spherical.")
            self.cell_radii.append(ctype.resize[0])

    def __create_positions(self, cluster_start, space):
        [x, y, z] = cluster_start
        self.initial_positions.append([x, y, z])
        for i in range(len(self.cell_types)-1):
            self.initial_positions.append(
                [self.initial_positions[i][0] + self.cell_radii[i] + self.cell_radii[i+1] + space, y, z])

    def __create_cells(self):
        start_particle_id = len(self.cell_types[0].system.part)
        for i in range(len(self.cell_types)):
            self.add_cell(OifCell(cell_type=self.cell_types[i],
                                  particle_type=start_particle_id + i,
                                  origin=self.initial_positions[i],
                                  particle_mass=0.5,
                                  exclusion_neighbours=False))

    def deform(self, n_cycles_deform=100, n_cycles_relax=10, vtk_directory="", force=5.0):
        counter = 0
        steps = 100  # integration steps
        time = steps
        pairs = []
        space = 0.8
        system = self.cell_types[0].system

        for m in range(len(self.cells) - 1):
            # move m+1st cell
            origin = self.cells[m].get_origin()
            advance = self.cell_radii[m] + self.cell_radii[m+1] + space
            self.cells[m +
                       1].set_origin([origin[0] + advance, origin[1], origin[2]])

            direction = (origin - self.cells[m + 1].get_origin()) / advance
            self.cells[m].set_force(-force /
                                    self.cells[m].get_n_nodes() * direction)
            self.cells[m + 1].set_force(force /
                                        self.cells[m + 1].get_n_nodes() * direction)

            for i in range(n_cycles_deform):
                if vtk_directory != "":
                    self.output_vtk_cluster(
                        output_directory=vtk_directory, num=counter)
                    # update positions of cell-cell bonds less frequently
                    if (i % 20 == 0 and i != 0) or i == n_cycles_deform - 1:
                        pairs = self.output_vtk_cell_bonds(
                            output_directory=vtk_directory, num=counter)
                    else:
                        oif.output_vtk_lines(lines=pairs, out_file=vtk_directory + "/" + str(
                            self.name) + "_lines_" + str(counter) + ".vtk")

                print("(deformation) time: " + str(time))
                system.integrator.run(steps=steps)
                time += steps
                counter += 1

            self.cells[m].set_force([0.0, 0.0, 0.0])
            self.cells[m+1].set_force([0.0, 0.0, 0.0])

        # relaxation
        for i in range(n_cycles_relax):
            if vtk_directory != "":
                self.output_vtk_cluster(
                    output_directory=vtk_directory, num=counter)
                # update positions of cell-cell bonds less frequently
                if (i % 20 == 0 and i != 0) or i == n_cycles_relax - 1:
                    pairs = self.output_vtk_cell_bonds(
                        output_directory=vtk_directory, num=counter)
                else:
                    oif.output_vtk_lines(lines=pairs, out_file=vtk_directory +
                                         "/" + str(self.name) + "_lines_" + str(counter) + ".vtk")

            print("(relaxation) time: " + str(time))
            system.integrator.run(steps=steps)
            time += steps
            counter += 1
