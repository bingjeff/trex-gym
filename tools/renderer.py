#!/usr/bin/env python

# Code modified from: 
# https://github.com/pycollada/pycollada/blob/master/examples/daeview/renderer/OldStyleRenderer.py

import logging
import numpy as np

from pyglet.gl import gl
from pyglet.gl import glu

def floatv3():
    return (gl.GLfloat * 3)(*[0]*3)

def to_floatv3(vec):
    return (gl.GLfloat * 3)(*vec)

def from_floatv3(floatv):
    return np.reshape(floatv, (3, 1), order='F')

def floatv4():
    return (gl.GLfloat * 4)(*[0]*4)

def to_floatv4(vec):
    return (gl.GLfloat * 4)(*vec)

def from_floatv4(floatv):
    return np.reshape(floatv, (4, 1), order='F')

def floatv4x4():
    return (gl.GLfloat * 16)(*[0]*16)

def to_floatv4x4(mat):
    return (gl.GLfloat * 16)(*np.array(mat).flatten(order='F'))

def from_floatv4x4(floatv):
    return np.reshape(floatv, (4, 4), order='F')

def get_gl_error():
    error_id = gl.glGetError()
    if error_id != 0:
        raise RuntimeError(f'GL Error ({error_id}): {glu.gluErrorString(error_id)}')

def get_model_view_matrix():
    return_val = floatv4x4()
    gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX, return_val)
    return from_floatv4x4(return_val)


class TriangleMeshShape:
    def __init__(self, color=None):
        self.diffuse_color = to_floatv4(color if color is not None else [0.3, 0.3, 0.3, 1.0])
        self.min_xyz = [np.inf] * 3
        self.max_xyz = [-np.inf] * 3

    def add_vertex(self, vertex, normal):
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, self.diffuse_color)
        gl.glNormal3fv(to_floatv3(normal))
        gl.glVertex3fv(to_floatv3(vertex))
        self.min_xyz = np.min((self.min_xyz, vertex), axis=0)
        self.max_xyz = np.max((self.max_xyz, vertex), axis=0)

    def draw(self):
        pass


class DaeMesh(TriangleMeshShape):
    def __init__(self, dae_object, color=None):
        super().__init__(color=color)
        self.dae = dae_object
    
    def draw(self):
        if self.dae.scene is None:
            logging.debug('Empty draw, DAE has no scene.')
            return
        gl.glBegin(gl.GL_TRIANGLES)
        normal = [0.0, 0.0, 1.0]
        for geometry in self.dae.scene.objects('geometry'):
            for primitive in geometry.primitives():                
                # Use primitive-specific ways to get triangles.
                primitive_type = type(primitive).__name__
                if primitive_type == 'BoundTriangleSet':
                    triangles = primitive
                elif primitive_type == 'BoundPolylist':
                    triangles = primitive.triangleset()
                else:
                    logging.warning(f'Unsupported mesh used: {primitive_type}')
                    break

                # Add triangles to the display list.
                for triangle in triangles:
                    nidx = 0
                    for vidx in triangle.indices:
                        if triangle.normals is None:
                            normal = [0.0, 0.0, 1.0]
                        else:
                            normal = triangle.normals[nidx]
                        nidx += 1
                        self.add_vertex(primitive.vertex[vidx], normal)
        get_gl_error()
        gl.glEnd()

class UvMeshShape(TriangleMeshShape):
    def __init__(self, color=None):
        super().__init__(color=color)
        self.num_u = 0
        self.num_v = 0

    def get_vertex_normal(self, u, v):
        return [0, 0, 0], [0, 0, 1]

    def draw(self):
        def add_uv_vertex(u, v):
            self.add_vertex(*self.get_vertex_normal(u, v))

        def add_tl_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v)
            add_uv_vertex(u - 1, v)

        def add_br_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v - 1)
            add_uv_vertex(u, v)

        gl.glBegin(gl.GL_TRIANGLES)
        for u in range(1, self.num_u):
            for v in range(1, self.num_v):
                add_br_triangle(u, v)
                add_tl_triangle(u, v)
        get_gl_error()
        gl.glEnd()


class SphereMesh(UvMeshShape):
    def __init__(self, radius, num_sectors=10, num_stacks=10, lo_phi=-0.5*np.pi, hi_phi=0.5*np.pi, z_offset=0.0, color=None):
        super().__init__(color=color)
        self.theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        self.phi = np.linspace(lo_phi, hi_phi, num_stacks)
        self.radius = radius
        self.num_u = num_sectors
        self.num_v = num_stacks
        self.z_offset = z_offset

    def get_vertex_normal(self, u, v):
        phi = self.phi[v]
        theta = self.theta[u]
        normal = np.array([np.cos(phi) * np.cos(theta),
                           np.cos(phi) * np.sin(theta),
                           np.sin(phi)])
        vertex = self.radius * normal
        vertex[2] += self.z_offset
        return vertex, normal


class TubeMesh(UvMeshShape):
    def __init__(self, radius, length, num_sectors=10, num_stacks=2, z_offset=0.0, color=None):
        super().__init__(color=color)
        self.theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        self.phi = np.linspace(-0.5 * length, 0.5 * length, num_stacks)
        self.radius = radius
        self.num_u = num_sectors
        self.num_v = num_stacks
        self.z_offset = z_offset

    def get_vertex_normal(self, u, v):
        phi = self.phi[v]
        theta = self.theta[u]
        normal = [np.cos(theta), np.sin(theta), 0.0]
        vertex = [self.radius * normal[0], self.radius * normal[1], phi + self.z_offset]
        return vertex, normal


class CapsuleMesh(TriangleMeshShape):
    def __init__(self, radius, length, num_sectors=10, num_stacks=10, color=None):
        super().__init__(color=color)
        self.top_cap = SphereMesh(radius, num_sectors, num_stacks, lo_phi=-0.5*np.pi, hi_phi=0.0, z_offset=-0.5*length, color=self.diffuse_color)
        self.tube = TubeMesh(radius, length, num_sectors, num_stacks=2, color=self.diffuse_color)
        self.bot_cap = SphereMesh(radius, num_sectors, num_stacks, lo_phi=0.0, hi_phi=0.5*np.pi, z_offset=0.5*length, color=self.diffuse_color)

    def draw(self):
        self.top_cap.draw()
        self.tube.draw()
        self.bot_cap.draw()
        self.max_xyz = np.max([self.max_xyz, self.top_cap.max_xyz, self.tube.max_xyz, self.bot_cap.max_xyz], axis=0)
        self.min_xyz = np.min([self.min_xyz, self.top_cap.min_xyz, self.tube.min_xyz, self.bot_cap.min_xyz], axis=0)

def get_aligned_xyz(points, rotation, translation):
    xyz = np.array(points) if np.shape(points)[1] == 3 else np.transpose(points)
    return (rotation @ (xyz + translation).T).T

def get_octant_xyz(points, min_points=100):
    xyz = np.array(points) if np.shape(points)[1] == 3 else np.transpose(points)
    x = xyz[:, 0] >= 0
    y = xyz[:, 1] >= 0
    z = xyz[:, 2] >= 0
    octants = []
    def check(logical_indices):
        if logical_indices.sum() > min_points:
            octants.append(points[logical_indices])
    check(x &  y &  z)
    check(x & ~y &  z)
    check(x & ~y & ~z)
    check(x &  y & ~z)
    check(~x &  y &  z)
    check(~x & ~y &  z)
    check(~x & ~y & ~z)
    check(~x &  y & ~z)
    return octants

def get_principle_axes(points):
    xyz = np.array(points) if np.shape(points)[1] == 3 else np.transpose(points)
    centroid = np.mean(xyz, axis=0)
    xyz -= centroid
    moment = xyz.T @ xyz
    # Axes are sorted x, y, z; make z the dominate axis.
    u, _, _ = np.linalg.svd(moment)
    z = u[:, 0]
    y = u[:, 1]
    x = np.cross(y, z)
    axes = np.vstack([x, y, z]).T
    aligned_xyz = (axes.T @ xyz.T).T
    aligned_center = 0.5 * (np.max(aligned_xyz, axis=0) + np.min(aligned_xyz, axis=0))
    aligned_hwl = np.max(aligned_xyz, axis=0) - np.min(aligned_xyz, axis=0)
    center = axes @ aligned_center + centroid
    return axes, center, aligned_hwl

def get_fitting_capsule(mesh, num_sectors=10, num_stacks=10, color=None):
    centroid = 0.5 * (mesh.max_xyz + mesh.min_xyz)
    hwl = mesh.max_xyz - mesh.min_xyz
    xyz = 0.5 * hwl + np.abs(centroid)
    radius = np.linalg.norm(xyz[:2])
    length = np.max([radius, xyz[2] - 2 * radius])
    return CapsuleMesh(radius, length, num_sectors=num_sectors, num_stacks=num_stacks, color=color)

class Camera:
    def __init__(self, origin=None, target=None, up_vector=None):
        self.origin = np.array(origin if origin is not None else [1.0, 1.0, 1.0])
        self.target = np.array(target if target is not None else [0.0, 0.0, 0.0])
        self.up = np.array(up_vector if up_vector is not None else [0.0, 0.0, 1.0])
        self.view = np.eye(4)
        self.look_at(self.target)

    def _normalize(self, vec):
        return vec / np.linalg.norm(vec)

    def look_at(self, point):
        self.target = np.array(point)
        camera_direction = self._normalize(self.origin - self.target)
        camera_right = self._normalize(np.cross(self.up, camera_direction))
        camera_up = self._normalize(np.cross(camera_direction, camera_right))
        self.view[0, :3] = camera_right
        self.view[1, :3] = camera_up
        self.view[2, :3] = camera_direction
        self.view[:3, 3] = np.dot(self.view[:3, :3], -self.origin)

    def load_view(self):
        gl.glLoadMatrixf(to_floatv4x4(self.view))

class DaeRenderer: 

    def __init__(self, dae, window):
        # Store the initial parameters.
        self.dae = dae
        self.window = window
        self.z_max = -1000
        self.z_min = 1000
        # Initialize the OpenGL parameters.
        self._init_gl_parameters()

        # Create a display list to store all the geometry in.
        print('Creating display list, could take a while...')
        self.displist = gl.glGenLists(1)
        gl.glNewList(self.displist, gl.GL_COMPILE)
        self.draw()
        gl.glEndList()
        print('...display list created. Ready to render.')
        print(f'z-range=[{self.z_min}, {self.z_max}]')

    def _init_gl_parameters(self):
        gl.glShadeModel(gl.GL_SMOOTH) # Enable Smooth Shading
        gl.glClearColor(0.0, 0.0, 0.0, 0.5) # Black Background
        gl.glClearDepth(1.0) # Depth Buffer Setup
        gl.glEnable(gl.GL_DEPTH_TEST) # Enables Depth Testing
        gl.glDepthFunc(gl.GL_LEQUAL) # The Type Of Depth Testing To Do
        
        gl.glEnable(gl.GL_MULTISAMPLE)

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)

        gl.glEnable(gl.GL_TEXTURE_2D) # Enable Texture Mapping
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    def clear_and_setup_window(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION) # Select The Projection Matrix
        gl.glLoadIdentity() # Reset The Projection Matrix
        field_of_view_y = 100
        aspect_ratio = self.window.width if self.window.height == 0 else self.window.width / self.window.height
        z_clip_near = 0.1
        z_clip_far = 1000.0
        glu.gluPerspective(field_of_view_y, aspect_ratio, z_clip_near, z_clip_far)

    def draw_simplified(self, xyz, num_divisions=0, division=0):
        diffuse_color = [0,1,0, 1]
        axes, center, hwl = get_principle_axes(xyz)
        gl.glPushMatrix()
        local_matrix = np.eye(4)
        local_matrix[:3, :3] = axes
        local_matrix[:3, 3] = center
        gl.glMultMatrixf(to_floatv4x4(local_matrix))
        if division < num_divisions:
            xyz_aligned = get_aligned_xyz(xyz, axes.T, -center)
            for xyz_partition in get_octant_xyz(xyz_aligned):
                self.draw_simplified(xyz_partition, num_divisions=num_divisions, division=division+1)
        else:
            radius = np.linalg.norm(0.5 * hwl[:2])
            length = hwl[2] - 2.0 * radius
            if length > 0.0:
                CapsuleMesh(radius, length, color=diffuse_color).draw()
            else:
                SphereMesh(radius, color=diffuse_color)
        gl.glPopMatrix()

    def draw(self):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        mesh = DaeMesh(self.dae)
        mesh.draw()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.draw_simplified(self.dae.geometries[0].primitives[0].vertex, num_divisions=2)
        self.z_max = mesh.max_xyz[2]
        self.z_min = mesh.min_xyz[2]
        
    def set_camera(self, origin, look_at_point):
        pass

    def set_light_position(self, origin):
        POINT_SOURCE = 1.0
        position = (gl.GLfloat * 4)(origin[0], origin[1], origin[2], POINT_SOURCE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, position)


    def render(self, rotate_x, rotate_y, rotate_z):
        self.clear_and_setup_window()

        # Create the "camera location" to start drawing from.
        gl.glMatrixMode(gl.GL_MODELVIEW) # Select The Model View Matrix
        gl.glLoadIdentity()
        z_radius = 3.0 * np.max(np.abs([self.z_min, self.z_max]))
        angle = np.deg2rad(rotate_y)
        origin = z_radius * np.array([np.sin(angle), 0.0, np.cos(angle)])
        self.set_light_position([0, 0, 0])
        camera = Camera(origin=origin, up_vector=[0, 1, 0])
        camera.load_view()
        
        # draw the display list
        gl.glCallList(self.displist)


    def cleanup(self):
        print('Renderer cleaning up')
        gl.glDeleteLists(self.displist, 1)