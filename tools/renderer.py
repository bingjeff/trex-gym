#!/usr/bin/env python

# Code modified from: 
# https://github.com/pycollada/pycollada/blob/master/examples/daeview/renderer/OldStyleRenderer.py

import numpy as np

from pyglet.gl import gl
from pyglet.gl import glu

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
        self.dae = dae
        self.window = window
        # to calculate model boundary
        self.z_max = -100000.0
        self.z_min = 100000.0
        self.textures = {}
        # Initialize the OpenGL parameters.
        self._init_gl_parameters()

        # create one display list
        print('Creating display list, could take a while...')
        self.displist = gl.glGenLists(1)
        # compile the display list, store a triangle in it
        gl.glNewList(self.displist, gl.GL_COMPILE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        # self.draw_primitives()
        self.draw_capsule(5.0, 15.0, 10, 20, color=[1,0,0, 0])
        gl.glEndList()
        print('...display list created. Ready to render.')
        print(f'z-range=[{self.z_min}, {self.z_max}]')

    def _init_gl_parameters(self):
        gl.glShadeModel(gl.GL_SMOOTH) # Enable Smooth Shading
        gl.glClearColor(0.0, 0.0, 0.0, 0.5) # Black Background
        gl.glClearDepth(1.0) # Depth Buffer Setup
        gl.glEnable(gl.GL_DEPTH_TEST) # Enables Depth Testing
        gl.glDepthFunc(gl.GL_LEQUAL) # The Type Of Depth Testing To Do
        
        gl.glEnable(gl.GL_MULTISAMPLE);

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)

        gl.glEnable(gl.GL_TEXTURE_2D) # Enable Texture Mapping
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    def draw_capsule(self, radius, length, num_sectors, num_stacks, color=None):
        self.draw_hemi_sphere(radius, num_sectors, num_stacks, lo_phi=-0.5*np.pi, hi_phi=0.0, z_offset=-0.5*length, color=color)
        self.draw_tube(radius, length, num_sectors, num_stacks=2, color=color)
        self.draw_hemi_sphere(radius, num_sectors, num_stacks, lo_phi=0.0, hi_phi=0.5*np.pi, z_offset=0.5*length, color=color)

    def draw_tube(self, radius, length, num_sectors, num_stacks, z_offset=0.0, color=None):
        theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        phi = np.linspace(-0.5 * length, 0.5 * length, num_stacks)

        diffuse_color = to_floatv4(color if color is not None else [1.0, 1.0, 1.0, 1.0])

        def add_uv_vertex(u, v):
            nx = np.cos(theta[u])
            ny = np.sin(theta[u])
            nz = 0.0
            vx, vy, vz = (radius * nx, radius * ny, phi[v] + z_offset)
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)
            gl.glNormal3fv((gl.GLfloat * 3)(nx, ny, nz))
            gl.glVertex3fv((gl.GLfloat * 3)(vx, vy, vz))
            self.z_max = np.max((self.z_max, vz))
            self.z_min = np.min((self.z_min, vz))

        def add_tl_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v)
            add_uv_vertex(u - 1, v)

        def add_br_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v - 1)
            add_uv_vertex(u, v)

        gl.glBegin(gl.GL_TRIANGLES)
        for u in range(1, num_sectors):
            for v in range(1, num_stacks):
                add_tl_triangle(u, v)
                add_br_triangle(u, v)
        get_gl_error()
        gl.glEnd()

    def draw_hemi_sphere(self, radius, num_sectors, num_stacks, lo_phi=0.0, hi_phi=1.5, z_offset=0.0, color=None):
        theta = np.linspace(0.0, 2 * np.pi, num_sectors)
        phi = np.linspace(lo_phi, hi_phi, num_stacks)

        diffuse_color = to_floatv4(color if color is not None else [1.0, 1.0, 1.0, 1.0])

        def add_uv_vertex(u, v):
            nx = np.cos(phi[v]) * np.cos(theta[u])
            ny = np.cos(phi[v]) * np.sin(theta[u])
            nz = np.sin(phi[v])
            vx, vy, vz = (radius * nx, radius * ny, radius * nz + z_offset)
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)
            gl.glNormal3fv((gl.GLfloat * 3)(nx, ny, nz))
            gl.glVertex3fv((gl.GLfloat * 3)(vx, vy, vz))
            self.z_max = np.max((self.z_max, vz))
            self.z_min = np.min((self.z_min, vz))

        def add_tl_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v)
            add_uv_vertex(u - 1, v)

        def add_br_triangle(u, v):
            add_uv_vertex(u - 1, v - 1)
            add_uv_vertex(u, v - 1)
            add_uv_vertex(u, v)

        gl.glBegin(gl.GL_TRIANGLES)
        for u in range(1, num_sectors):
            for v in range(1, num_stacks):
                add_br_triangle(u, v)
                add_tl_triangle(u, v)
        get_gl_error()
        gl.glEnd()

    def draw_primitives(self):
        if self.dae.scene is None:
            print('Empty scene.')
            return
        gl.glBegin(gl.GL_TRIANGLES)
        for geometry in self.dae.scene.objects('geometry'):
            for primitive in geometry.primitives():
                diffuse_color = (gl.GLfloat * 4)(0.3, 0.3, 0.3, 0.0)
                
                # use primitive-specific ways to get triangles
                primitive_type = type(primitive).__name__
                if primitive_type == 'BoundTriangleSet':
                    triangles = primitive
                elif primitive_type == 'BoundPolylist':
                    triangles = primitive.triangleset()
                else:
                    print('Unsupported mesh used:', primitive_type)
                    break

                # add triangles to the display list
                for t in triangles:
                    nidx = 0

                    for vidx in t.indices:
                        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)

                        if not t.normals is None:
                            gl.glNormal3fv((gl.GLfloat * 3)(*t.normals[nidx]))

                        nidx += 1

                        vx, vy, vz = primitive.vertex[vidx]
                        gl.glVertex3fv((gl.GLfloat * 3)(vx, vy, vz))

                        # Calculate max and min Z coordinate
                        if vz > self.z_max:
                            self.z_max = vz
                        elif vz < self.z_min:
                            self.z_min = vz
        get_gl_error()
        gl.glEnd()

    def clear_and_setup_window(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION) # Select The Projection Matrix
        gl.glLoadIdentity() # Reset The Projection Matrix
        field_of_view_y = 100
        aspect_ratio = self.window.width if self.window.height == 0 else self.window.width / self.window.height
        z_clip_near = 0.1
        z_clip_far = 1000.0
        glu.gluPerspective(field_of_view_y, aspect_ratio, z_clip_near, z_clip_far)
        
    def set_camera(self, origin, look_at_point):
        pass

    def set_light_position(self, origin):
        POINT_SOURCE = 1.0
        position = (gl.GLfloat * 4)(origin[0], origin[1], origin[2], POINT_SOURCE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, position)


    def render(self, rotate_x, rotate_y, rotate_z):
        self.clear_and_setup_window()

        gl.glMatrixMode(gl.GL_MODELVIEW) # Select The Model View Matrix
        gl.glLoadIdentity()
        z_radius = 1.5 * np.max(np.abs([self.z_min, self.z_max]))
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