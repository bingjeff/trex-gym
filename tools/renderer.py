#!/usr/bin/env python

# Code modified from: 
# https://github.com/pycollada/pycollada/blob/master/examples/daeview/renderer/OldStyleRenderer.py

from pyglet.gl import gl
from pyglet.gl import glu



class DaeRenderer: 

    def __init__(self, dae, window):
        self.dae = dae
        self.window = window
        # to calculate model boundary
        self.z_max = -100000.0
        self.z_min = 100000.0
        self.textures = {}

        gl.glShadeModel(gl.GL_SMOOTH) # Enable Smooth Shading
        gl.glClearColor(0.0, 0.0, 0.0, 0.5) # Black Background
        gl.glClearDepth(1.0) # Depth Buffer Setup
        gl.glEnable(gl.GL_DEPTH_TEST) # Enables Depth Testing
        gl.glDepthFunc(gl.GL_LEQUAL) # The Type Of Depth Testing To Do
        
        gl.glEnable(gl.GL_MULTISAMPLE);

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)
        gl.glCullFace(gl.GL_BACK)

        gl.glEnable(gl.GL_TEXTURE_2D) # Enable Texture Mapping
        # gl.glEnable(gl.GL_TEXTURE_RECTANGLE_ARB) # Enable Texture Mapping
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        # create one display list
        print('Creating display list...')
        print('It could take some time. Please be patient :-) .')
        self.displist = gl.glGenLists(1)
        # compile the display list, store a triangle in it
        gl.glNewList(self.displist, gl.GL_COMPILE)
        self.draw_primitives()
        # self.draw_sphere()
        gl.glEndList()
        print('done. Ready to render.')

    def get_gl_error(self):
        error_id = gl.glGetError()
        if error_id != 0:
            raise RuntimeError(f'GL Error ({error_id}): {glu.gluErrorString(error_id)}')

    def draw_test(self):
        gl.glBegin(gl.GL_TRIANGLES)
        diffuse_color = (gl.GLfloat * 4)(0.3, 0.3, 0.3, 0.0)

        # Triangle 0.
        # T0 - Vertex 0.
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)
        gl.glNormal3fv((gl.GLfloat * 3)(0.0, 0.0, 1.0))
        gl.glVertex3fv((gl.GLfloat * 3)(0.0, 1.0, 0.0))
        # T0 - Vertex 1.
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)
        gl.glNormal3fv((gl.GLfloat * 3)(0.0, 0.0, 1.0))
        gl.glVertex3fv((gl.GLfloat * 3)(-0.5, 0.0, 0.0))
        # T0 - Vertex 2.
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, diffuse_color)
        gl.glNormal3fv((gl.GLfloat * 3)(0.0, 0.0, 1.0))
        gl.glVertex3fv((gl.GLfloat * 3)(0.5, 0.0, 0.0))

        self.z_min = -0.1
        self.z_max = 0.1
        gl.glEnd()

    def draw_sphere(self):
        sphere = glu.gluNewQuadric()
        radius, slices, stacks = (10.0, 10, 10)
        glu.gluSphere(sphere, radius, slices, stacks)
        self.z_min = -radius
        self.z_max = radius


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
        self.get_gl_error()
        gl.glEnd()


    def render(self, rotate_x, rotate_y, rotate_z):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION) # Select The Projection Matrix
        gl.glLoadIdentity() # Reset The Projection Matrix
        if self.window.height == 0: # Calculate The Aspect Ratio Of The Window
            glu.gluPerspective(100, self.window.width, 1.0, 5000.0)
        else:
            glu.gluPerspective(100, self.window.width / self.window.height, 1.0, 5000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW) # Select The Model View Matrix
        gl.glLoadIdentity()
        z_offset = self.z_min - (self.z_max - self.z_min) * 3
        light_pos = (gl.GLfloat * 3)(100.0, 100.0, 100.0 * -z_offset)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_pos)
        gl.glTranslatef(0, 0, z_offset)
        gl.glRotatef(rotate_x, 1.0, 0.0, 0.0)
        gl.glRotatef(rotate_y, 0.0, 1.0, 0.0)
        gl.glRotatef(rotate_z, 0.0, 0.0, 1.0)
        
        # draw the display list
        gl.glCallList(self.displist)


    def cleanup(self):
        print('Renderer cleaning up')
        gl.glDeleteLists(self.displist, 1)