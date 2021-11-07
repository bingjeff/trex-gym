#!/usr/bin/env python

from pyglet.gl import *


class DaeRenderer: 

    def __init__(self, dae, window):
        self.dae = dae
        self.window = window
        # to calculate model boundary
        self.z_max = -100000.0
        self.z_min = 100000.0
        self.textures = {}

        glShadeModel(GL_SMOOTH) # Enable Smooth Shading
        glClearColor(0.0, 0.0, 0.0, 0.5) # Black Background
        glClearDepth(1.0) # Depth Buffer Setup
        glEnable(GL_DEPTH_TEST) # Enables Depth Testing
        glDepthFunc(GL_LEQUAL) # The Type Of Depth Testing To Do
        
        glEnable(GL_MULTISAMPLE);

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glCullFace(GL_BACK)

        glEnable(GL_TEXTURE_2D) # Enable Texture Mapping
        # glEnable(GL_TEXTURE_RECTANGLE_ARB) # Enable Texture Mapping
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # create one display list
        print('Creating display list...')
        print('It could take some time. Please be patient :-) .')
        self.displist = glGenLists(1)
        # compile the display list, store a triangle in it
        glNewList(self.displist, GL_COMPILE)
        self.draw_primitives()
        # self.drawTest()
        glEndList()
        print('done. Ready to render.')

    def get_gl_error(self):
        error_id = glGetError()
        if error_id != 0:
            raise RuntimeError(f'GL Error ({error_id}): {gluErrorString(error_id)}')

    def drawTest(self):
        glBegin(GL_TRIANGLES)
        diffuse_color = (GLfloat * 4)(0.3, 0.3, 0.3, 0.0)

        # Triangle 0.
        # T0 - Vertex 0.
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color)
        glNormal3fv((GLfloat * 3)(0.0, 0.0, 1.0))
        glVertex3fv((GLfloat * 3)(0.0, 1.0, 0.0))
        # T0 - Vertex 1.
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color)
        glNormal3fv((GLfloat * 3)(0.0, 0.0, 1.0))
        glVertex3fv((GLfloat * 3)(-0.5, 0.0, 0.0))
        # T0 - Vertex 2.
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color)
        glNormal3fv((GLfloat * 3)(0.0, 0.0, 1.0))
        glVertex3fv((GLfloat * 3)(0.5, 0.0, 0.0))

        self.z_min = -0.1
        self.z_max = 0.1
        glEnd()


    def draw_primitives(self):
        if self.dae.scene is None:
            print('Empty scene.')
            return
        glBegin(GL_TRIANGLES)
        for geometry in self.dae.scene.objects('geometry'):
            for primitive in geometry.primitives():
                diffuse_color = (GLfloat * 4)(0.3, 0.3, 0.3, 0.0)
                
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
                        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_color)

                        if not t.normals is None:
                            glNormal3fv((GLfloat * 3)(*t.normals[nidx]))

                        nidx += 1

                        vx, vy, vz = primitive.vertex[vidx]
                        glVertex3fv((GLfloat * 3)(vx, vy, vz))

                        # Calculate max and min Z coordinate
                        if vz > self.z_max:
                            self.z_max = vz
                        elif vz < self.z_min:
                            self.z_min = vz
        self.get_gl_error()
        glEnd()


    def render(self, rotate_x, rotate_y, rotate_z):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION) # Select The Projection Matrix
        glLoadIdentity() # Reset The Projection Matrix
        if self.window.height == 0: # Calculate The Aspect Ratio Of The Window
            gluPerspective(100, self.window.width, 1.0, 5000.0)
        else:
            gluPerspective(100, self.window.width / self.window.height, 1.0, 5000.0)
        glMatrixMode(GL_MODELVIEW) # Select The Model View Matrix
        glLoadIdentity()
        z_offset = self.z_min - (self.z_max - self.z_min) * 3
        light_pos = (GLfloat * 3)(100.0, 100.0, 100.0 * -z_offset)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glTranslatef(0, 0, z_offset)
        glRotatef(rotate_x, 1.0, 0.0, 0.0)
        glRotatef(rotate_y, 0.0, 1.0, 0.0)
        glRotatef(rotate_z, 0.0, 0.0, 1.0)
        
        # draw the display list
        glCallList(self.displist)


    def cleanup(self):
        print('Renderer cleaning up')
        glDeleteLists(self.displist, 1)