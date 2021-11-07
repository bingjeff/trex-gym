#!/usr/bin/env python
import collada
import sys
import renderer

import pyglet
from pyglet.gl import *


try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True)
    window = pyglet.window.Window(resizable=False, config=config, vsync=True)
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=False)

window.rotate_x  = 0.0
window.rotate_y = 0.0
window.rotate_z = 0.0


@window.event
def on_draw():
    daerender.render(window.rotate_x, window.rotate_y, window.rotate_z)


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if abs(dx) > 2:
        if dx > 0:
            window.rotate_y += 2
        else:
            window.rotate_y -= 2
		
    if abs(dy) > 1:
        if dy > 0:
            window.rotate_x -= 2
        else:
            window.rotate_x += 2

    
@window.event
def on_resize(width, height):
    if height==0: height=1
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print('Usage: viewer.py path/to/a/collada.dae')
    
    print(f'Loading: {filename}')

    # open COLLADA file ignoring some errors in case they appear
    collada_file = collada.Collada(filename, ignore=[collada.DaeUnsupportedError,
                                            collada.DaeBrokenRefError])
    
    print(f'Loaded: {collada_file.filename}')

    daerender = renderer.DaeRenderer(collada_file, window)

    print(f'Renderer created: {daerender.dae.filename}')
	
    window.width = 1024
    window.height = 768
    
    print('Starting OpenGL context...')
    pyglet.app.run()
    print('Cleaning up...')
    daerender.cleanup()
    print('Finished.')