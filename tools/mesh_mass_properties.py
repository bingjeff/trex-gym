# A set of tools to aid calculating mass properties from a basic mesh.
import numpy as np

def extract_vertices(mesh):
  """Extract the vertex tuplet from a triangle/quadrangle.

  Args:
      mesh: An object filled with polygons that have vertices. Nominally a blender object.

  Returns:
      (v0, v1, v2) are lists of vertices corresponding to each face.
  """
  v0 = []
  v1 = []
  v2 = []
  for face in mesh.polygons:
    num_verts = len(face.vertices[:])
    if num_verts == 3:
      tris = [[0, 1, 2]]
    elif num_verts == 4:
      tris = quadrangle_to_triangle(face.vertices)
    else:
      raise ValueError(f'Unhandled polygon size: {num_verts}')
    for tri in tris:
      v0.append(mesh.vertices[tri[0]].co)
      v1.append(mesh.vertices[tri[1]].co)
      v2.append(mesh.vertices[tri[2]].co)
  return v0, v1, v2

def quadrangle_to_triangle(vertices):
  """Splits a quandrangle index list into two triangles.

  Assumes the vertices are sorted in a cycle and the quadralateral is planar.

  Args:
      vertices: A list of 4 indices corresponding to the vertices of a quadrangle.

  Returns:
      (tri0, tri1): a pair of lists, each representing the split triangles.
  """
  tri0 = [vertices[0], vertices[1], vertices[2]]
  tri1 = [vertices[2], vertices[3], vertices[0]]
  return tri0, tri1

def get_mass_properties(v0, v1, v2):
  """Calculate the volume, center-of-mass, and inertia.

  Assuming a watertight mesh the list of vertices is used to calculate the volume, center-of-mass, and inertia tensor about the center-of-mass. Assumes a uniform density and a mass of 1.0. Vertices are assumed to be ordered (v0, v1, v2) with consistent cycle corresponding to normals that face exterior.

  Algorithm follows what is proposed by:
  http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

  Args:
      v0: a list of triplets (x, y, z) where each element describes the first vertex in a triangular face.
      v1: a list of triplets (x, y, z) where each element describes the second vertex in a triangular face.
      v2: a list of triplets (x, y, z) where each element describes the third vertex in a triangular face.

  Returns:
      A dictionary with keys {'volume', 'com', 'inertia'}.
  """
  def subexpression(w0, w1, w2):
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0 * w0
    temp2 = temp1 + w1 * temp0
    f2 = temp2 + w2 * f1
    f3 = w0 * temp1 + w1 * temp2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2
  # Extract the coordinates, edges, and common expressions.
  x0, y0, z0 = v0[:, 0], v0[:, 1], v0[:, 2]
  x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
  x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]
  a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
  a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
  d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1
  # Calculate the subexpressions.
  f1x, f2x, f3x, g0x, g1x, g2x = subexpression(x0, x1, x2)
  f1y, f2y, f3y, g0y, g1y, g2y = subexpression(y0, y1, y2)
  f1z, f2z, f3z, g0z, g1z, g2z = subexpression(z0, z1, z2)
  # Calculate the integrals.
  integrals = [
    np.sum(d0 * f1x) / 6.0,
    np.sum(d0 * f2x) / 24.0,
    np.sum(d1 * f2y) / 24.0,
    np.sum(d2 * f2z) / 24.0,
    np.sum(d0 * f3x) / 60.0,
    np.sum(d1 * f3y) / 60.0,
    np.sum(d2 * f3z) / 60.0,
    np.sum(d0 * (y0 * g0x + y1 * g1x + y2 * g2x)) / 120.0,
    np.sum(d1 * (z0 * g0y + z1 * g1y + z2 * g2y)) / 120.0,
    np.sum(d2 * (x0 * g0z + x1 * g1z + x2 * g2z)) / 120.0,
  ]
  volume = integrals[0]
  cm = integrals[1:4] / volume
  cm2 = cm ** 2
  inertia = np.zeros((3, 3))
  inertia[0, 0] = integrals[5] + integrals[6] - volume * (cm2[1] + cm2[2])
  inertia[1, 1] = integrals[4] + integrals[6] - volume * (cm2[2] + cm2[0])
  inertia[2, 2] = integrals[4] + integrals[5] - volume * (cm2[0] + cm2[1])
  inertia[0, 1] = -(integrals[7] - volume * cm[0] * cm[1])
  inertia[0, 2] = -(integrals[9] - volume * cm[2] * cm[0])
  inertia[1, 2] = -(integrals[8] - volume * cm[1] * cm[2])
  inertia[1, 0] = inertia[0, 1]
  inertia[2, 0] = inertia[0, 2]
  inertia[2, 1] = inertia[1, 2]
  return {'volume': volume, 'center_of_mass': cm, 'inertia': inertia}
