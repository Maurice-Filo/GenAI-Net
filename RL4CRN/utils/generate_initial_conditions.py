import numpy as np

def generate_triangular_prism_points(
    max_val=10.0,
    radius=1.0,          # "size" of the triangle around the line
    n_height=20,         # samples along the line x=y=z
    n_along_edge=20      # samples along each edge of the triangle
):
    """
    Generate surface samples of a triangular prism centered on the diagonal line x=y=z.

    This utility constructs a *regular* triangle in the plane orthogonal to the
    diagonal direction (i.e., in the plane x+y+z=0), then translates that triangle
    along the axis x=y=z to form a prism. Points are sampled on the three
    rectangular faces of the prism by linearly interpolating along each triangle
    edge at multiple heights.

    After sampling, each point is assigned to one of three classes based on the
    index of its maximum coordinate:

    - `A`: x is strictly/weakly the largest coordinate (argmax = 0)
    - `B`: y is strictly/weakly the largest coordinate (argmax = 1)
    - `C`: z is strictly/weakly the largest coordinate (argmax = 2)

    Args:
        max_val (float): Upper bound of the ambient cube [0, max_val]^3 in which
            points are generated.
        radius (float): Distance from the axis x=y=z to the triangle vertices
            (controls prism thickness).
        n_height (int): Number of sample positions along the axis x=y=z.
        n_along_edge (int): Number of samples along each triangle edge for each
            height (controls face resolution).

    Returns:
        dict[str, list[list[float]]]: Dictionary mapping class labels to lists of
            3D points:
            
            - `A: [[x, y, z], ...]`, points with x largest
            - `B: [[x, y, z], ...]`, points with y largest
            - `C: [[x, y, z], ...]`, points with z largest

    Notes:
        - Ties are broken by `np.argmax`, which returns the first occurrence of
          the maximum (e.g., if x==y>z the label will be indicated as "A").
        - The prism is kept inside the box [0, max_val]^3 by restricting the
          diagonal coordinate s to [radius, max_val-radius].
        - Only the *faces* of the prism are sampled (not the interior).
        
    """

    # Direction vectors in the plane x+y+z=0
    qA = np.array([ 2., -1., -1.])  # towards x-largest region
    qB = np.array([-1.,  2., -1.])  # towards y-largest region
    qC = np.array([-1., -1.,  2.])  # towards z-largest region

    # Normalize and scale by radius to get triangle vertices relative to the line
    qA_u = qA / np.linalg.norm(qA)
    qB_u = qB / np.linalg.norm(qB)
    qC_u = qC / np.linalg.norm(qC)

    vA = radius * qA_u
    vB = radius * qB_u
    vC = radius * qC_u

    # Choose positions along the line x=y=z (stay away from boundaries of [0,max_val]^3)
    s_min = radius
    s_max = max_val - radius
    line_vals = np.linspace(s_min, s_max, n_height)

    # Helper to sample a face given two vertices of the triangle
    def sample_face(v1, v2):
        pts_face = []
        t_vals = np.linspace(0.0, 1.0, n_along_edge)  # along the edge
        for s in line_vals:
            line_point = np.array([s, s, s])  # point on x=y=z
            for t in t_vals:
                offset = (1.0 - t) * v1 + t * v2
                p = line_point + offset
                pts_face.append(p)
        return np.array(pts_face)

    # Triangle vertices [vA, vB, vC].
    # Geometrically these are three faces of one triangular prism, BUT
    # they are NOT pure class-A/B/C faces, so we'll classify points afterwards.
    face_AB = sample_face(vA, vB)
    face_BC = sample_face(vB, vC)
    face_CA = sample_face(vC, vA)

    # Stack all face points
    all_pts = np.vstack([face_AB, face_BC, face_CA])

    # Classify by which coordinate is largest: 0->x, 1->y, 2->z
    labels = np.argmax(all_pts, axis=1)

    pts_A = all_pts[labels == 0]
    pts_B = all_pts[labels == 1]
    pts_C = all_pts[labels == 2]

    # Convert to lists of lists for output
    return {
        "A": pts_A.tolist(),  # x largest -> class (10, 0, 0)
        "B": pts_B.tolist(),  # y largest -> class (0, 10, 0)
        "C": pts_C.tolist(),  # z largest -> class (0, 0, 10)
    }
