import json
import os
import plotly.graph_objects as go

def get_sort_perview_data(pth, prefix='PSNR'):
    assert os.path.exists(pth), f'Path {pth} does not exist'

    with open(pth, 'r') as f:
        data = json.load(f)['ours_20000']

    res = data[prefix]
    results = [0. for i in range(len(res))]

    for p in res:
        idx = int(p.split('.')[0])

        results[idx] = res[p]

    return results

pths = ['output/dnerf/jumpingjacks_MUL/per_view.json',
        'output/dnerf/jumpingjacks_MUL/4dgs_per_view.json']

names = ['Rot-W-4DGS', '4DGS']

# Plot the per view results of a givn scene
data = []
for idx, p in enumerate(pths):
    res = get_sort_perview_data(p, 'PSNR')

    data.append(go.Scatter(x=[i for i in range(len(res))], y=res, mode='lines', name=names[idx]))

fig = go.Figure(data=data)
fig.show()


# Plot the difference between the first two sets of results
res0 = get_sort_perview_data(pths[0], 'PSNR')
res1 = get_sort_perview_data(pths[1], 'PSNR')

data = []
for a, b in zip(res0, res1):
    diff = a #- b
    data.append(diff)

# fig = go.Figure(data=[go.Scatter(x=[i for i in range(len(data))], y=data, mode='lines', name='Rot-NonRot')])
# fig.show()

min_x, max_x = min(data), max(data)


def psnr_to_color(psnr, min_psnr, max_psnr, time=False):
    # Normalize PSNR between 0 and 1
    normalized_psnr = (psnr - min_psnr) / (max_psnr - min_psnr)

    # Map normalized PSNR to a color between blue (low PSNR) and red (high PSNR)
    # Color can be interpolated based on a gradient (e.g., blue for low, red for high)

    # Return color in rgb format

    if time:
        B = int(255 * psnr)
        G = int(255 * psnr)
        R = int(255 * psnr)

        # R = int(255 * (1 - psnr))  # Less PSNR = more blue
        return f'rgb({R}, {G}, {B})'

    blue_intensity = int(255 * normalized_psnr)  # More PSNR = more blue
    red_intensity = int(255 * (1 - normalized_psnr))  # Less PSNR = more red
    return f'rgb({red_intensity}, 0, {blue_intensity})'


# Plot the 3D Test Camera Position
test_data_pth = '/home/xi22005/DATA/dnerf/jumpingjacks/transforms_test.json'
assert os.path.exists(test_data_pth), f'Path {test_data_pth} does not exist'
with open(test_data_pth, 'r') as f:
    frames = json.load(f)['frames']

import numpy as np



cones = []
time_cones = []
for idx, cam in enumerate(frames):
    transformation_matrix = np.array(cam["transform_matrix"])

    # Extract the direction vector (uvw) and the position vector (xyz)
    uvw = transformation_matrix[:3, 2]  # Get the direction vector (u, v, w)
    xyz = transformation_matrix[:3, 3]  # Get the position vector (x, y, z)

    # Convert NumPy arrays to flat lists for use in Plotly
    uvw = uvw.flatten().tolist()  # Convert to list for Plotly
    xyz = xyz.flatten().tolist()


    col = psnr_to_color(data[idx], min_x, max_x)
    print(cam['file_path'], cam['time'], data[idx])

    # Create a cone object for the vector
    cones.append(go.Cone(
        x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],  # Start of the vector
        u=[uvw[0]], v=[uvw[1]], w=[uvw[2]],  # Direction of the vector
        colorscale=[[0, col], [1, col]],  # Vector color
        # sizemode="absolute",  # Use absolute size for length
        showscale=False,  # Hide the color scale
        anchor="tip",  # Anchor the cone at the tip
        # sizeref=2  # Size scaling factor
    ))

    col = psnr_to_color(cam['time'], 0., 1., time=True)
    time_cones.append(go.Cone(
        x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],  # Start of the vector
        u=[uvw[0]], v=[uvw[1]], w=[uvw[2]],  # Direction of the vector
        colorscale=[[0, col], [1, col]],  # Vector color
        # sizemode="absolute",  # Use absolute size for length
        showscale=False,  # Hide the color scale
        anchor="tip",  # Anchor the cone at the tip
        # sizeref=2  # Size scaling factor
    ))


print(len(cones))
fig = go.Figure(data=cones)

# Customize the layout
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(nticks=4, range=[0, 10]),
#         yaxis=dict(nticks=4, range=[0, 10]),
#         zaxis=dict(nticks=4, range=[0, 10])
#     ),
#     title='3D Vector Plot',
#     showlegend=True
# )
fig.show()

fig = go.Figure(data=time_cones)

# Customize the layout
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(nticks=4, range=[0, 10]),
#         yaxis=dict(nticks=4, range=[0, 10]),
#         zaxis=dict(nticks=4, range=[0, 10])
#     ),
#     title='3D Vector Plot',
#     showlegend=True
# )
fig.show()