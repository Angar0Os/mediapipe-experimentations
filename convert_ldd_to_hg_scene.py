import harfang as hg
import re
import ast

hg.InputInit()
hg.WindowSystemInit()

res_x, res_y = 1280, 720
win = hg.RenderInit('Harfang - Draw Lines', res_x, res_y, hg.RF_VSync | hg.RF_MSAA4X)


line_count = 1000
shader = hg.LoadProgramFromFile('resources_compiled/shaders/white')

vtx_layout = hg.VertexLayout()
vtx_layout.Begin()
vtx_layout.Add(hg.A_Position, 3, hg.AT_Float)
vtx_layout.End()

vtx = hg.Vertices(vtx_layout, line_count * 2)

def to_vec3(p): return hg.Vec3(p['x'], -p['y'], -p['z'])

skeleton_links = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("RIGHT_HIP", "RIGHT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX")
]

with open("landmarks_data.txt", "r") as f:
    content = f.read()
    
    frames_raw = re.split(r'Frame \d+:', content)[1:] 
frames = []

for frame_raw in frames_raw:
    points = {}
    lines = frame_raw.strip().split('\n')
    for line in lines:
        match = re.match(r"(\w+): ({.*})", line)
        if match:
            name, data_str = match.groups()
            points[name] = ast.literal_eval(data_str)
    frames.append(points)
frame_index = 0

while not hg.ReadKeyboard().Key(hg.K_Escape) and hg.IsWindowOpen(win):
    dt = hg.TickClock()

    hg.SetViewClear(0, hg.CF_Color | hg.CF_Depth, hg.ColorI(50, 50, 50), 1, 0)
    hg.SetViewRect(0, 0, 0, 1280, 720)

    vtx.Clear()
    if frame_index < len(frames):
        points = frames[frame_index]
        line_id = 0
        for a, b in skeleton_links:
            if a in points and b in points:
                pa = to_vec3(points[a])
                pb = to_vec3(points[b])
                vtx.Begin(line_id * 2).SetPos(pa).End()
                vtx.Begin(line_id * 2 + 1).SetPos(pb).End()
                line_id += 1
        frame_index += 1 

    hg.DrawLines(0, vtx, shader)
    hg.Frame()
    hg.UpdateWindow(win)

hg.RenderShutdown()
hg.WindowSystemShutdown()