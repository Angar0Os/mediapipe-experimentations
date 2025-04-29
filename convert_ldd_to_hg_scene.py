import harfang as hg 
import re
import ast
import time

skeleton_links = [
	("LEFT_SHOULDER", "RIGHT_SHOULDER"),
	("LEFT_SHOULDER", "LEFT_ELBOW"), 
	("LEFT_ELBOW", "LEFT_WRIST"),
	("RIGHT_SHOULDER", "RIGHT_ELBOW"), 
	("RIGHT_ELBOW", "RIGHT_WRIST"),
	("LEFT_SHOULDER", "LEFT_HIP"), 
	("RIGHT_SHOULDER", "RIGHT_HIP"),
	("LEFT_HIP", "RIGHT_HIP"),
	("LEFT_HIP", "LEFT_KNEE"), 
	("LEFT_KNEE", "LEFT_ANKLE"),
	("RIGHT_HIP", "RIGHT_KNEE"), 
	("RIGHT_KNEE", "RIGHT_ANKLE")
]

def to_vec3(p): return hg.Vec3(p['x'], -p['y'], -p['z'])

def create_material(diffuse, specular, self):
	mat = hg.CreateMaterial(prg_ref, 'uDiffuseColor', diffuse, 'uSpecularColor', specular)
	hg.SetMaterialValue(mat, 'uSelfColor', self)
	return mat

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

hg.InputInit()
hg.WindowSystemInit()

res_x, res_y = 1280, 720
win = hg.RenderInit('Harfang - Draw Bones & Joints', res_x, res_y, hg.RF_VSync | hg.RF_MSAA4X)

hg.AddAssetsFolder('resources_compiled')

pipeline = hg.CreateForwardPipeline()
res = hg.PipelineResources()

vtx_layout = hg.VertexLayoutPosFloatNormUInt8()

sphere_mdl = hg.CreateSphereModel(vtx_layout, 0.5, 12, 24)
sphere_ref = res.AddModel('sphere', sphere_mdl)

cube_mdl = hg.CreateCubeModel(vtx_layout, 0.1, 0.1, 1.0)  # z-axis length = 1.0
cube_ref = res.AddModel('cube', cube_mdl)

prg_ref = hg.LoadPipelineProgramRefFromAssets('core/shader/default.hps', res, hg.GetForwardPipelineInfo())
mat_objects = create_material(hg.Vec4(0.5, 0.5, 0.5), hg.Vec4(1, 1, 1), hg.Vec4(0, 0, 0))

scene = hg.Scene()
scene.canvas.color = hg.ColorI(22, 56, 76)
scene.environment.fog_color = scene.canvas.color
scene.environment.fog_near = 20
scene.environment.fog_far = 80

cam_mtx = hg.TransformationMat4(hg.Vec3(0, 5, -10), hg.Deg3(35, 0, 0))
cam = hg.CreateCamera(scene, cam_mtx, 0.01, 5000)
scene.SetCurrentCamera(cam)

lgt = hg.CreateLinearLight(scene, hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Deg3(30, 59, 0)), hg.Color(1, 0.8, 0.7), hg.Color(1, 0.8, 0.7), 10, hg.LST_Map, 0.002, hg.Vec4(50, 100, 200, 400))
back_lgt = hg.CreatePointLight(scene, hg.TranslationMat4(hg.Vec3(0, 10, 10)), 100, hg.ColorI(94, 155, 228), hg.ColorI(94, 255, 228))

joints = set()
for a, b in skeleton_links:
	joints.add(a)
	joints.add(b)
joints = list(joints)

sphere_nodes = {joint: hg.CreateObject(scene, hg.TranslationMat4(hg.Vec3(0,0,0)), sphere_ref, [mat_objects]) for joint in joints}

cube_nodes = [hg.CreateObject(scene, hg.TranslationMat4(hg.Vec3(0,0,0)), cube_ref, [mat_objects]) for _ in skeleton_links]

target_fps = 120
frame_delay = 1.0 / target_fps

frame_index = 0
repeat_counter = 0
repeats_per_frame = 3 

while hg.IsWindowOpen(win):
	start_time = time.time()
	state = hg.ReadKeyboard()
	dt = hg.TickClock()
	
	scene.Update(dt)
	if frame_index < len(frames):
		points = frames[frame_index]

		for joint, node in sphere_nodes.items():
			if joint in points:
				pos = to_vec3(points[joint]) * 10
				node.GetTransform().SetPos(pos)
				node.GetTransform().SetScale(hg.Vec3(0.3, 0.3, 0.3))

		for i, (a, b) in enumerate(skeleton_links):
			if a in points and b in points:
				pa = to_vec3(points[a]) * 10
				pb = to_vec3(points[b]) * 10

				mid_pos = (pa + pb) * 0.5
				dir_vec = pb - pa
				length = hg.Len(dir_vec)
				dir_vec = hg.Normalize(dir_vec)

				up_vec = hg.Vec3(0, 1, 0)
				if abs(hg.Dot(dir_vec, up_vec)) > 0.99:
					up_vec = hg.Vec3(1, 0, 0)

				rot_mat = hg.Mat3LookAt(dir_vec, up_vec)
				transform = hg.TransformationMat4(mid_pos, rot_mat)
				cube_nodes[i].GetTransform().SetWorld(transform)
				cube_nodes[i].GetTransform().SetScale(hg.Vec3(0.2, 0.2, length))

		repeat_counter += 1
		if repeat_counter >= repeats_per_frame:
			frame_index += 1
			repeat_counter = 0
 
	view_id = 0
	view_id, _ = hg.SubmitSceneToPipeline(view_id, scene, hg.IntRect(0, 0, res_x, res_y), True, pipeline, res)
 
	hg.Frame()
	hg.UpdateWindow(win)
 
	elapsed = time.time() - start_time
	if elapsed < frame_delay:
		time.sleep(frame_delay - elapsed)

hg.RenderShutdown()
hg.DestroyWindow(win)
hg.WindowSystemShutdown()
hg.InputShutdown()
