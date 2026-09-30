import bpy
import cProfile
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_plane, intersect_point_tri_2d
import numpy as np
import os
from bpy_extras import view3d_utils
from .functions_general import *
from .functions_drawing import *
from .functions_modal_keymap import *
from .classes import *
from .keymap import addon_keymaps

def match_loops_vecs(source_vecs, target_vecs, target_inds):
    #
    # Match the list of vectors with the target list
    # matches by smallest angle
    # Source vecs should be a 2d array Nx3 of just 3d vectors
    # Target vecs should be a 3d array NxIx3 where it is a list of lists of 3d vectors for the source vec to be tested against
    # Each I of the NxIx3 array is a list of vectors to test against the N vectors of source
    #

    # Get the dot product from the matched cos tangents to the original loops tangent
    # and filter out -1 loops from the matched co
    dots = np.sum(source_vecs[:, np.newaxis] * target_vecs, axis=2)

    dots *= -1
    dots[target_inds < 0] = np.nan

    sort = np.argsort(dots)[:, 0]

    indeces = np.arange(sort.size)

    return target_inds[indeces, sort]


def set_new_normals(modal):
    modal._object.data.edges.foreach_set(
        'use_edge_sharp', modal._container.og_sharp)

    # Lerp between cached and new normals by the filter weights
    if modal._container.filter_mask.any():
        modal._container.new_norms[:] = modal._container.cache_norms * (
            1.0-modal._container.filter_weights[:, None]) + modal._container.new_norms * modal._container.filter_weights[:, None]

    # Get the scale factor to normalized new normals
    magnitudes_sq = np.sum(np.square(modal._container.new_norms), axis=1)
    scale = np.ones_like(magnitudes_sq)
    non_zero_mask = magnitudes_sq > 1e-10  # 使用小阈值避免数值问题
    scale[non_zero_mask] = 1 / np.sqrt(magnitudes_sq[non_zero_mask])
    modal._container.new_norms = modal._container.new_norms * scale[:, None]

    if modal._mirror_x:
        sel_norms = modal._container.new_norms[modal._container.sel_status]
        sel_norms[:, 0] *= -1
        modal._container.new_norms[modal.mir_loops_x[modal._container.sel_status]] = sel_norms
    if modal._mirror_y:
        sel_norms = modal._container.new_norms[modal._container.sel_status]
        sel_norms[:, 1] *= -1
        modal._container.new_norms[modal.mir_loops_y[modal._container.sel_status]] = sel_norms
    if modal._mirror_z:
        sel_norms = modal._container.new_norms[modal._container.sel_status]
        sel_norms[:, 2] *= -1
        modal._container.new_norms[modal.mir_loops_z[modal._container.sel_status]] = sel_norms

    # modal._container.new_norms.shape = [len(modal._object.data.loops), 3]
    # 将法线数据应用到模型上的方法
    # 不用将修改后的法线数据存入到模型法线中，而是用来计算顶点色再存入模型中
    # modal._object.data.normals_split_custom_set(modal._container.new_norms)

    # print("设置法线数据")

    # 获取网格对象
    mesh = modal._object.data
    mesh.calc_normals_split()

    loops = mesh.loops

    # 创建一个自定义原始法线参数用于存储原始法线数据
    customNormal_attr = mesh.attributes.get("CustomNormal")
    if not customNormal_attr:
        customNormal_attr = mesh.attributes.new(name="CustomNormal",type ='FLOAT_VECTOR', domain = 'CORNER')

    # 批量设置自定义法线数据
    custom_normal_data = customNormal_attr.data
    for i in range(len(loops)):
        custom_normal_data[i].vector = modal._container.new_norms[i]

    # 重新绘制，刷新界面
    modal.redraw = True
    return


def mirror_normals(modal, axis):
    sel_norms = modal._container.new_norms[modal._container.sel_status]

    sel_norms[:, axis] *= -1
    if axis == 0:
        modal._container.new_norms[modal.mir_loops_x[modal._container.sel_status]] = sel_norms
    if axis == 1:
        modal._container.new_norms[modal.mir_loops_y[modal._container.sel_status]] = sel_norms
    if axis == 2:
        modal._container.new_norms[modal.mir_loops_z[modal._container.sel_status]] = sel_norms

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def incremental_rotate_vectors(modal, axis, direction):
    modal.translate_mode = 2
    modal.translate_axis = axis
    modal._container.cache_norms[:] = modal._container.new_norms
    rotate_vectors(modal, math.radians(direction * modal._rot_increment))
    modal.translate_mode = 0
    modal.translate_axis = 2

    modal.redraw = True
    return


def rotate_vectors(modal, angle):
    if modal.translate_axis == 0:
        axis = 'X'
    if modal.translate_axis == 1:
        axis = 'Y'
    if modal.translate_axis == 2:
        axis = 'Z'

    rot = np.array(Matrix.Rotation(angle, 3, axis))

    # Viewspace rotation matrix
    if modal.translate_mode == 0:
        persp_mat = bpy.context.region_data.view_matrix.to_3x3().normalized()
        ob_mat = modal._object.matrix_world.to_3x3().normalized()

        modal._container.new_norms[modal._container.sel_status] = (np.array(
            ob_mat) @ modal._container.cache_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (np.array(
            persp_mat) @ modal._container.new_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (
            rot @ modal._container.new_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (np.array(
            persp_mat.inverted()) @ modal._container.new_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (np.array(
            ob_mat.inverted()) @ modal._container.new_norms[modal._container.sel_status].T).T

    # World space rotation matrix
    elif modal.translate_mode == 1:
        ob_mat = modal._object.matrix_world.to_3x3().normalized()

        modal._container.new_norms[modal._container.sel_status] = (np.array(
            ob_mat) @ modal._container.cache_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (
            rot @ modal._container.new_norms[modal._container.sel_status].T).T
        modal._container.new_norms[modal._container.sel_status] = (np.array(
            ob_mat.inverted()) @ modal._container.new_norms[modal._container.sel_status].T).T

    # Local space roatation matrix
    elif modal.translate_mode == 2:
        if modal.gizmo_click:
            orb_mat = modal._orbit_ob.matrix_world.to_3x3().normalized()
            ob_mat = modal._object.matrix_world.to_3x3().normalized()

            modal._container.new_norms[modal._container.sel_status] = (np.array(
                ob_mat) @ modal._container.cache_norms[modal._container.sel_status].T).T
            modal._container.new_norms[modal._container.sel_status] = (np.array(
                orb_mat.inverted()) @ modal._container.new_norms[modal._container.sel_status].T).T
            modal._container.new_norms[modal._container.sel_status] = (
                rot @ modal._container.new_norms[modal._container.sel_status].T).T
            modal._container.new_norms[modal._container.sel_status] = (
                np.array(orb_mat) @ modal._container.new_norms[modal._container.sel_status].T).T
            modal._container.new_norms[modal._container.sel_status] = (np.array(
                ob_mat.inverted()) @ modal._container.new_norms[modal._container.sel_status].T).T

        else:
            modal._container.new_norms[modal._container.sel_status] = (
                rot @ modal._container.cache_norms[modal._container.sel_status].T).T

    set_new_normals(modal)
    return


#
# AXIS ALIGNMENT
#
def flatten_normals(modal, axis):

    norms = modal._container.new_norms[modal._container.sel_status]
    norms[:, axis] = 0.0

    # Check for zero length vector after flattening
    # If zero then set it back to 1.0 on flattened axis
    zero_len = np.sum(np.absolute(norms), axis=1) == 0.0

    # All vecs are zero length so no change occurs
    if zero_len.all():
        return

    if zero_len.any():
        norms[zero_len] = modal._container.new_norms[modal._container.sel_status][zero_len]

    modal._container.new_norms[modal._container.sel_status] = norms

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def align_to_axis_normals(modal, axis, dir):

    vec = [0, 0, 0]
    vec[axis] = dir
    modal._container.new_norms[modal._container.sel_status] = vec

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


#
# MANIPULATE NORMALS
#
def average_vertex_normals(modal):

    sel_pos = get_selected_points(modal, any_selected=True)
    sel_loops = modal._container.vert_link_ls[sel_pos]

    loop_status = modal._container.sel_status[sel_loops]
    loop_status[sel_loops < 0] = False

    cur_norms = modal._container.new_norms[sel_loops]
    cur_norms[~loop_status] = np.nan
    cur_norms = np.nanmean(cur_norms, axis=1)[:, np.newaxis]

    new_norms = modal._container.new_norms[sel_loops]
    new_norms[:] = cur_norms
    sel_loops = sel_loops[loop_status]
    new_norms = new_norms[loop_status]

    modal._container.new_norms[sel_loops] = new_norms

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def average_selected_normals(modal):

    avg_norm = np.mean(
        modal._container.new_norms[modal._container.sel_status], axis=0)

    modal._container.new_norms[modal._container.sel_status] = avg_norm

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return

def smooth_normals(modal, fac):

    sel_pos = get_selected_points(modal, any_selected=True)
    conn_pos = modal._container.vert_link_vs[sel_pos]
    sel_loops = modal._container.vert_link_ls[sel_pos]
    sel_mask = sel_loops >= 0

    for i in range(modal._smooth_iterations):

        conn_norms = modal._container.new_norms[modal._container.vert_link_ls[conn_pos]]
        conn_norms[modal._container.vert_link_ls[conn_pos] < 0] = np.nan
        conn_norms[conn_pos < 0] = np.nan
        conn_norms = np.nanmean(conn_norms, axis=(1, 2))[:, np.newaxis]
        conn_norms = modal._container.new_norms[sel_loops] * (1.0-fac*modal._smooth_strength) + conn_norms*(fac*modal._smooth_strength)

        conn_norms = conn_norms[sel_mask]

        modal._container.new_norms[sel_loops[sel_mask]] = conn_norms

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


#
# NORMAL DIRECTION
#
def flip_normals(modal):
    modal._container.new_norms[modal._container.sel_status] *= -1

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def set_outside_inside(modal, direction):

    if modal._object_smooth:
        sel_pos = get_selected_points(modal, any_selected=True)
        sel_loops = modal._container.vert_link_ls[sel_pos]

        f_norms = modal._container.face_normals[modal._container.loop_faces[sel_loops]]
        f_norms[sel_loops < 0] = np.nan

        loop_status = modal._container.sel_status[sel_loops]
        loop_status[sel_loops < 0] = False

        f_norms[~loop_status] = np.nan
        f_norms = np.nanmean(f_norms, axis=1)[:, np.newaxis]
        new_norms = modal._container.new_norms[sel_loops]

        new_norms[:] = f_norms

        sel_loops = sel_loops[loop_status]
        new_norms = new_norms[loop_status] * direction

        modal._container.new_norms[sel_loops] = new_norms

    else:
        modal._container.new_norms[modal._container.sel_status] = modal._container.face_normals[
            modal._container.loop_faces[modal._container.sel_status]]

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def reset_normals(modal):
    modal._container.new_norms[modal._container.sel_status] = modal._container.og_norms[modal._container.sel_status]

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def set_normals_from_faces(modal):

    # Get all faces that have all loops selected
    sel_fs = modal._container.sel_status[modal._container.face_link_ls]
    sel_fs[modal._container.face_link_ls < 0] = True
    sel_fs = sel_fs.all(axis=1).nonzero()[0]

    # Get the unique vertices of these faces in a flat array
    face_vs = modal._container.face_link_vs[sel_fs].ravel()
    face_vs = np.unique(face_vs[face_vs >= 0])

    # Get the loops of each vert
    vert_ls = modal._container.vert_link_ls[face_vs]

    # Get the faces fand face normals of these loops
    loop_fs = modal._container.loop_faces[vert_ls]
    face_l_norms = modal._container.face_normals[loop_fs]

    # Find which of these loops is valid based on if its face is apart of the fully selected faces
    loop_status = np.in1d(loop_fs, sel_fs)
    loop_status.shape = [face_l_norms.shape[0], face_l_norms.shape[1]]
    loop_status[vert_ls < 0] = False

    # Remove loop face normals for non valid loops and average per vertex
    face_l_norms[~loop_status] = np.nan
    face_l_norms = np.nanmean(face_l_norms, axis=1)

    # Create array of vertex averaged normals for all of the loops connected to the verts
    new_norms = modal._container.new_norms[vert_ls]
    new_norms[:] = face_l_norms[:, np.newaxis]

    # Filter out loops that should not be set
    # Filter by verts if individual loops is off and by loops if it is on
    if modal._individual_loops:
        new_norms = new_norms[loop_status]
        vert_ls = vert_ls[loop_status]
    else:
        new_norms = new_norms[vert_ls >= 0]
        vert_ls = vert_ls[vert_ls >= 0]

    modal._container.new_norms[vert_ls] = new_norms

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


#
# COPY/PASTE
#
def copy_active_to_selected(modal):

    act_loops = modal._container.act_status.nonzero()[0]

    # 1 active loop so paste it onto all selected
    if act_loops.size == 1:
        modal._container.new_norms[modal._container.sel_status] = modal._container.new_norms[act_loops[0]]

    # Find active po and match the tangents of this po to the selected loops
    else:
        act_po = get_active_point(modal)
        act_ls = modal._container.vert_link_ls[act_po]
        act_ls = act_ls[act_ls >= 0]

        sel_loops = modal._container.sel_status
        sel_loops[act_ls] = False

        match_pos = [act_po]*sel_loops.nonzero()[0].size
        target_tans = modal._container.loop_tangents[modal._container.vert_link_ls[match_pos]]

        loop_matches = match_loops_vecs(
            modal._container.loop_tangents[sel_loops], target_tans, modal._container.vert_link_ls[match_pos])

        modal._container.new_norms[sel_loops] = modal._container.new_norms[loop_matches]

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


def store_active_normal(modal):
    act_loops = modal._container.act_status.nonzero()[0]

    # 1 active loop so paste it onto all selected
    if act_loops.size == 1:
        modal._copy_normals = act_loops

    # Find active po and match the tangents of this po to the selected loops
    else:
        act_po = get_active_point(modal)
        act_ls = modal._container.vert_link_ls[act_po]
        act_ls = act_ls[act_ls >= 0]

        modal._copy_normals = act_ls
    return


def paste_normal(modal):

    if modal._copy_normals.size > 0:
        # 1 active loop so paste it onto all selected
        if modal._copy_normals.size == 1:
            modal._container.new_norms[modal._container.sel_status] = modal._container.new_norms[modal._copy_normals[0]]

        # Find active po and match the tangents of this po to the selected loops
        else:
            sel_loops = modal._container.sel_status
            sel_loops[modal._copy_normals] = False

            target_tans = np.tile(
                modal._container.loop_tangents[modal._copy_normals], (sel_loops.nonzero()[0].size, 1, 1))
            target_inds = np.tile(modal._copy_normals,
                                  (sel_loops.nonzero()[0].size, 1))

            loop_matches = match_loops_vecs(
                modal._container.loop_tangents[sel_loops], target_tans, target_inds)

            modal._container.new_norms[sel_loops] = modal._container.new_norms[loop_matches]

    set_new_normals(modal)
    add_to_undostack(modal, 1)
    return


#
#


def translate_axis_draw(modal):
    mat = None
    if modal.translate_mode == 0:
        modal.translate_draw_line.clear()

    elif modal.translate_mode == 1:
        mat = generate_matrix(Vector((0, 0, 0)), Vector(
            (0, 0, 1)), Vector((0, 1, 0)), False, True)
        mat.translation = modal._mode_cache[0]

    elif modal.translate_mode == 2:
        mat = modal._object.matrix_world.normalized()
        mat.translation = modal._mode_cache[0]

    if mat is not None:
        modal.translate_draw_line.clear()

        if modal.translate_axis == 0:
            vec = Vector((1000, 0, 0))
        if modal.translate_axis == 1:
            vec = Vector((0, 1000, 0))
        if modal.translate_axis == 2:
            vec = Vector((0, 0, 1000))

        modal.translate_draw_line.append(mat @ vec)
        modal.translate_draw_line.append(mat @ -vec)

    modal.batch_translate_line = batch_for_shader(
        modal.shader_3d, 'LINES', {"pos": modal.translate_draw_line})
    return


def clear_translate_axis_draw(modal):
    modal.batch_translate_line = batch_for_shader(
        modal.shader_3d, 'LINES', {"pos": []})
    return


def translate_axis_change(modal, text, axis):
    if modal.translate_axis != axis:
        modal.translate_axis = axis
        modal.translate_mode = 1

    else:
        modal.translate_mode += 1
        if modal.translate_mode == 3:
            modal.translate_mode = 0
            modal.translate_axis = 2

    if modal.translate_mode == 0:
        modal._window.set_status('VIEW ' + text)
    elif modal.translate_mode == 1:
        modal._window.set_status('GLOBAL ' + text)
    else:
        modal._window.set_status('LOCAL ' + text)

    translate_axis_draw(modal)

    return


def translate_axis_side(modal):
    view_vec = view3d_utils.region_2d_to_vector_3d(
        modal.act_reg, modal.act_rv3d, modal._mouse_reg_loc)

    if modal.translate_mode == 1:
        mat = generate_matrix(Vector((0, 0, 0)), Vector(
            (0, 0, 1)), Vector((0, 1, 0)), False, True)
    else:
        mat = modal._object.matrix_world.normalized()

    pos_vec = Vector((0, 0, 0))
    neg_vec = Vector((0, 0, 0))
    pos_vec[modal.translate_axis] = 1.0
    neg_vec[modal.translate_axis] = -1.0

    pos_vec = (mat @ pos_vec) - mat.translation
    neg_vec = (mat @ neg_vec) - mat.translation

    if pos_vec.angle(view_vec) < neg_vec.angle(view_vec):
        side = -1
    else:
        side = 1

    # if modal.translate_axis == 1:
    #     side *= -1
    return side


def cleanup_other_color_attributes(mesh):
    """删除除了 NormalColor 以外的所有顶点色属性，只保留插件写入的法线颜色"""
    if not hasattr(mesh, 'color_attributes'):
        return
    to_remove = [attr.name for attr in mesh.color_attributes if attr.name != "NormalColor"]
    if to_remove:
        for name in to_remove:
            try:
                mesh.color_attributes.remove(mesh.color_attributes[name])
            except Exception as e:
                print(f"移除多余颜色属性 [{name}] 失败: {e}")
        print(f"已清理多余顶点色属性: {to_remove}")


def calc_loop_tangents_safe(mesh, uv_name=None):
    """安全计算每个 loop 的切线和副切线符号，兼容含 n-gon（5边+多边形）的网格。

    Blender 的 mesh.calc_tangents() 内部基于三角化计算切线，其实现只完整支持
    三角形和四边形面；当网格中存在 5 边及以上的多边形（n-gon）时会直接报错。
    此函数在检测到 n-gon 时，会临时三角化网格副本计算切线，再通过顶点索引
    映射回原网格的 loop，保持原网格拓扑不变。

    返回:
        tangents: (loop_amnt*3,) numpy float32 数组，每个 loop 的切线
        signs:    (loop_amnt,) numpy float32 数组，每个 loop 的副切线符号
    """
    loop_amnt = len(mesh.loops)
    has_ngon = any(p.loop_total > 4 for p in mesh.polygons)

    if not has_ngon:
        # 没有 n-gon，直接使用 Blender 原生切线计算
        if uv_name and uv_name in mesh.uv_layers:
            mesh.calc_tangents(uvmap=uv_name)
        else:
            mesh.calc_tangents()
        tangents = np.zeros(loop_amnt * 3, dtype=np.float32)
        mesh.loops.foreach_get("tangent", tangents)
        signs = np.zeros(loop_amnt, dtype=np.float32)
        mesh.loops.foreach_get("bitangent_sign", signs)
        return tangents, signs

    # 有 n-gon：临时三角化网格副本计算切线，再映射回原 loop
    import bmesh
    from collections import defaultdict

    temp_mesh = mesh.copy()
    bm = bmesh.new()
    bm.from_mesh(temp_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='FIXED', ngon_method='BEAUTY')
    bm.to_mesh(temp_mesh)
    bm.free()

    if uv_name and uv_name in temp_mesh.uv_layers:
        temp_mesh.calc_tangents(uvmap=uv_name)
    else:
        temp_mesh.calc_tangents()

    tri_loop_amnt = len(temp_mesh.loops)
    tri_tangents = np.zeros(tri_loop_amnt * 3, dtype=np.float32)
    temp_mesh.loops.foreach_get("tangent", tri_tangents)
    tri_signs = np.zeros(tri_loop_amnt, dtype=np.float32)
    temp_mesh.loops.foreach_get("bitangent_sign", tri_signs)

    # 建立 顶点索引 -> 三角化副本中 loop 列表 的映射（三角化不新增顶点，索引保持一致）
    vert_to_loops = defaultdict(list)
    for j, l in enumerate(temp_mesh.loops):
        vert_to_loops[l.vertex_index].append(j)

    tangents = np.zeros(loop_amnt * 3, dtype=np.float32)
    signs = np.zeros(loop_amnt, dtype=np.float32)
    for i, l in enumerate(mesh.loops):
        j = vert_to_loops[l.vertex_index][0]
        tangents[i * 3:i * 3 + 3] = tri_tangents[j * 3:j * 3 + 3]
        signs[i] = tri_signs[j]

    bpy.data.meshes.remove(temp_mesh)
    return tangents, signs


def triangulate_ngons(mesh):
    """只三角化网格中的 n-gon（5边及以上）面，三角形和四边形面保持不动。

    游戏引擎（如 Unity）导入时会对 n-gon 自动三角化，拆分后产生独立的
    三角形顶点，导致 Blender 的 per-corner 切线空间数据与引擎顶点无法
    一一对应，描边效果出错。提前只三角化 n-gon 面，能让顶点数据完全对齐引擎。

    返回: 是否执行了三角化
    """
    if mesh is None:
        return False

    ngon_count = sum(1 for p in mesh.polygons if p.loop_total > 4)
    if ngon_count == 0:
        return False

    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # 只选择 n-gon 面，三角形/四边形不处理
    faces = [f for f in bm.faces if len(f.verts) > 4]
    if faces:
        bmesh.ops.triangulate(bm, faces=faces, quad_method='FIXED', ngon_method='BEAUTY')
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()
    print(f"已自动三角化 {ngon_count} 个多边形面（n-gon），保证描边数据与引擎对齐")
    return True


#
# MODAL 缓存顶点数据的方法(初始化的地方)
#
def cache_point_data(modal):
    # 保存当前模式状态
    # modal.mode_state = ModeState()
    # modal.mode_state.save(bpy.context)

    modal._object.data.calc_normals_split()

    vert_amnt = len(modal._object.data.vertices)
    edge_amnt = len(modal._object.data.edges)
    loop_amnt = len(modal._object.data.loops)
    face_amnt = len(modal._object.data.polygons)
    mesh = modal._object.data
    # 安全计算切线（兼容含 n-gon 的网格）
    calc_loop_tangents_safe(mesh, None)
    
    modal._container.og_sharp = np.zeros(edge_amnt, dtype=bool)
    modal._object.data.edges.foreach_get(
        'use_edge_sharp', modal._container.og_sharp)

    modal._container.og_seam = np.zeros(edge_amnt, dtype=bool)
    modal._object.data.edges.foreach_get('use_seam', modal._container.og_seam)

    # 初始化获取到法线数据，存入到指定的数组中，
    modal._container.og_norms = np.zeros(loop_amnt*3, dtype=np.float32)
    modal._object.data.loops.foreach_get('normal', modal._container.og_norms)
    modal._container.og_norms.shape = [loop_amnt, 3]

    # 获取或创建自定义属性（域为 CORNER，对应每个循环）
    customNormal_attr = mesh.attributes.get("CustomNormal")
    if not customNormal_attr:
        # 创建属性（类型为 FLOAT_VECTOR，域为 CORNER）
        customNormal_attr = mesh.attributes.new(name="CustomNormal",type='FLOAT_VECTOR',domain='CORNER')

    # 初始化 NumPy 数组
    modal._container.custom_norms = np.zeros(loop_amnt * 3, dtype=np.float32)
    # 从属性中批量读取数据（关键步骤）
    # 使用 data.foreach_get 直接操作底层数据，把CustonNormal数据存入到自定义的custom_norms列表中
    customNormal_attr.data.foreach_get("vector", modal._container.custom_norms)
    # 重塑数组形状为 [loop_amnt, 3]
    modal._container.custom_norms.shape = (loop_amnt, 3)

    # 检查是否存在NaN值
    has_nan = np.isnan(modal._container.custom_norms).any()
    all_zero = np.all(modal._container.custom_norms == 0)

    if has_nan or all_zero:
        # 使用原始法线数据
        # 同时把原始法线数据复制到customNormal上，防止第一次启用编辑的时候没有绘制出描边效果
        modal._container.custom_norms = modal._container.og_norms.copy()
        # customNormal_attr.data
        # 循环存储数据
        for i, loop in enumerate(modal._object.data.loops):
            customNormal_attr.data[i].vector = modal._container.og_norms.copy()[i]

        modal._container.new_norms = modal._container.og_norms.copy()
        modal._container.cache_norms = modal._container.og_norms.copy()
    else:
        # 使用自定义法线数据
        modal._container.new_norms = modal._container.custom_norms.copy()
        modal._container.cache_norms = modal._container.custom_norms.copy()

    # 清理多余的顶点色属性（只保留插件写入的 NormalColor）
    if modal._display_prefs.cleanup_other_colors:
        cleanup_other_color_attributes(mesh)

    # 在这里同时也设置好顶点色
    color_attr = mesh.color_attributes.get("NormalColor")
    # 检查属性是否存在，且类型是否为 FLOAT_COLOR
    if color_attr and color_attr.data_type != 'FLOAT_COLOR':
        mesh.color_attributes.remove(color_attr)
        color_attr = None

    if not color_attr:
        color_attr = mesh.color_attributes.new(name="NormalColor", type='FLOAT_COLOR', domain='CORNER')
        # 确保数据被初始化，设置默认值
        for loop in mesh.loops:
            color_attr.data[loop.index].color = (0.5, 0.5, 0.5, 1.0)

    uv_name = None
    if len(mesh.uv_layers) > 0:
        uv_layer = mesh.uv_layers.active
        if uv_layer is None and "UVMap" in mesh.uv_layers:
            uv_layer = mesh.uv_layers["UVMap"]
        if uv_layer is None:
            uv_layer = mesh.uv_layers[0]
        uv_name = uv_layer.name

    loops = mesh.loops
    loop_amnt = len(loops)
    data_len = len(color_attr.data)
    if data_len != loop_amnt:
        mesh.color_attributes.remove(color_attr)
        color_attr = mesh.color_attributes.new(name="NormalColor", type='FLOAT_COLOR', domain='CORNER')
        data_len = len(color_attr.data)

    loop_nors = np.zeros(loop_amnt * 3, dtype=np.float32)
    loops.foreach_get("normal", loop_nors)
    loop_nors.shape = (loop_amnt, 3)

    # 安全计算切线（兼容含 n-gon 的网格）
    loop_tans, bitan_sign = calc_loop_tangents_safe(mesh, uv_name)
    loop_tans.shape = (loop_amnt, 3)

    nor_len = np.linalg.norm(loop_nors, axis=1)
    nor_len[nor_len == 0] = 1.0
    loop_nors = loop_nors / nor_len[:, None]

    tan_len = np.linalg.norm(loop_tans, axis=1)
    tan_len[tan_len == 0] = 1.0
    loop_tans = loop_tans / tan_len[:, None]

    loop_bitans = bitan_sign[:, None] * np.cross(loop_nors, loop_tans)
    bitan_len = np.linalg.norm(loop_bitans, axis=1)
    bitan_len[bitan_len == 0] = 1.0
    loop_bitans = loop_bitans / bitan_len[:, None]

    cur_norms = modal._container.new_norms[:loop_amnt].astype(np.float32, copy=False)
    ts_x = np.sum(cur_norms * loop_tans, axis=1)
    ts_y = np.sum(cur_norms * loop_bitans, axis=1)
    ts_z = np.sum(cur_norms * loop_nors, axis=1)

    colors = np.empty((loop_amnt, 4), dtype=np.float32)
    colors[:, 0] = ts_x * 0.5 + 0.5
    colors[:, 1] = ts_y * 0.5 + 0.5
    colors[:, 2] = ts_z * 0.5 + 0.5
    colors[:, 3] = 1.0
    colors[:, :3] = np.clip(colors[:, :3], 0.0, 1.0)

    color_attr.data.foreach_set("color", colors[:data_len].ravel())

    # 同时也要处理描边参数数据和CustonNormal数据也要传递过来

    max_link_eds = max([len(v.link_edges) for v in modal._object_bm.verts])
    max_link_loops = max([len(v.link_loops) for v in modal._object_bm.verts])
    max_link_f_vs = max([len(f.verts) for f in modal._object_bm.faces])
    max_link_f_loops = max([len(f.loops) for f in modal._object_bm.faces])
    max_link_f_eds = max([len(f.edges) for f in modal._object_bm.faces])

    #

    link_vs = []
    link_ls = []
    link_fs = [None for i in range(loop_amnt)]
    for v in modal._object_bm.verts:
        l_v_inds = [-1] * max_link_eds
        l_l_inds = [-1] * max_link_loops

        for e, ed in enumerate(v.link_edges):
            l_v_inds[e] = ed.other_vert(v).index

        for l, loop in enumerate(v.link_loops):
            l_l_inds[l] = loop.index
            link_fs[loop.index] = loop.face.index

        link_vs += l_v_inds
        link_ls += l_l_inds

    modal._container.loop_verts = np.zeros(loop_amnt, dtype=np.int32)
    modal._object.data.loops.foreach_get(
        'vertex_index', modal._container.loop_verts)

    modal._container.loop_edges = np.zeros(loop_amnt, dtype=np.int32)
    modal._object.data.loops.foreach_get(
        'edge_index', modal._container.loop_edges)

    modal._container.vert_link_vs = np.array(link_vs, dtype=np.int32)
    modal._container.vert_link_vs.shape = [vert_amnt, max_link_eds]

    modal._container.vert_link_ls = np.array(link_ls, dtype=np.int32)
    modal._container.vert_link_ls.shape = [vert_amnt, max_link_loops]

    modal._container.loop_faces = np.array(link_fs, dtype=np.int32)
    modal._container.filter_weights = np.zeros(loop_amnt, dtype=np.float32)
    modal._container.filter_mask = np.zeros(loop_amnt, dtype=bool)

    #

    link_f_vs = []
    link_f_ls = []
    link_f_eds = []
    face_normals = []
    l_tangents = [None for i in range(loop_amnt)]
    for f in modal._object_bm.faces:
        l_v_inds = [-1] * max_link_f_vs
        l_l_inds = [-1] * max_link_f_loops
        l_e_inds = [-1] * max_link_f_eds

        for v, vert in enumerate(f.verts):
            l_v_inds[v] = vert.index

        for l, loop in enumerate(f.loops):
            l_l_inds[l] = loop.index
            l_tangents[loop.index] = loop.calc_tangent()
            l_e_inds[l] = loop.edge.index

        face_normals.append(f.normal)

        link_f_vs += l_v_inds
        link_f_ls += l_l_inds
        link_f_eds += l_e_inds

    modal._container.face_link_vs = np.array(link_f_vs, dtype=np.int32)
    modal._container.face_link_vs.shape = [face_amnt, max_link_f_vs]

    modal._container.face_link_ls = np.array(link_f_ls, dtype=np.int32)
    modal._container.face_link_ls.shape = [face_amnt, max_link_f_loops]

    modal._container.face_link_eds = np.array(link_f_eds, dtype=np.int32)
    modal._container.face_link_eds.shape = [face_amnt, max_link_f_eds]

    modal._container.loop_tangents = np.array(l_tangents, dtype=np.float32)
    modal._container.loop_tangents.shape = [loop_amnt, 3]

    modal._container.face_normals = np.array(face_normals, dtype=np.float32)
    modal._container.face_normals.shape = [face_amnt, 3]

    #

    link_ed_f_inds = []
    link_ed_fs = []
    link_ed_vs = []
    for ed in modal._object_bm.edges:
        link_ed_vs.append([ed.verts[0].index, ed.verts[1].index])

        for f in ed.link_faces:
            link_ed_f_inds.append(ed.index)
            link_ed_fs.append(f.index)

    modal._container.edge_link_vs = np.array(link_ed_vs, dtype=np.int32)
    modal._container.edge_link_vs.shape = [edge_amnt, 2]

    modal._container.edge_link_f_inds = np.array(
        link_ed_f_inds, dtype=np.int32)
    modal._container.edge_link_fs = np.array(link_ed_fs, dtype=np.int32)

    #

    modal._container.po_coords = np.array(
        [v.co for v in modal._object_bm.verts], dtype=np.float32)
    modal._container.loop_coords = np.array(
        [modal._object_bm.verts[l.vertex_index].co for l in modal._object.data.loops], dtype=np.float32)

    loop_tri_cos = [[] for i in range(loop_amnt)]
    for v in modal._object_bm.verts:
        ed_inds = [ed.index for ed in v.link_edges]
        for loop in v.link_loops:
            loop_cos = [v.co+v.normal*.001]
            for ed in loop.face.edges:
                if ed.index in ed_inds:
                    ov = ed.other_vert(v)
                    vec = (ov.co+ov.normal*.001) - (v.co+v.normal*.001)

                    loop_cos.append((v.co+v.normal*.001) + vec * 0.5)

            loop_tri_cos[loop.index] = loop_cos

    modal._container.loop_tri_coords = np.array(loop_tri_cos, dtype=np.float32)

    #
    #

    loop_sel = [False] * loop_amnt
    loop_hide = [True] * loop_amnt
    loop_act = [False] * loop_amnt
    for v in modal._object_bm.verts:
        for loop in v.link_loops:
            # Vertex selection
            if bpy.context.tool_settings.mesh_select_mode[0]:
                loop_sel[loop.index] = v.select

            loop_hide[loop.index] = v.hide

    # Edge selection
    if bpy.context.tool_settings.mesh_select_mode[1]:
        for ed in modal._object_bm.edges:
            if ed.select:
                for v in ed.verts:
                    for loop in v.link_loops:
                        loop_sel[loop.index] = True

    # Face selection
    if bpy.context.tool_settings.mesh_select_mode[2]: 
        for f in modal._object_bm.faces:
            if f.select:
                if modal._individual_loops:
                    for loop in f.loops:
                        loop_sel[loop.index] = True
                else:
                    for v in f.verts:
                        for loop in v.link_loops:
                            loop_sel[loop.index] = True

    modal._container.hide_status = np.array(loop_hide, dtype=bool)
    modal._container.sel_status = np.array(loop_sel, dtype=bool)
    modal._container.act_status = np.array(loop_act, dtype=bool)

    cache_mirror_data(modal)

    add_geometry_nodes_outline(modal)

    return

# 获取到预设资源路径
def find_asset_library_path(library_name="Outline Node"):
    """获取指定名称的资产库路径"""
    for lib in bpy.context.preferences.filepaths.asset_libraries:
        if lib.name == library_name:
            return lib.path
    return None

# 获取到预设几何节点
def load_node_group_from_asset_lib(asset_file="OutlinePlus.blend", node_group_name="OutlinePlus"):
    """优先使用本地已存在的节点组，避免重复加载"""
    # 先检查本地是否存在
    existing_group = bpy.data.node_groups.get(node_group_name)
    if existing_group:
        return existing_group
    
    # 若不存在则从资产库加载
    asset_lib_path = find_asset_library_path()
    if not asset_lib_path:
        print("未找到资产库路径")
        return None
        
    blend_path = os.path.join(asset_lib_path, asset_file)
    if not os.path.exists(blend_path):
        print(f"资产文件不存在: {blend_path}")
        return None

    # 加载并返回节点组
    with bpy.data.libraries.load(blend_path, link=False) as (data_src, data_dst):
        if node_group_name in data_src.node_groups:
            data_dst.node_groups = [node_group_name]
    
    return bpy.data.node_groups.get(node_group_name)


# 从资产中获取到几何节点并添加到模型上的方法
def add_geometry_nodes_outline(modal):
    obj = modal._object
    if not obj or obj.type != 'MESH':
        return
    
    # 检查是否已存在该修改器
    mod = next((m for m in obj.modifiers if m.type == 'NODES' and m.node_group and m.node_group.name == "OutlinePlus"), None)
    if mod:
        return
    
    # 确保节点组存在（优先复用本地）
    outline_group = load_node_group_from_asset_lib()
    if not outline_group:
        print("加载几何节点组失败")
        return
    
    # 创建新修改器
    mod = obj.modifiers.new(name="OutlinePlus", type='NODES')
    mod.node_group = outline_group
    print("轮廓几何节点修改器已添加")


def cache_mirror_data(modal):
    mat = np.array(modal._object.matrix_world)
    mat_inv = np.array(modal._object.matrix_world.inverted())

    loc = mat[:3, 3]
    mir_cos = (mat_inv[:3, :3] @ (modal._container.po_coords-loc).T).T

    kd = create_kd_from_np(modal._container.po_coords)

    modal.mir_loops_x = find_coord_mirror(modal, mir_cos.copy(), 0, mat, kd)
    modal.mir_loops_y = find_coord_mirror(modal, mir_cos.copy(), 1, mat, kd)
    modal.mir_loops_z = find_coord_mirror(modal, mir_cos.copy(), 2, mat, kd)
    return


def find_coord_mirror(modal, mir_coords, mir_axis, mat, kd):
    #
    # Match the list of loops with their mirror on the set axis
    # detect the proper loop on the mirror point by getting the smallest angle
    #

    # Get the loop coords on the mirrored axis
    mir_coords[:, mir_axis] *= -1
    l_coords = (mat[:3, :3] @ mir_coords.T).T+mat[:3, 3]

    # ORIGINAL VERSION SUPER SLOW FOR A 12K VERT MESH 10 SECONDS PER AXIS
    # # Test distance for nearest points from the loops mirroed coord
    # po_matches = get_np_vecs_ordered_dists(
    #     modal._container.po_coords, l_coords)[:, 0]

    # TEST VERSION USING get_np_vec_ordered_dists SLIGHTLY FASTER THAN get_np_vecs_ordered_dists
    # match_inds = []
    # for co in l_coords:
    #     match = get_np_vec_ordered_dists(modal._container.po_coords, co)[0]
    #     match_inds.append(match)
    # po_matches = np.array(match_inds)

    # Use kdtree to find nearest co on the mirror axis
    # Far far far faster than current implementation of get_np_vecs_ordered_dists
    # Despite that it is using a loop it cuts the time down immensely
    po_matches = []
    for co in l_coords:
        res = kd.find(co)
        po_matches.append(res[1])

    po_matches = np.array(po_matches)

    match_loops = modal._container.vert_link_ls[po_matches[modal._container.loop_verts]]

    # Get the tangents of the matched mirror coord
    tans = modal._container.loop_tangents[match_loops]

    mir_tangs = modal._container.loop_tangents.copy()
    mir_tangs[:, mir_axis] *= -1

    # Test the dot products for the smallest of the current loop to the matched points loops
    # filters out -1 mathc loop indices
    match_tans = match_loops_vecs(mir_tangs, tans, match_loops)
    return match_tans


def init_nav_list(modal):
    modal.nav_list = []

    names = ['Zoom View', 'Rotate View', 'Pan View', 'Dolly View',
             'View Selected', 'View Camera Center', 'View All', 'View Axis',
             'View Orbit', 'View Roll', 'View Persp/Ortho', 'View Camera', 'Frame Selected']
    key_settings = []

    config = bpy.context.window_manager.keyconfigs.user
    if config:
        for item in config.keymaps['3D View'].keymap_items:
            if item.name in names:
                if [item.name, item.type, item.value, item.any, item.ctrl, item.shift, item.alt] not in key_settings:
                    key_settings.append(
                        [item.name, item.type, item.value, item.any, item.ctrl, item.shift, item.alt])
                    modal.nav_list.append(item)

    # Add basic pass thru keys
    for item in modal.keymap.keymap_items:
        if 'Pass Thru' in item.name:
            modal.nav_list.append(item)
    return

def ob_data_structures(modal, ob):
    if ob.data.shape_keys is not None:
        for sk in ob.data.shape_keys.key_blocks:
            modal._objects_sk_vis.append(sk.mute)
            sk.mute = True

    bm = create_simple_bm(modal, ob)

    bvh = BVHTree.FromBMesh(bm)

    kd = create_kd(bm)

    return bm, kd, bvh


def add_to_undostack(modal, stack_type):
    if modal._history_position > 0:
        while modal._history_position > 0:
            if modal._history_stack[0] == 0:
                modal._history_select_stack.pop(0)
                modal._history_select_position -= 1
            elif modal._history_stack[0] == 1:
                modal._history_normal_stack.pop(0)
                modal._history_normal_position -= 1
            elif modal._history_stack[0] == 2:
                modal._history_filter_stack.pop(0)
                modal._history_filter_position -= 1

            modal._history_stack.pop(0)
            modal._history_position -= 1

    if len(modal._history_stack)+1 > modal._history_steps:
        if modal._history_stack[-1] == 0:
            modal._history_select_stack.pop(-1)
        elif modal._history_stack[-1] == 1:
            modal._history_normal_stack.pop(-1)
        elif modal._history_stack[-1] == 2:
            modal._history_filter_stack.pop(-1)

        modal._history_stack.pop(-1)

    # Selection status
    if stack_type == 0:
        sel_status = modal._container.sel_status.nonzero()[0]
        vis_status = modal._container.hide_status.nonzero()[0]
        act_status = modal._container.act_status.nonzero()[0]

        modal._history_stack.insert(0, stack_type)
        modal._history_select_stack.insert(
            0, [sel_status, vis_status, act_status])

        modal.redraw = True
        update_orbit_empty(modal)

    # Normals status
    elif stack_type == 1:
        cur_normals = modal._container.new_norms.copy()

        modal._history_stack.insert(0, stack_type)
        modal._history_normal_stack.insert(0, cur_normals)

    # Filter mask weights status
    elif stack_type == 2:
        cur_mask = modal._container.filter_mask.copy()
        cur_weights = modal._container.filter_weights.copy()

        modal._history_stack.insert(0, stack_type)
        modal._history_filter_stack.insert(0, [cur_mask, cur_weights])

    # Initial status
    elif stack_type == 3:
        sel_status = modal._container.sel_status.nonzero()[0]
        vis_status = modal._container.hide_status.nonzero()[0]
        act_status = modal._container.act_status.nonzero()[0]
        cur_normals = modal._container.new_norms.copy()
        cur_mask = modal._container.filter_mask.copy()
        cur_weights = modal._container.filter_weights.copy()

        modal._history_stack.insert(0, stack_type)

        modal._history_select_stack.insert(
            0, [sel_status, vis_status, act_status])
        modal._history_normal_stack.insert(0, cur_normals)
        modal._history_filter_stack.insert(0, [cur_mask, cur_weights])

        modal.redraw = True
        update_orbit_empty(modal)

    return


def move_undostack(modal, dir):
    if (dir > 0 and len(modal._history_stack)-1 > modal._history_position) or (dir < 0 and modal._history_position > 0):
        if dir > 0:
            state_type = modal._history_stack[modal._history_position]
            modal._history_position += dir
        else:
            modal._history_position += dir
            state_type = modal._history_stack[modal._history_position]

        if state_type == 0:
            modal._history_select_position += dir
            state = modal._history_select_stack[modal._history_select_position]

            modal._container.sel_status[:] = False
            modal._container.sel_status[state[0]] = True

            modal._container.hide_status[:] = False
            modal._container.hide_status[state[1]] = True

            modal._container.act_status[:] = False
            modal._container.act_status[state[2]] = True

            update_orbit_empty(modal)
            modal.redraw = True

        elif state_type == 1:
            modal._history_normal_position += dir
            state = modal._history_normal_stack[modal._history_normal_position]

            modal._container.new_norms[:] = state

            set_new_normals(modal)
            modal.redraw = True

        if state_type == 2:
            modal._history_select_position += dir
            modal._history_normal_position += dir
            sel_state = modal._history_select_stack[modal._history_select_position]
            norm_state = modal._history_normal_stack[modal._history_normal_position]

            modal._container.sel_status[:] = False
            modal._container.sel_status[sel_state[0]] = True

            modal._container.hide_status[:] = False
            modal._container.hide_status[sel_state[1]] = True

            modal._container.act_status[:] = False
            modal._container.act_status[sel_state[2]] = True

            modal._container.new_norms[:] = norm_state

            set_new_normals(modal)
            update_orbit_empty(modal)
            modal.redraw = True

    return


def img_load(img_name, path):
    script_file = os.path.realpath(path)
    directory = os.path.dirname(script_file)

    img_fp = directory.replace('/', '\\') + '\\' + img_name

    not_there = True
    for img in bpy.data.images:
        if img.filepath == img_fp:
            not_there = False
            break

    if not_there:
        img = bpy.data.images.load(img_fp)
    try:
        img.colorspace_settings.name = 'Raw'
    except:
        pass

    if img.gl_load():
        raise Exception()

    return img

# 结束法线编辑操作
def finish_modal(modal, restore):
    print("结束编辑")

    mesh = modal._object.data
    # 安全计算切线（兼容含 n-gon 的网格）
    calc_loop_tangents_safe(mesh, None)

    # 恢复保存的模式状态
    if hasattr(modal, 'mode_state') and modal.mode_state is not None:
        modal.mode_state.restore(bpy.context)

    # 将Blender语言切换回为中文
    prefs = bpy.context.preferences
    # 设置中文界面并启用翻译
    prefs.view.language = 'zh_CN'              # 中文语言代码
    prefs.view.use_translate_interface = True  # 启用界面翻译
    prefs.view.use_translate_tooltips = True   # 启用工具提示翻译

    # 保存偏好设置
    modal._behavior_prefs.rotate_gizmo_use = modal._use_gizmo
    modal._display_prefs.gizmo_size = modal._gizmo_size

    modal._display_prefs.totalOutline_width = modal._totalOutline_width
    modal._display_prefs.outline_width_multiplier = modal._outline_width_multiplier
    modal._display_prefs.outline_width_multiplier_locked = modal._outline_width_multiplier_locked
    modal._display_prefs.outline_scale = modal._outline_scale
    modal._display_prefs.z_offset = modal._z_offset

    modal._display_prefs.normal_size = modal._normal_size
    modal._display_prefs.line_brightness = modal._line_brightness
    modal._display_prefs.point_size = modal._point_size
    modal._display_prefs.loop_tri_size = modal._loop_tri_size
    modal._display_prefs.selected_only = modal._selected_only
    modal._display_prefs.draw_weights = modal._draw_weights
    modal._display_prefs.selected_scale = modal._selected_scale
    modal._behavior_prefs.individual_loops = modal._individual_loops
    modal._display_prefs.ui_scale = modal._ui_scale
    modal._display_prefs.display_wireframe = modal._use_wireframe_overlay

    # 还原3D视图的显示选项
    if bpy.context.area is not None:
        if bpy.context.area.type == 'VIEW_3D':
            for space in bpy.context.area.spaces:
                if space.type == 'VIEW_3D':
                    space.show_region_toolbar = modal._reg_header
                    space.show_region_ui = modal._reg_ui
                    space.overlay.show_cursor = modal._cursor
                    space.overlay.show_wireframes = modal._wireframe
                    space.overlay.wireframe_threshold = modal._thresh
                    space.overlay.show_text = modal._text

    bpy.context.window.cursor_modal_set('DEFAULT')

    # 清除绘制的句柄
    clear_drawing(modal)

    # 如果为true则恢复对象法线数据
    if restore:
        ob = modal._object
        if ob.as_pointer() != modal._object_pointer:
            for o_ob in bpy.data.objects:
                if o_ob.as_pointer() == modal._object_pointer:
                    ob = o_ob

        modal._object.data.edges.foreach_set(
            'use_edge_sharp', modal._container.og_sharp)

        # restore normals
        ob.data.normals_split_custom_set(modal._container.og_norms)

    restore_modifiers(modal)

    addon_prefs = bpy.context.preferences.addons[__package__.split('.')[
        0]].preferences
    addon_prefs.object = ''

    delete_orbit_empty(modal)
    if modal._target_emp is not None:
        try:
            bpy.data.objects.remove(modal._target_emp)
        except:
            modal._target_emp = None

    modal._object.select_set(True)
    bpy.context.view_layer.objects.active = modal._object

    # 在这里添加上删除几何节点的操作
    remove_geometry_nodes_outline(modal)

    # 清理多余的顶点色属性（只保留插件写入的 NormalColor）
    if modal._display_prefs.cleanup_other_colors:
        cleanup_other_color_attributes(mesh)

    # 在结束前再次写入法线数据到顶点色（以日志方式处理异常，不抛出错误）
    try:
        mesh = modal._object.data
        color_attr = mesh.color_attributes.get("NormalColor")

        # 确保属性类型始终为 FLOAT_COLOR
        if color_attr and color_attr.data_type != 'FLOAT_COLOR':
            mesh.color_attributes.remove(color_attr)
            color_attr = None

        if not color_attr:
            # 创建使用 FLOAT_COLOR 类型的颜色属性
            color_attr = mesh.color_attributes.new(name="NormalColor", type='FLOAT_COLOR', domain='CORNER')
            # 确保数据被初始化，设置默认值
            for loop in mesh.loops:
                color_attr.data[loop.index].color = (0.5, 0.5, 0.5, 1.0)

        if color_attr and len(color_attr.data) > 0:
            mesh.calc_normals_split()
            uv_name = mesh.uv_layers[0].name if len(mesh.uv_layers) > 0 else None

            loops = mesh.loops
            loop_amnt = len(loops)
            data_len = len(color_attr.data)
            if data_len != loop_amnt:
                mesh.color_attributes.remove(color_attr)
                color_attr = mesh.color_attributes.new(name="NormalColor", type='FLOAT_COLOR', domain='CORNER')
                data_len = len(color_attr.data)

            loop_nors = np.zeros(loop_amnt * 3, dtype=np.float32)
            loops.foreach_get("normal", loop_nors)
            loop_nors.shape = (loop_amnt, 3)

            # 安全计算切线（兼容含 n-gon 的网格）
            loop_tans, bitan_sign = calc_loop_tangents_safe(mesh, uv_name)
            loop_tans.shape = (loop_amnt, 3)

            nor_len = np.linalg.norm(loop_nors, axis=1)
            nor_len[nor_len == 0] = 1.0
            loop_nors = loop_nors / nor_len[:, None]

            tan_len = np.linalg.norm(loop_tans, axis=1)
            tan_len[tan_len == 0] = 1.0
            loop_tans = loop_tans / tan_len[:, None]

            loop_bitans = bitan_sign[:, None] * np.cross(loop_nors, loop_tans)
            bitan_len = np.linalg.norm(loop_bitans, axis=1)
            bitan_len[bitan_len == 0] = 1.0
            loop_bitans = loop_bitans / bitan_len[:, None]

            cur_norms = modal._container.new_norms[:loop_amnt].astype(np.float32, copy=False)
            ts_x = np.sum(cur_norms * loop_tans, axis=1)
            ts_y = np.sum(cur_norms * loop_bitans, axis=1)
            ts_z = np.sum(cur_norms * loop_nors, axis=1)

            colors = np.empty((loop_amnt, 4), dtype=np.float32)
            colors[:, 0] = ts_x * 0.5 + 0.5
            colors[:, 1] = ts_y * 0.5 + 0.5
            colors[:, 2] = ts_z * 0.5 + 0.5
            colors[:, 3] = 1.0
            colors[:, :3] = np.clip(colors[:, :3], 0.0, 1.0)

            color_attr.data.foreach_set("color", colors[:data_len].ravel())
        else:
            try:
                modal.report({'INFO'}, "未找到 NormalColor 或颜色数据为空，已跳过写入")
            except Exception:
                print("未找到 NormalColor 或颜色数据为空，已跳过写入")
    except Exception as e:
        try:
            modal.report({'WARNING'}, f"写入 NormalColor 失败：{e}")
        except Exception:
            print(f"写入 NormalColor 失败：{e}")

    return


# 删除几何节点的方法
def remove_geometry_nodes_outline(modal):
    """安全移除几何节点修改器及关联数据"""
    obj = modal._object
    
    # 防御性检查
    if not obj or not hasattr(obj, 'modifiers') or obj.type != 'MESH':
        return False

    removed = False
    modifiers_to_remove = []

    # 第一阶段：收集需要移除的修改器
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group and "OutlinePlus" in mod.node_group.name:
            modifiers_to_remove.append(mod)

    # 第二阶段：实际移除操作
    for mod in modifiers_to_remove:
        node_group = mod.node_group
        
        try:
            # 提前获取需要的信息
            group_name = node_group.name if node_group else ""
            obj.modifiers.remove(mod)
            
            # 安全清理节点组
            if node_group and node_group.users == 0:
                try:
                    bpy.data.node_groups.remove(node_group)
                    print(f"清理未使用节点组: {group_name}")
                except Exception as e:
                    print(f"清理节点组失败: {str(e)}")
            
            removed = True
        except ReferenceError as e:
            print(f"移除修改器时遇到引用错误: {str(e)}")
        except Exception as e:
            print(f"意外错误: {str(e)}")

    return removed


def restore_modifiers(modal):
    if modal._object.data.shape_keys is not None:
        for s in range(len(modal._object.data.shape_keys.key_blocks)):
            modal._object.data.shape_keys.key_blocks[s].mute = modal._objects_sk_vis[s]

    # restore modifier status
    for m, mod_dat in enumerate(modal._objects_mod_status):
        for mod in modal._object.modifiers:
            if mod.name == modal._objects_mod_status[m][2]:
                mod.show_viewport = modal._objects_mod_status[m][0]
                mod.show_render = modal._objects_mod_status[m][1]
                break

    return


def check_area(modal):
    # # inside region check
    # if modal._mouse_reg_loc[0] >= 0.0 and modal._mouse_reg_loc[0] <= bpy.context.area.width and modal._mouse_reg_loc[1] >= 0.0 and modal._mouse_reg_loc[1] <= bpy.context.area.height:
    #     return bpy.context.region, bpy.context.region_data

    # # if not inside check other areas to find if we are in another region that is a valid view_3d and return that ones data
    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D' and area != modal._draw_area:
    #         if area.spaces.active.type == 'VIEW_3D':
    #             if modal._mouse_abs_loc[0] > area.x and modal._mouse_abs_loc[0] < area.x+area.width and modal._mouse_abs_loc[1] > area.y and modal._mouse_abs_loc[1] < area.y+area.height:
    #                 for region in area.regions:
    #                     if region.type == 'WINDOW':
    #                         return region, area.spaces.active.region_3d

    return bpy.context.region, bpy.context.region_data


def update_filter_from_vg(modal):

    addon_prefs = bpy.context.preferences.addons[__package__.split('.')[
        0]].preferences

    if addon_prefs.vertex_group != '' and addon_prefs.vertex_group != modal._current_filter:
        if addon_prefs.vertex_group in modal._object.vertex_groups:

            for v in modal._object_bm.verts:
                vg = modal._object.vertex_groups[addon_prefs.vertex_group]
                modal._current_filter = addon_prefs.vertex_group

                try:
                    weight = vg.weight(v.index)
                    modal._container.filter_weights[modal._container.vert_link_ls[v.index]] = weight

                except:
                    modal._container.filter_weights[modal._container.vert_link_ls[v.index]] = 0.0

            modal._container.filter_mask[:] = False
            modal._container.filter_mask[modal._container.filter_weights > 0.0] = True

        else:
            modal._current_filter = ''
            addon_prefs.vertex_group = ''

        modal._container.cache_norms[:] = modal._container.new_norms

        modal.redraw = True

    return


def selection_to_filer_mask(modal):
    modal._container.filter_mask[:] = modal._container.sel_status
    modal._container.filter_weights[modal._container.sel_status] = 1.0
    modal._container.filter_weights[~modal._container.sel_status] = 0.0
    modal._current_filter = ''

    modal._container.cache_norms[:] = modal._container.new_norms

    modal.redraw = True
    return


def clear_filter_mask(modal):
    modal._container.filter_mask[:] = False
    modal._container.filter_weights[:] = 0.0
    modal._current_filter = ''

    modal._container.cache_norms[:] = modal._container.new_norms

    modal.redraw = True
    return


#
# GIZMO
#
def gizmo_click_init(modal, event, giz_status):
    if modal._use_gizmo:
        if event.alt == False:
            if modal._container.sel_status.any() == False:
                return True

        # Cache current normals before rotation starts and setup gizmo as being used
        if event.alt == False:

            for g, gizmo in enumerate(modal._window.gizmo_sets[giz_status[1]].gizmos):
                if g != giz_status[2]:
                    gizmo.active = False
                else:
                    gizmo.in_use = True

        orb_mat = modal._orbit_ob.matrix_world

        view_vec = view3d_utils.region_2d_to_vector_3d(
            modal.act_reg, modal.act_rv3d, modal._mouse_reg_loc)
        view_orig = view3d_utils.region_2d_to_origin_3d(
            modal.act_reg, modal.act_rv3d, modal._mouse_reg_loc)

        line_a = view_orig
        line_b = view_orig + view_vec*10000

        # Project cursor from view onto the rotation axis plane
        if giz_status[0] == 'ROT_X':
            giz_vec = orb_mat @ Vector((1, 0, 0)) - orb_mat.translation
            modal.translate_axis = 0

        if giz_status[0] == 'ROT_Y':
            giz_vec = orb_mat @ Vector((0, 1, 0)) - orb_mat.translation
            modal.translate_axis = 1

        if giz_status[0] == 'ROT_Z':
            giz_vec = orb_mat @ Vector((0, 0, 1)) - orb_mat.translation
            modal.translate_axis = 2

        mouse_co_3d = intersect_line_plane(
            line_a, line_b, orb_mat.translation, giz_vec)
        mouse_co_local = orb_mat.inverted() @ mouse_co_3d

        # Get start angle for rotation
        if giz_status[0] == 'ROT_X':
            test_vec = mouse_co_local.yz

        if giz_status[0] == 'ROT_Y':
            test_vec = mouse_co_local.xz

        if giz_status[0] == 'ROT_Z':
            test_vec = mouse_co_local.xy

        modal.translate_mode = 2
        ang_offset = Vector((0, 1)).angle_signed(test_vec)
        modal._mode_cache.append(test_vec)
        # Add cache data for tool mode
        if event.alt == False:
            modal._window.update_gizmo_rot(0, -ang_offset)
            modal._mode_cache.append(0)
            modal._mode_cache.append(-ang_offset)
            modal._mode_cache.append(orb_mat.copy())
            modal._mode_cache.append(True)
        else:
            modal._mode_cache.append(0)
            modal._mode_cache.append(-ang_offset)
            modal._mode_cache.append(orb_mat.copy())
            modal._mode_cache.append(False)

        modal._container.cache_norms[:] = modal._container.new_norms

        keymap_gizmo(modal)
        modal.gizmo_click = True
        modal._current_tool = modal._gizmo_tool
        start_active_drawing(modal)

        return False
    return True


def relocate_gizmo_panel(modal):
    # _gizmo_panel 已移除，Incremental Rotations 内容已移动到"法线方向"面板下
    # 此函数保留以兼容其他代码，但不执行任何操作
    return


def gizmo_update_hide(modal, status):
    if modal._use_gizmo == False:
        status = False

    # _gizmo_panel 已移除，只更新旋转gizmo的可见性
    modal._rot_gizmo.set_visibility(status)
    return


#
# MODES
#
def add_target_empty(ob):
    emp = bpy.data.objects.new('ABN_Target Empty', None)
    emp.empty_display_size = 0.0
    emp.show_in_front = True
    emp.matrix_world = ob.matrix_world.copy()
    emp.empty_display_type = 'SPHERE'
    bpy.context.collection.objects.link(emp)

    return emp


def start_sphereize_mode(modal):

    for i in range(len(bpy.context.selected_objects)):
        bpy.context.selected_objects[0].select_set(False)

    modal._container.cache_norms[:] = modal._container.new_norms

    avg_loc = np.mean(
        modal._container.loop_coords[modal._container.sel_status], axis=0)
    modal._target_emp.location = avg_loc
    modal._target_emp.empty_display_size = 0.5
    modal._target_emp.select_set(True)
    bpy.context.view_layer.objects.active = modal._target_emp

    modal._mode_cache.append(avg_loc)

    gizmo_update_hide(modal, False)
    modal._current_tool = modal._sphereize_tool

    keymap_target(modal)
    modal.sphere_strength.set_value(modal.target_strength)
    modal.sphere_strength.create_shape_data()
    # modal._export_panel.set_visibility(False)
    modal._tools_panel.set_visibility(False)
    modal._modes_panel.set_visibility(False)
    modal._sphere_panel.set_visibility(True)
    modal._sphere_panel.set_new_position(
        modal._mouse_reg_loc, window_dims=modal._window.dimensions)

    sphereize_normals(modal)
    return


def end_sphereize_mode(modal, keep_normals):
    if keep_normals == False:
        modal._container.new_norms[:] = modal._container.cache_norms
        set_new_normals(modal)

    else:
        add_to_undostack(modal, 1)

    # modal._export_panel.set_visibility(True)
    modal._tools_panel.set_visibility(True)
    modal._sphere_panel.set_visibility(False)

    modal.translate_axis = 2
    modal.translate_mode = 0
    clear_translate_axis_draw(modal)
    modal._target_emp.empty_display_size = 0.0
    modal._target_emp.select_set(False)
    modal._orbit_ob.select_set(True)
    bpy.context.view_layer.objects.active = modal._orbit_ob

    modal._mouse_init = None
    modal._mode_cache.clear()
    keymap_refresh(modal)
    modal._current_tool = modal._basic_tool

    gizmo_update_hide(modal, True)
    end_active_drawing(modal)
    return


def sphereize_normals(modal):
    targ_loc = get_np_matrix_transformed_vecs(
        np.array(modal._target_emp.location), modal._object.matrix_world.inverted())
    local_cos = get_np_matrix_transformed_vecs(
        modal._container.loop_coords[modal._container.sel_status], modal._object.matrix_world.inverted())
    
    cache_norms = modal._container.cache_norms[modal._container.sel_status]*(
        1.0-modal.target_strength)
    modal._container.new_norms[modal._container.sel_status] = (
        local_cos - targ_loc)*modal.target_strength + cache_norms

    modal.redraw_active = True

    set_new_normals(modal)
    return


def start_point_mode(modal):

    for i in range(len(bpy.context.selected_objects)):
        bpy.context.selected_objects[0].select_set(False)

    modal._container.cache_norms[:] = modal._container.new_norms

    avg_loc = np.mean(
        modal._container.loop_coords[modal._container.sel_status], axis=0)
    modal._target_emp.location = avg_loc
    modal._target_emp.location[2] += .01
    modal._target_emp.empty_display_size = 0.5
    modal._target_emp.select_set(True)
    bpy.context.view_layer.objects.active = modal._target_emp

    modal._mode_cache.append(avg_loc)

    gizmo_update_hide(modal, False)
    modal._current_tool = modal._point_tool

    keymap_target(modal)
    modal.point_strength.set_value(modal.target_strength)
    modal.point_strength.create_shape_data()
    # modal._export_panel.set_visibility(False)
    modal._tools_panel.set_visibility(False)
    modal._modes_panel.set_visibility(False)
    modal._point_panel.set_visibility(True)
    modal._point_panel.set_new_position(
        modal._mouse_reg_loc, window_dims=modal._window.dimensions)

    point_normals(modal)
    return


def end_point_mode(modal, keep_normals):
    if keep_normals == False:
        modal._container.new_norms[:] = modal._container.cache_norms
        set_new_normals(modal)

    else:
        add_to_undostack(modal, 1)

    # modal._export_panel.set_visibility(True)
    modal._tools_panel.set_visibility(True)
    modal._point_panel.set_visibility(False)

    modal.translate_axis = 2
    modal.translate_mode = 0
    clear_translate_axis_draw(modal)
    modal._target_emp.empty_display_size = 0.0
    modal._target_emp.select_set(False)
    modal._orbit_ob.select_set(True)
    bpy.context.view_layer.objects.active = modal._orbit_ob

    modal._mouse_init = None
    modal._mode_cache.clear()
    keymap_refresh(modal)
    modal._current_tool = modal._basic_tool

    gizmo_update_hide(modal, True)
    end_active_drawing(modal)
    return


def point_normals(modal):
    targ_loc = get_np_matrix_transformed_vecs(
        np.array(modal._target_emp.location), modal._object.matrix_world.inverted())

    cache_norms = modal._container.cache_norms[modal._container.sel_status]*(
        1.0-modal.target_strength)
    if modal.point_align:
        avg_loc = get_np_matrix_transformed_vecs(np.mean(
            modal._container.loop_coords[modal._container.sel_status], axis=0), modal._object.matrix_world.inverted())

        modal._container.new_norms[modal._container.sel_status] = (
            targ_loc - avg_loc)*modal.target_strength + cache_norms

    else:
        local_cos = get_np_matrix_transformed_vecs(
            modal._container.loop_coords[modal._container.sel_status], modal._object.matrix_world.inverted())

        modal._container.new_norms[modal._container.sel_status] = (
            targ_loc - local_cos)*modal.target_strength + cache_norms

    modal.redraw_active = True

    set_new_normals(modal)
    return


def move_target(modal, shift):
    offset = modal._mouse_reg_loc - modal._mode_cache[1]

    if shift:
        offset[0] = offset[0]*.1
        offset[1] = offset[1]*.1

    modal._mode_cache[3][0] = modal._mode_cache[3][0] + offset[0]
    modal._mode_cache[3][1] = modal._mode_cache[3][1] + offset[1]

    new_co = view3d_utils.region_2d_to_location_3d(
        modal.act_reg, modal.act_rv3d, modal._mode_cache[3], modal._mode_cache[2])
    if modal.translate_mode == 0:
        modal._target_emp.location = new_co

    elif modal.translate_mode == 1:
        modal._target_emp.location = modal._mode_cache[2].copy()
        modal._target_emp.location[modal.translate_axis] = new_co[modal.translate_axis]

    elif modal.translate_mode == 2:
        loc_co = modal._object.matrix_world.inverted() @ new_co
        def_dist = loc_co[modal.translate_axis]

        def_vec = Vector((0, 0, 0))
        def_vec[modal.translate_axis] = def_dist

        def_vec = (modal._object.matrix_world @ def_vec) - \
            modal._object.matrix_world.translation

        modal._target_emp.location = modal._mode_cache[2].copy()
        modal._target_emp.location += def_vec

    return


#
# SELECTION
#
def get_hidden_faces(modal):
    if modal._individual_loops:
        # Get indices of faces that all connected loops are hidden
        hid_faces = modal._container.hide_status[modal._container.face_link_ls]
        hid_faces[modal._container.face_link_ls < 0] = False
        hid_faces = hid_faces.any(axis=1)
        hid_faces = list(hid_faces.nonzero()[0])

    else:
        # Get indices of faces that all connected vert loops are hidden
        face_ls = modal._container.vert_link_ls[modal._container.face_link_vs]

        hid_faces = modal._container.hide_status[face_ls]
        hid_faces[face_ls < 0] = False
        hid_faces = hid_faces.any(axis=1)
        hid_faces = list(hid_faces.nonzero()[0])

    return hid_faces


def get_hidden_points(modal):
    # Get indices of points that all connected loops are hidden
    hid_pos = modal._container.hide_status[modal._container.vert_link_ls]
    hid_pos[modal._container.vert_link_ls < 0] = True
    hid_pos = (hid_pos.all(axis=1)).nonzero()[0]
    return hid_pos


def get_hidden_loops(modal):
    # Get indices of loops that are hidden
    hid_ls = modal._container.hide_status.nonzero()[0]
    return hid_ls


def get_visible_points(modal):
    # Get indices of points that all connected loops are visible
    vis_pos = modal._container.hide_status[modal._container.vert_link_ls]
    vis_pos[modal._container.vert_link_ls < 0] = True
    vis_pos = (~vis_pos.all(axis=1)).nonzero()[0]
    return vis_pos


def get_selectable_points(modal):
    # Get indices of points that have a loop that is unselected
    hid_pos = modal._container.hide_status[modal._container.vert_link_ls]
    hid_pos[modal._container.vert_link_ls < 0] = True
    hid_pos = (hid_pos.all(axis=1))

    sel_pos = modal._container.sel_status[modal._container.vert_link_ls]
    sel_pos[modal._container.vert_link_ls < 0] = True
    sel_pos = sel_pos.all(axis=1)
    sel_pos[hid_pos] = True

    sel_pos = (~sel_pos).nonzero()[0]
    return sel_pos


def get_selectable_loops(modal):
    # Get indices of points that have a loop that is unselected
    sel_ls = modal._container.sel_status.copy()
    sel_ls[modal._container.hide_status] = True
    return (~sel_ls).nonzero()[0]


def get_selected_loops(modal):
    # 获取选中且未隐藏的循环索引
    selected_visible = modal._container.sel_status & (~modal._container.hide_status)
    return selected_visible.nonzero()[0]

def get_selected_points(modal, any_selected=False):
    # Get indices of points that all connected loops are visible
    vis_pos = modal._container.sel_status[modal._container.vert_link_ls]
    if any_selected:
        vis_pos[modal._container.vert_link_ls < 0] = False
        vis_pos = vis_pos.any(axis=1)
    else:
        vis_pos[modal._container.vert_link_ls < 0] = True
        vis_pos = vis_pos.all(axis=1)

    vis_pos = vis_pos.nonzero()[0]
    return vis_pos


def get_active_point(modal):
    # Find active point to find a vert path from if no active then do nothing
    act_point = None
    loop_act_status = modal._container.act_status[modal._container.vert_link_ls]
    loop_act_status[modal._container.vert_link_ls < 0] = False
    loop_act_status = loop_act_status.any(axis=1)
    act_points = loop_act_status.nonzero()[0]

    if act_points.size > 0:
        act_point = act_points[0]

    return act_point


def get_active_face(modal):
    # Find active face by testing every face for a face with all loops being active
    act_face = None
    face_act_status = modal._container.act_status[modal._container.face_link_ls]
    face_act_status[modal._container.face_link_ls < 0] = True
    face_act_status = face_act_status.all(axis=1)
    act_faces = face_act_status.nonzero()[0]

    if act_faces.size > 0:
        act_face = act_faces[0]

    return act_face


def get_face_ls(modal, face_inds):
    if modal._individual_loops:
        sel_loops = modal._container.face_link_ls[face_inds]
    else:
        face_vs = modal._container.face_link_vs[face_inds]
        face_vs = face_vs[face_vs >= 0]

        sel_loops = modal._container.vert_link_ls[face_vs]

    sel_loops = sel_loops[sel_loops >= 0]
    loops = filter_hidden_loops(modal, sel_loops)
    return loops


def get_vert_ls(modal, vert_inds):
    sel_loops = modal._container.vert_link_ls[vert_inds]
    sel_loops = sel_loops[sel_loops >= 0]

    loops = filter_hidden_loops(modal, sel_loops)
    return loops


def get_edge_ls(modal, edge_inds):
    sel_loops = modal._container.vert_link_ls[modal._container.edge_link_vs[edge_inds]]
    sel_loops = sel_loops[sel_loops >= 0]

    loops = filter_hidden_loops(modal, sel_loops)
    return loops


def filter_hidden_verts(modal, mask):
    v_ls = modal._container.vert_link_ls

    hidden = ~modal._container.hide_status[v_ls[mask]].all(axis=1)
    mask = mask[hidden]
    return mask


def filter_hidden_faces(modal, mask):
    if modal._individual_loops:
        sel_loops = modal._container.face_link_ls[mask]
        hidden = ~modal._container.hide_status[sel_loops]
        hidden[sel_loops < 0] = False
        hidden = hidden.any(axis=1)
    else:
        face_vs = modal._container.face_link_vs[mask]
        sel_loops = modal._container.vert_link_ls[face_vs]
        hidden = ~modal._container.hide_status[sel_loops]
        hidden[face_vs < 0] = False
        hidden[sel_loops < 0] = False
        hidden = hidden.any(axis=1).any(axis=1)

    mask = mask[hidden]
    return mask


def filter_hidden_loops(modal, mask):
    hidden = ~modal._container.hide_status[mask]
    mask = mask[hidden]
    return mask


def set_click_selection(modal, shift, mask):
    # New selection so clear act and sel and set current as sel/act
    if shift == False:
        modal._container.sel_status[:] = False
        modal._container.act_status[:] = False

        modal._container.sel_status[mask] = True
        modal._container.act_status[mask] = True

    # Adding to selection
    else:
        l_sel = modal._container.sel_status[mask]
        l_act = modal._container.act_status[mask]

        # Check if any point loops are not sel/act if so make all sel/act
        if l_sel.all() == False or l_act.all() == False:
            modal._container.sel_status[mask] = True
            modal._container.act_status[:] = False
            modal._container.act_status[mask] = True

        # If all loops neither act/sel then all loops are sel/act so clear both
        else:
            modal._container.sel_status[mask] = False
            modal._container.act_status[:] = False

    return


def set_multi_selection(modal, shift, mask, act_mask):
    modal._container.act_status[:] = False
    if shift == False:
        modal._container.sel_status[:] = False

    if modal._container.sel_status[mask].all():
        modal._container.sel_status[mask] = False
    else:
        modal._container.sel_status[mask] = True
        modal._container.act_status[act_mask] = True
    return


def set_group_selection(modal, shift, ctrl, mask):
    if shift == False and ctrl == False:
        modal._container.sel_status[:] = False

    if ctrl:
        modal._container.sel_status[mask] = False
    else:
        modal._container.sel_status[mask] = True

    # Check if active point/face lost is no longer selected
    act_face = get_active_face(modal)
    if act_face is not None:
        act_ls = get_face_ls(modal, [act_face])
        if modal._container.sel_status[act_ls].all() == False:
            modal._container.act_status[:] = False

    act_point = get_active_point(modal)
    if act_point is not None:
        act_ls = get_vert_ls(modal, [act_point])

        if modal._individual_loops:
            if modal._container.sel_status[act_ls].all() == False:
                modal._container.act_status[:] = False
        else:
            modal._container.act_status[act_ls] = modal._container.sel_status[act_ls]

    return


def filter_selection_points(modal, shift, ctrl):
    if shift:
        avail_pos = get_selectable_points(modal)
    elif ctrl:
        avail_pos = get_selected_points(modal, any_selected=True)
    else:
        avail_pos = get_visible_points(modal)

    return avail_pos


def filter_selection_loops(modal, shift, ctrl):
    if shift:
        avail_ls = get_selectable_loops(modal)
    elif ctrl:
        avail_ls = modal._container.sel_status.nonzero()[0]
    else:
        avail_ls = (~modal._container.hide_status).nonzero()[0]

    return avail_ls


#


def selection_test(modal, shift):
    # Get coords in region space
    rcos = get_np_region_cos(modal._container.po_coords,
                             modal.act_reg, modal.act_rv3d)

    # Get ordered list of closest points within the threshold
    d_order = get_np_vec_ordered_dists(
        rcos, modal._mouse_reg_loc, threshold=15.0)

    d_order = filter_hidden_verts(modal, d_order)

    change = False
    # Point selection
    if d_order.size > 0:
        # Test for first point that is non occluded if xray off
        if modal._x_ray_mode == False:
            po_ind = None
            for ind in d_order:
                valid_po = not ray_cast_view_occlude_test(
                    Vector(modal._container.po_coords[ind]), modal._mouse_reg_loc, modal._object_bvh, modal.act_reg, modal.act_rv3d)
                if valid_po:
                    po_ind = ind
                    break
        else:
            po_ind = d_order[0]

        if po_ind is not None:
            mask = get_vert_ls(modal, po_ind)
            set_click_selection(modal, shift, mask)
            change = True

    # Face/Loop Tri selection
    if change == False:
        face_res = ray_cast_to_mouse(modal)

        if face_res is not None:
            hidden = filter_hidden_faces(modal, np.array([face_res[1]]))
            if hidden.size > 0:
                # If using individual loops first test for loop a tri selection
                if modal._individual_loops:
                    mask = get_face_ls(modal, face_res[1])

                    tri_cos = modal._container.loop_tri_coords[mask]
                    tri_cos.shape = [tri_cos.shape[0] *
                                     tri_cos.shape[1], tri_cos.shape[2]]

                    rcos = get_np_region_cos(
                        tri_cos, modal.act_reg, modal.act_rv3d)
                    rcos.shape = [mask.size, 3, 3]

                    for c, co_set in enumerate(rcos):
                        intersect = intersect_point_tri_2d(
                            modal._mouse_reg_loc, Vector(co_set[0]), Vector(co_set[1]), Vector(co_set[2]))
                        if intersect:
                            set_click_selection(modal, shift, mask[c])
                            change = True
                            break

                # If still haven't gotten a selection then select the face loops/verts
                if change == False:
                    mask = get_face_ls(modal, face_res[1])
                    set_click_selection(modal, shift, mask)
                    change = True

    return change


def loop_selection_test(modal, shift):
    change = False
    face_res = ray_cast_to_mouse(modal)
    if face_res is not None:
        # Get edges of clicked face
        face_edges = modal._container.face_link_eds[face_res[1]]
        face_edges = face_edges[face_edges >= 0]

        # Get coords of the 2 verts of the faces edges
        po_inds = modal._container.edge_link_vs[face_edges]
        edge_cos = modal._container.po_coords[po_inds]

        # Get region coords of face edges
        shape = [edge_cos.shape[0], edge_cos.shape[1]]
        edge_cos.shape = [shape[0]*shape[1], 3]
        rcos = get_np_region_cos(edge_cos, modal.act_reg, modal.act_rv3d)

        rcos.shape = [shape[0], shape[1], 3]

        face_co = get_np_region_cos(
            [face_res[0]], modal.act_reg, modal.act_rv3d)

        # Get nearest edge to the click location
        ed_dists = get_np_dist_to_edge(rcos, face_co)
        ed_order = np.argsort(ed_dists)
        ed_dists = ed_dists[ed_order]
        if ed_order.size > 0:
            # Edge loop selection
            if ed_dists[0] < 10.0:
                # Get indices of points that all connected loops are hidden
                skip_vs = list(get_hidden_points(modal))

                face_edge = modal._container.face_link_eds[face_res[1]
                                                           ][ed_order[0]]
                sel_eds = get_edge_loop(
                    modal._object_bm, modal._object_bm.edges[face_edge], skip_verts=skip_vs)

                edge_vs = modal._container.edge_link_vs[face_edge]
                v_order = get_np_vec_ordered_dists(
                    modal._container.po_coords[edge_vs], face_res[0])
                near_vert = edge_vs[v_order[0]]

                sel_loops = get_edge_ls(modal, sel_eds)
                act_ls = get_vert_ls(modal, near_vert)

                set_multi_selection(modal, shift, sel_loops, act_ls)
                change = True

            # Face loop selection
            else:
                skip_fs = list(get_hidden_faces(modal))

                face_edge = modal._container.face_link_eds[face_res[1]
                                                           ][ed_order[0]]
                sel_loop = get_face_loop(
                    modal._object_bm, modal._object_bm.edges[face_edge], skip_fs=list(skip_fs))

                sel_loops = get_face_ls(modal, sel_loop)
                act_ls = get_face_ls(modal, face_res[1])

                set_multi_selection(modal, shift, sel_loops, act_ls)
                change = True

    return change


def path_selection_test(modal, shift):
    change = False
    face_res = ray_cast_to_mouse(modal)
    if face_res is not None:
        act_face = get_active_face(modal)

        # Test for face path
        if act_face is not None:
            modal._container.act_status[:] = False
            if shift == False:
                modal._container.sel_status[:] = False

            skip_faces = list(get_hidden_faces(modal))

            path_f = find_path_between_faces(
                [act_face, face_res[1]], modal._object_bm, skip_fs=skip_faces)

            sel_loops = get_face_ls(modal, path_f)
            act_ls = get_face_ls(modal, face_res[1])

            set_multi_selection(modal, shift, sel_loops, act_ls)
            change = True

        # Test for vert path
        else:
            act_point = get_active_point(modal)

            if act_point is not None:
                near_ind = modal._object_kd.find(face_res[0])

                if near_ind[1] != act_point:
                    skip_verts = list(get_hidden_points(modal))

                    path_v, path_ed = find_path_between_verts(
                        [act_point, near_ind[1]], modal._object_bm, skip_verts=skip_verts)

                    sel_loops = get_vert_ls(modal, path_v)
                    act_ls = get_vert_ls(modal, near_ind[1])

                    set_multi_selection(modal, shift, sel_loops, act_ls)
                    change = True

    return change


def group_vert_selection_test(modal, vert_ind_array, shift, ctrl):
    change = False
    # Vertex selection
    if vert_ind_array.size > 0:
        # Test for occlusion
        if modal._x_ray_mode == False:
            valid_pos = []

            for ind in vert_ind_array:
                valid_po = not ray_cast_view_occlude_test(
                    Vector(modal._container.po_coords[ind]), modal._mouse_reg_loc, modal._object_bvh, modal.act_reg, modal.act_rv3d)
                if valid_po:
                    valid_pos.append(ind)

            valid_pos = np.array(valid_pos)
        else:
            valid_pos = vert_ind_array

        if valid_pos.size > 0:
            sel_loops = get_vert_ls(modal, valid_pos)
            if sel_loops.size > 0:
                set_group_selection(modal, shift, ctrl, sel_loops)
                change = True
    return change


def group_loop_selection_test(modal, loop_ind_array, tri_cos, shift, ctrl):
    change = False

    if loop_ind_array.size > 0:
        # Test for occlusion
        if modal._x_ray_mode == False:
            sel_loops = []

            for ind in loop_ind_array:
                valid_po = not ray_cast_view_occlude_test(
                    Vector(tri_cos[ind]), modal._mouse_reg_loc, modal._object_bvh, modal.act_reg, modal.act_rv3d)
                if valid_po:
                    sel_loops.append(ind)

            sel_loops = np.array(sel_loops)
        else:
            sel_loops = loop_ind_array

        if sel_loops.size > 0:
            set_group_selection(modal, shift, ctrl, sel_loops)
            change = True
    return change


def box_selection_test(modal, shift, ctrl):
    avail_pos = filter_selection_points(modal, shift, ctrl)

    # Get coords in region space
    rcos = get_np_region_cos(modal._container.po_coords[avail_pos],
                             modal.act_reg, modal.act_rv3d)

    x_cos = np.array([modal._mouse_reg_loc[0], modal._mode_cache[0][0][0]])
    y_cos = np.array([modal._mouse_reg_loc[1], modal._mode_cache[0][0][1]])
    in_range = np_box_selection_test(rcos, x_cos, y_cos)

    change = group_vert_selection_test(modal, avail_pos[in_range], shift, ctrl)

    # Test loop tris for selection
    if modal._individual_loops:
        avail_ls = filter_selection_loops(modal, shift, ctrl)

        loop_tri_cos = modal._container.loop_tri_coords.mean(axis=1)

        # Get coords in region space
        rcos = get_np_region_cos(
            loop_tri_cos[avail_ls], modal.act_reg, modal.act_rv3d)

        in_range = np_box_selection_test(rcos, x_cos, y_cos)

        l_change = group_loop_selection_test(
            modal, avail_ls[in_range], loop_tri_cos, shift, ctrl)
        if l_change:
            change = l_change

    return change


def circle_selection_test(modal, shift, ctrl, radius):
    avail_pos = filter_selection_points(modal, shift, ctrl)

    # Get coords in region space
    rcos = get_np_region_cos(modal._container.po_coords[avail_pos],
                             modal.act_reg, modal.act_rv3d)

    # Get ordered list of closest points within the threshold
    in_range = get_np_vec_ordered_dists(
        rcos, modal._mouse_reg_loc, threshold=radius)

    change = group_vert_selection_test(modal, avail_pos[in_range], True, ctrl)

    # Test loop tris for selection
    if modal._individual_loops:
        avail_ls = filter_selection_loops(modal, shift, ctrl)

        loop_tri_cos = modal._container.loop_tri_coords.mean(axis=1)

        # Get coords in region space
        rcos = get_np_region_cos(
            loop_tri_cos[avail_ls], modal.act_reg, modal.act_rv3d)

        in_range = get_np_vec_ordered_dists(
            rcos, modal._mouse_reg_loc, threshold=radius)

        l_change = group_loop_selection_test(
            modal, avail_ls[in_range], loop_tri_cos, True, ctrl)
        if l_change:
            change = l_change

    return change


def lasso_selection_test(modal, shift, ctrl):
    avail_pos = filter_selection_points(modal, shift, ctrl)

    # Get coords in region space
    rcos = get_np_region_cos(modal._container.po_coords[avail_pos],
                             modal.act_reg, modal.act_rv3d)

    # First filter by box selection to reduce size of points to test with lasso
    lasso_shape = np.array(modal._mode_cache[0])

    x_cos = np.array([lasso_shape[:, 0].min(), lasso_shape[:, 0].max()])
    y_cos = np.array([lasso_shape[:, 1].min(), lasso_shape[:, 1].max()])
    box_in_range = np_box_selection_test(rcos, x_cos, y_cos)

    # Test which of the valid points are inside the lasso shape
    in_range = box_in_range[np_test_cos_in_shape(
        rcos[box_in_range], lasso_shape)]

    change = group_vert_selection_test(modal, avail_pos[in_range], shift, ctrl)

    # Test loop tris for selection
    if modal._individual_loops:
        avail_ls = filter_selection_loops(modal, shift, ctrl)

        loop_tri_cos = modal._container.loop_tri_coords[avail_ls].mean(axis=1)
        # Get coords in region space
        rcos = get_np_region_cos(
            loop_tri_cos, modal.act_reg, modal.act_rv3d)

        box_in_range = np_box_selection_test(rcos, x_cos, y_cos)

        in_range = box_in_range[np_test_cos_in_shape(
            rcos[box_in_range], lasso_shape)]

        l_change = group_loop_selection_test(
            modal, avail_ls[in_range], loop_tri_cos, shift, ctrl)
        if l_change:
            change = l_change

    return change


##
##
##
##
##


#
# ORBIT EMPTY
#
def add_orbit_empty(ob):
    for i in range(len(bpy.context.selected_objects)):
        bpy.context.selected_objects[0].select_set(False)

    emp = bpy.data.objects.new('ABN_Orbit Empty', None)
    emp.empty_display_size = 0.0
    emp.matrix_world = ob.matrix_world.copy()
    bpy.context.collection.objects.link(emp)
    bpy.context.view_layer.objects.active = emp
    return emp


def update_orbit_empty(modal):
    # Reset selection to only orbit object
    for i in range(len(bpy.context.selected_objects)):
        bpy.context.selected_objects[0].select_set(False)

    modal._orbit_ob.select_set(True)
    bpy.context.view_layer.objects.active = modal._orbit_ob

    if modal._container.sel_status.any():
        avg_loc = np.mean(
            modal._container.loop_coords[modal._container.sel_status], axis=0)
        gizmo_update_hide(modal, True)
        modal._orbit_ob.matrix_world.translation = avg_loc
    else:
        gizmo_update_hide(modal, False)
        modal._orbit_ob.matrix_world.translation = modal._object.location

    modal._orbit_ob.select_set(True)
    bpy.context.view_layer.objects.active = modal._orbit_ob

    if modal._use_gizmo:
        modal._window.update_gizmo_pos(modal._orbit_ob.matrix_world)
        relocate_gizmo_panel(modal)

    return


def delete_orbit_empty(modal):
    if modal._orbit_ob is not None:
        try:
            bpy.data.objects.remove(modal._orbit_ob)
        except:
            modal._orbit_ob = None

    return


#
# KEYMAP TEST/LOAD
#
def load_keymap(modal):
    # modal.keymap = {}

    # for item in addon_keymaps[0][0].keymap_items:
    #     modal.keymap[item.name] = item

    modal.keymap = addon_keymaps[0]
    return


def keys_find(keymap_items, event):
    scroll_up = ['WHEELINMOUSE', 'WHEELUPMOUSE']
    scroll_down = ['WHEELOUTMOUSE', 'WHEELDOWNMOUSE']

    key_val = []
    for key in keymap_items:
        if key.type == event.type or (key.type in scroll_up and event.type in scroll_up) or (key.type in scroll_down and event.type in scroll_down):
            if (key.alt == event.alt and key.ctrl == event.ctrl and key.shift == event.shift) or key.any:
                if key.value == event.value:
                    key_val.append(key.name)

    # if len(key_val) == 0:
    #     key_val = None
    return key_val


def test_navigation_key(nav_list, event):
    nav_status = False

    scroll_up = ['WHEELINMOUSE', 'WHEELUPMOUSE']
    scroll_down = ['WHEELOUTMOUSE', 'WHEELDOWNMOUSE']

    nav_inds = [i for i in range(
        len(nav_list)) if (nav_list[i][0] == event.type and nav_list[i][1] == event.value) or (nav_list[i][0] in scroll_up and event.type in scroll_up) or (nav_list[i][0] in scroll_down and event.type in scroll_down)]
    if len(nav_inds) > 0:
        for ind in nav_inds:
            nav_key = nav_list[ind]
            if nav_key[2] or (event.ctrl == nav_key[3] and event.shift == nav_key[4] and event.alt == nav_key[5]):
                nav_status = True

    return nav_status
