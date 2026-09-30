import bpy
import platform
from bpy.props import *
from bpy.types import Operator
# from .functions_general import *
# from .functions_drawing import *
# from .functions_modal import *
from .functions_modal_buttons import *
from .functions_modal_keymap import *
from .functions_tools import *
# from .classes import *

# 新增类来保存模式状态
class ModeState:
    def __init__(self):
        self.object_mode = None
        self.edit_mode = None
        self.active_object = None
    
    def save(self, context):
        """保存当前模式状态"""
        self.active_object = context.active_object
        self.object_mode = context.mode
        self.edit_mode = context.object.mode if context.object else None
        print(self.edit_mode)
    
    def restore(self, context):
        """恢复保存的模式状态"""
        try:
            # 确保活动对象仍然存在
            if self.active_object and self.active_object.name in bpy.data.objects:
                # 设置活动对象
                context.view_layer.objects.active = self.active_object
                
                # 恢复模式
                if self.object_mode == 'EDIT_MESH' and self.edit_mode == 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
                elif self.object_mode == 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                elif self.object_mode == 'POSE':
                    bpy.ops.object.mode_set(mode='POSE')
                # 其他模式也可以类似处理...
            else:
                # 如果活动对象不存在，默认设置为物体模式
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as e:
            print(f"恢复模式失败: {e}")
            # 出错时默认设置为物体模式
            bpy.ops.object.mode_set(mode='OBJECT')

class ABN_OT_normal_editor_modal(Operator):
    bl_idname = "abnormal.normal_editor_modal"
    bl_label = "开始法线编辑"
    
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def modal(self, context, event):
        self._modal_running = False
        status = {"PASS_THROUGH"}

        if bpy.context.area == None:
            finish_modal(self, True)
            self.report({'WARNING'}, "Something went wrong. Cancelling modal")
            return {"CANCELLED"}

        if bpy.context.area.type != 'VIEW_3D':
            finish_modal(self, True)
            self.report({'WARNING'}, "Left 3D View. Cancelling modal")
            return {"CANCELLED"}

        # If event.type is blank avoid testing it agaisnt keys as it prints a lot of errors
        if event.type == '':
            self._modal_running = True
            return status

        # 性能优化：只在视图矩阵或FOV改变时才更新顶点属性
        # 获取到视图相机的位置和旋转角度
        view_matrix = None
        fov = None
        space = None
        
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # 获取视图矩阵（无论透视或正交）
                        view_matrix = space.region_3d.view_matrix.copy()
                        # 根据视图模式计算 FOV 或使用正交参数
                        if space.region_3d.view_perspective == 'PERSP':
                            # 透视模式：计算水平 FOV
                            lens = space.lens
                            sensor_width = 36.0
                            fov = 2 * math.atan(sensor_width / (2 * lens))
                        else:
                            # 正交模式
                            lens = space.lens
                            fov = lens / 10.0
                        # 找到后立即退出循环
                        break
                if view_matrix is not None:
                    break

        if fov is None or view_matrix is None:
            # 如果找不到视图，不报错，继续执行
            pass
        else:
            # 检查视图矩阵是否改变（只在改变时更新）
            view_changed = False
            if not hasattr(self, '_last_view_matrix') or self._last_view_matrix != view_matrix:
                view_changed = True
                self._last_view_matrix = view_matrix
            
            if not hasattr(self, '_last_fov') or abs(self._last_fov - fov) > 0.0001:
                view_changed = True
                self._last_fov = fov
            
            # 只在视图改变时才更新顶点属性（性能优化）
            if view_changed:
                mesh = self._object.data
                fov_attr = mesh.attributes.get('fov') or mesh.attributes.new(name='fov', type='FLOAT', domain='POINT')
                pos_vs_z_attr = mesh.attributes.get('pos_vs_z') or mesh.attributes.new(name='pos_vs_z', type='FLOAT', domain='POINT')
                
                # 性能优化：使用批量操作替代循环
                import numpy as np
                vert_count = len(mesh.vertices)
                
                # 批量设置fov值
                fov_values = np.full(vert_count, fov, dtype=np.float32)
                fov_attr.data.foreach_set('value', fov_values)
                
                # 批量计算和设置pos_vs_z
                if vert_count > 0:
                    # 批量获取顶点坐标
                    vert_coords = np.zeros(vert_count * 3, dtype=np.float32)
                    mesh.vertices.foreach_get('co', vert_coords)
                    vert_coords.shape = (vert_count, 3)
                    
                    # 批量转换到世界空间
                    matrix_world = np.array(self._object.matrix_world)
                    world_coords = np.ones((vert_count, 4), dtype=np.float32)
                    world_coords[:, :3] = vert_coords
                    world_coords = (matrix_world @ world_coords.T).T
                    
                    # 批量转换到视图空间
                    view_matrix_np = np.array(view_matrix)
                    view_coords = (view_matrix_np @ world_coords.T).T
                    
                    # 批量设置z值
                    pos_vs_z_values = view_coords[:, 2].astype(np.float32)
                    pos_vs_z_attr.data.foreach_set('value', pos_vs_z_values)

        self._mouse_abs_loc[:] = [event.mouse_x, event.mouse_y, 0.0]
        self._mouse_reg_loc[:] = [
            event.mouse_region_x, event.mouse_region_y, 0.0]

        self.act_reg, self.act_rv3d = context.region, context.region_data
        # self.act_reg, self.act_rv3d = check_area(self)
        # self._mouse_act_loc = [self._mouse_abs_loc[0]-self.act_reg.x, self._mouse_abs_loc[1]-self.act_reg.y]

        self._window.check_dimensions(context)
        # Check that mousemove is larger than a pixel to be tested
        mouse_move_check = True
        if event.type == 'MOUSEMOVE' and get_np_vec_lengths((self._mouse_reg_loc-self._prev_mouse_loc).reshape(-1, 3)) < 1.0:
            mouse_move_check = False

        if mouse_move_check:
            status = self._current_tool.test_mode(
                self, context, event, self.keymap, None)

            self._prev_mouse_loc[:] = self._mouse_reg_loc

        # 确认修改法线
        if self._confirm_modal:
            finish_modal(self, False)
            status = {"FINISHED"}
        # 取消修改法线
        elif self._cancel_modal:
            finish_modal(self, True)
            status = {"CANCELLED"}

        refresh_batches(self, context)

        self._modal_running = True
        return status

    def invoke(self, context, event):

        # 获取首选项对象
        prefs = context.preferences

        # 保存当前模式状态
        self.mode_state = ModeState()
        self.mode_state.save(bpy.context)

        # 设置语言为英文并关闭翻译，如果不是英文面板则会出现无法转动视角的问题
        prefs.view.language = 'en_US'               # 英文语言代码
        prefs.view.use_translate_interface = True   # 关闭界面翻译
        prefs.view.use_translate_tooltips = True    # 关闭工具提示翻译

        self.act_reg, self.act_rv3d = context.region, context.region_data
        # self.act_reg, self.act_rv3d = check_area(self)
        rh = self.act_reg.height
        rw = self.act_reg.width

        if context.active_object == None:
            self.report({'WARNING'}, "No valid active object selected")
            return {'CANCELLED'}

        if context.active_object.type != 'MESH':
            self.report({'WARNING'}, "Active object is not a mesh")
            return {'CANCELLED'}
    
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

        if context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # INITIALIZE PROPERTIES
        self._addon_prefs = bpy.context.preferences.addons[__package__.split('.')[
            0]].preferences
        self._display_prefs = self._addon_prefs.display
        self._behavior_prefs = self._addon_prefs.behavior
        self._keymap_sel_prefs = self._addon_prefs.keymap_sel
        self._keymap_shortcut_prefs = self._addon_prefs.keymap_shortcut
        self._keymap_tool_prefs = self._addon_prefs.keymap_tool
        self._mouse_abs_loc = np.array([event.mouse_x, event.mouse_y, 0.0])
        self._mouse_reg_loc = np.array(
            [event.mouse_region_x, event.mouse_region_y, 0.0])
        self._prev_mouse_loc = np.array(
            [event.mouse_region_x, event.mouse_region_y, 0.0])

        self._mouse_init = np.array([0.0, 0.0, 0.0])
        self._active_point = None
        self._active_face = None

        self._mode_cache = []
        self._line_drawing_pos = []

        self._copy_normals = np.array([])

        self.target_strength = 1.0
        self._target_emp = None
        self.point_align = False

        self.translate_mode = 0
        self.translate_axis = 2
        self.translate_draw_line = []

        self._rot_increment_one = True
        self._rot_increment_five = False
        self._rot_increment_ten = False
        self._rot_increment = 1

        self._object_smooth = True

        self._smooth_iterations = 5
        self._smooth_strength = 0.25

        self._mirror_range = 0.1

        self._mirror_x = False
        self._mirror_y = False
        self._mirror_z = False

        self._confirm_modal = False
        self._cancel_modal = False

        self._current_filter = ''

        self._draw_area = context.area
        self._modal_running = True
        self.redraw = False
        self.redraw_active = False
        self.circle_radius = 50

        self._addon_prefs.object = context.active_object.name

        # VIEWPORT DISPLAY SETTINGS

        self._x_ray_mode = False
        self._use_gizmo = self._behavior_prefs.rotate_gizmo_use
        self._gizmo_size = self._display_prefs.gizmo_size

        self._totalOutline_width = self._display_prefs.totalOutline_width
        self._outline_width_multiplier = self._display_prefs.outline_width_multiplier
        self._outline_width_multiplier_locked = self._display_prefs.outline_width_multiplier_locked
        self._outline_scale = self._display_prefs.outline_scale
        self._z_offset = self._display_prefs.z_offset

        self._normal_size = self._display_prefs.normal_size
        self._line_brightness = self._display_prefs.line_brightness
        self._point_size = self._display_prefs.point_size
        self._loop_tri_size = self._display_prefs.loop_tri_size
        self._selected_only = self._display_prefs.selected_only
        self._draw_weights = self._display_prefs.draw_weights
        self._selected_scale = self._display_prefs.selected_scale
        self._individual_loops = self._behavior_prefs.individual_loops
        if self._display_prefs.ui_scale == 0.0:
            self._ui_scale = context.window.width/1920
        else:
            self._ui_scale = self._display_prefs.ui_scale
        self.prev_view = context.region_data.view_matrix.copy()

        # 测试添加自定义参数
        self._outline_scale = self._display_prefs.outline_scale
        self._z_offset = self._display_prefs.z_offset
        
        # CACHE VIEWPORT SETTINGS
        viewport_change_cache(self, context)

        # MODES
        self.rotating = False
        self.gizmo_click = False

        self.box_selecting = False
        self.lasso_selecting = False
        self.circle_selecting = False
        self.circle_resizing = False
        self.circle_removing = False

        self._popup_panel = None

        self._hover_timer = None
        self._hover_stop_time = 0.0
        self._hover_delay_passed = False
        self._hover_delay = 0.7

        self.click_hold = False
        self.ui_hover = False
        self.selection_drawing = True
        self.bezier_changing = False

        # UNDO STACK STORAGE
        self._history_stack = []
        self._history_select_stack = []
        self._history_normal_stack = []
        self._history_filter_stack = []
        self._history_position = 0
        self._history_select_position = 0
        self._history_normal_position = 0
        self._history_filter_position = 0
        self._history_steps = 128

        # INITIALIZE OBJECTS
        self._objects_mod_status = []
        self._objects_sk_vis = []
        # 只三角化 n-gon 面，保证描边数据与 Unity 等引擎对齐
        if context.active_object.type == 'MESH':
            triangulate_ngons(context.active_object.data)
        ob_bm, ob_kd, ob_bvh = ob_data_structures(
            self, context.active_object)

        # INITIALIZE OBJECT DATA LISTS
        self._object = context.active_object
        self._object_name = context.active_object.name
        self._object_pointer = context.active_object.as_pointer()

        self._object_bm = ob_bm
        self._object_bvh = ob_bvh
        self._object_kd = ob_kd

        if self._object.data.use_auto_smooth == False:
            self._object.data.use_auto_smooth = True
            self._object.data.auto_smooth_angle = 180

        # INITIALIZE BATCHES AND SHADERS
        self.shader_2d = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        self.shader_3d = gpu.shader.from_builtin('3D_UNIFORM_COLOR')

        use_alt_shader = self._behavior_prefs.alt_drawing
        if platform.system() == 'Darwin':
            use_alt_shader = True

        self._container = ABNContainer(
            self._object.matrix_world.normalized(), alt_shader=use_alt_shader)

        self._container.set_scale_selection(self._selected_scale)
        self._container.set_brightess(self._line_brightness)

        self._container.set_totalOutline_width(self._totalOutline_width)
        self._container.set_outine_scale(self._outline_scale)
        self._container.set_z_offset(self._z_offset)

        self._container.set_normal_scale(self._normal_size)
        self._container.set_point_size(self._point_size)
        self._container.set_loop_scale(self._loop_tri_size)
        self._container.set_draw_only_selected(self._selected_only)
        self._container.set_draw_weights(self._draw_weights)
        self._container.set_draw_tris(self._individual_loops)

        # INITIALIZE POINT DATA
        cache_point_data(self)
        self._orbit_ob = add_orbit_empty(self._object)
        self._target_emp = add_target_empty(self._object)

        update_filter_from_vg(self)

        # INITIALIZE UI WINDOW
        load_keymap(self)

        # NAVIGATION KEYS LIST
        init_nav_list(self)

        init_ui_panels(self, rw, rh, self._ui_scale)

        update_orbit_empty(self)

        setup_tools(self)

        # SETUP BATCHES
        self._container.clear_batches()

        # 更新数据
        refresh_batches(self, context)

        # OPENGL DRAWING HANDLER
        args = (self, context)
        self._draw_handle_2d = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_2d, args, "WINDOW", "POST_PIXEL")
        
        # 绘制法线，顶点等
        self._draw_handle_3d = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_3d, args, "WINDOW", "POST_VIEW")

        dns = bpy.app.driver_namespace
        dns["dh2d"] = self._draw_handle_2d
        dns["dh3d"] = self._draw_handle_3d

        add_to_undostack(self, 3)

        self._window.check_in_window()

        # SET MODAL
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


def register():
    bpy.utils.register_class(ABN_OT_normal_editor_modal)
    return


def unregister():
    bpy.utils.unregister_class(ABN_OT_normal_editor_modal)
    return
