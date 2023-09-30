import webbrowser
from spk2extract import DirectoryManager


import dearpygui.dearpygui as dpg


def _help(message):
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)


def _hyperlink(text, address):
    b = dpg.add_button(label=text, callback=lambda: webbrowser.open(address))
    dpg.bind_item_theme(b, "_spk_hyperlinkTheme")


def _config(sender, keyword, user_data):
    widget_type = dpg.get_item_type(sender)
    items = user_data

    if widget_type == "mvAppItemType::mvRadioButton":
        value = True

    else:
        keyword = dpg.get_item_label(sender)
        value = dpg.get_value(sender)

    if isinstance(user_data, list):
        for item in items:
            dpg.configure_item(item, **{keyword: value})
    else:
        dpg.configure_item(items, **{keyword: value})


def _add_config_options(item, columns, *names, **kwargs):
    if columns == 1:
        if "before" in kwargs:
            for name in names:
                dpg.add_checkbox(
                    label=name,
                    callback=_config,
                    user_data=item,
                    before=kwargs["before"],
                    default_value=dpg.get_item_configuration(item)[name],
                )
        else:
            for name in names:
                dpg.add_checkbox(
                    label=name,
                    callback=_config,
                    user_data=item,
                    default_value=dpg.get_item_configuration(item)[name],
                )

    else:
        if "before" in kwargs:
            dpg.push_container_stack(
                dpg.add_table(header_row=False, before=kwargs["before"])
            )
        else:
            dpg.push_container_stack(dpg.add_table(header_row=False))

        for i in range(columns):
            dpg.add_table_column()

        for i in range(int(len(names) / columns)):
            with dpg.table_row():
                for j in range(columns):
                    dpg.add_checkbox(
                        label=names[i * columns + j],
                        callback=_config,
                        user_data=item,
                        default_value=dpg.get_item_configuration(item)[
                            names[i * columns + j]
                        ],
                    )
        dpg.pop_container_stack()


def _add_config_option(item, default_value, *names):
    dpg.add_radio_button(
        names, default_value=default_value, callback=_config, user_data=item
    )


def _hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)  # XXX assume int() truncates!
    j = (h * 6.0) - i
    p, q, t = v * (1.0 - s), v * (1.0 - s * j), v * (1.0 - s * (1.0 - j))
    i %= 6
    if i == 0:
        return 255 * v, 255 * t, 255 * p
    if i == 1:
        return 255 * q, 255 * v, 255 * p
    if i == 2:
        return 255 * p, 255 * v, 255 * t
    if i == 3:
        return 255 * p, 255 * q, 255 * v
    if i == 4:
        return 255 * t, 255 * p, 255 * v
    if i == 5:
        return 255 * v, 255 * p, 255 * q


def _create_static_textures():
    ## create static textures
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    texture_data3 = []
    for row in range(50):
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
    for row in range(50):
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)

    dpg.add_static_texture(
        100,
        100,
        texture_data1,
        parent="spk_texture_container",
        tag="__spk_static_texture_1",
        label="Static Texture 1",
    )
    dpg.add_static_texture(
        50,
        50,
        texture_data2,
        parent="spk_texture_container",
        tag="__spk_static_texture_2",
        label="Static Texture 2",
    )
    dpg.add_static_texture(
        100,
        100,
        texture_data3,
        parent="spk_texture_container",
        tag="__spk_static_texture_3",
        label="Static Texture 3",
    )


def _create_dynamic_textures():
    ## create dynamic textures
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    dpg.add_dynamic_texture(
        100,
        100,
        texture_data1,
        parent="spk_texture_container",
        tag="spk_dynamic_texture_1",
    )
    dpg.add_dynamic_texture(
        50,
        50,
        texture_data2,
        parent="spk_texture_container",
        tag="spk_dynamic_texture_2",
    )


def _update_dynamic_textures(_sender, app_data, user_data):
    new_color = app_data
    new_color[0] = new_color[0]
    new_color[1] = new_color[1]
    new_color[2] = new_color[2]
    new_color[3] = new_color[3]

    if user_data == 1:
        texture_data = []
        for i in range(100 * 100):
            texture_data.append(new_color[0])
            texture_data.append(new_color[1])
            texture_data.append(new_color[2])
            texture_data.append(new_color[3])
        dpg.set_value("__spk_dynamic_texture_1", texture_data)

    elif user_data == 2:
        texture_data = []
        for i in range(50 * 50):
            texture_data.append(new_color[0])
            texture_data.append(new_color[1])
            texture_data.append(new_color[2])
            texture_data.append(new_color[3])
        dpg.set_value("spk_dynamic_texture_2", texture_data)

def _on_spk_close(sender, app_data, user_data):
    dpg.delete_item(sender)
    dpg.delete_item("spk_texture_container")
    dpg.delete_item("spk_colormap_registry")
    dpg.delete_item("spk_hyperlinkTheme")
    dpg.delete_item("__spk_theme_progressbar")
    dpg.delete_item("stock_theme1")
    dpg.delete_item("stock_theme2")
    dpg.delete_item("stock_theme3")
    dpg.delete_item("stock_theme4")
    dpg.delete_item("stem_theme1")
    dpg.delete_item("__spk_keyboard_handler")
    dpg.delete_item("__spk_mouse_handler")
    dpg.delete_item("__spk_filedialog")
    dpg.delete_item("__spk_stage1")
    dpg.delete_item("__spk_popup1")
    dpg.delete_item("__spk_popup2")
    dpg.delete_item("__spk_popup3")
    dpg.delete_item("__spk_item_reg3")
    dpg.delete_item("__spk_item_reg6")
    dpg.delete_item("__spk_item_reg7")
    dpg.delete_item("spkitemregistry")
    for i in range(7):
        dpg.delete_item("__spk_theme"+str(i))
        dpg.delete_item("__spk_theme2_"+str(i))
    for i in range(5):
        dpg.delete_item("__spk_item_reg1_"+str(i))
        dpg.delete_item("__spk_item_reg2_"+str(i))
    for i in range(3):
        dpg.delete_item("__spk_item_reg4_"+str(i))
    for i in range(4):
        dpg.delete_item("__spk_item_reg5_"+str(i))

def _file_dialog(sender, app_data, user_data):
    with dpg.file_dialog(
        label="Demo File Dialog",
        width=300,
        height=400,
        show=False,
        callback=lambda s, a, u: print(s, a, u),
        tag="__spk_filedialog",
    ):
        dpg.add_file_extension(".INI", color=(255, 255, 255, 255))

    dpg.add_button(
        label="Show File Selector",
        user_data=dpg.last_container(),
        callback=lambda s, a, u: dpg.configure_item(u, show=True),
    )


def show_gui():

    dpg.add_texture_registry(label="Hello", tag="spk_texture_container")
    dpg.add_colormap_registry(label="Demo Colormap Registry", tag="spk_colormap_registry")

    with dpg.theme(tag="_spk_hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])

    def _log(sender, app_data, user_data):
        print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")

    _create_static_textures()
    _create_dynamic_textures()

    with dpg.window(
        label="Spike Extractor",
        width=800,
        height=800,
        pos=(100, 100),
        on_close=_on_spk_close,
        tag="__spk_id",
    ):
        with dpg.menu_bar():
            with dpg.menu(label="Menu"):
                dpg.add_text("This menu is just for show!")
                dpg.add_menu_item(label="New")
                dpg.add_menu_item(label="Open")

                dpg.add_menu_item(label="Save")
                dpg.add_menu_item(label="Save As...")

                with dpg.menu(label="Settings"):
                    dpg.add_menu_item(label="Option 1", callback=_log)
                    dpg.add_menu_item(label="Option 2", check=True, callback=_log)
                    dpg.add_menu_item(
                        label="Option 3", check=True, default_value=True, callback=_log
                    )

                    with dpg.child_window(height=60, autosize_x=True, delay_search=True):
                        for i in range(10):
                            dpg.add_text(f"Scolling Text{i}")

                    dpg.add_slider_float(label="Slider Float")
                    dpg.add_input_int(label="Input Int")
                    dpg.add_combo(("Yes", "No", "Maybe"), label="Combo")

            with dpg.menu(label="Tools"):
                dpg.add_menu_item(
                    label="Show About", callback=lambda: dpg.show_tool(dpg.mvTool_About)
                )
                dpg.add_menu_item(
                    label="Show Metrics", callback=lambda: dpg.show_tool(dpg.mvTool_Metrics)
                )
                dpg.add_menu_item(
                    label="Show Documentation",
                    callback=lambda: dpg.show_tool(dpg.mvTool_Doc),
                )
                dpg.add_menu_item(
                    label="Show Debug", callback=lambda: dpg.show_tool(dpg.mvTool_Debug)
                )
                dpg.add_menu_item(
                    label="Show Style Editor",
                    callback=lambda: dpg.show_tool(dpg.mvTool_Style),
                )
                dpg.add_menu_item(
                    label="Show Font Manager",
                    callback=lambda: dpg.show_tool(dpg.mvTool_Font),
                )
                dpg.add_menu_item(
                    label="Show Item Registry",
                    callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry),
                )
    
            with dpg.menu(label="Settings"):
                dpg.add_menu_item(
                    label="Wait For Input",
                    check=True,
                    callback=lambda s, a: dpg.configure_app(wait_for_input=a),
                )
                dpg.add_menu_item(
                    label="Toggle Fullscreen",
                    callback=lambda: dpg.toggle_viewport_fullscreen(),
                )
    
        with dpg.group(horizontal=True):
            dpg.add_loading_indicator(circle_count=3)
            with dpg.group():
                dpg.add_text(f"Spike Sorter.")
                with dpg.group(horizontal=True):
                    dpg.add_text(
                        "Source code:"
                    )
                    _hyperlink(
                        "github:spk2extract",
                        "https://github.com/FlynnOConnell/spk2extract",
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text("Tutorial:")
                    _hyperlink(
                        "Documentation",
                        "https://spk2extract.readthedocs.io/en/latest/usage.html",
                    )

        with dpg.collapsing_header(label="Configuration"):
            with dpg.file_dialog(
                label="Demo File Dialog",
                width=300,
                height=400,
                show=False,
                callback=lambda s, a, u: print(s, a, u),
                tag="__demo_filedialog",
            ):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                dpg.add_file_extension(
                    "Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255)
                )
                dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))
                # dpg.add_button(label="Button on file dialog")

            dpg.add_button(
                label="Show File Selector",
                user_data=dpg.last_container(),
                callback=lambda s, a, u: dpg.configure_item(u, show=True),
            )

            dpg.add_slider_int(
                label=" ", default_value=1, vertical=True, max_value=5, height=160
            )
            dpg.add_slider_double(
                label=" ",
                default_value=1.0,
                vertical=True,
                max_value=5.0,
                height=160,
            )

            with dpg.group(horizontal=True):
                with dpg.group(horizontal=True):
                    values = [0.0, 0.60, 0.35, 0.9, 0.70, 0.20, 0.0]

                    for i in range(7):
                        with dpg.theme(tag="__spk_theme2_" + str(i)):
                            with dpg.theme_component(0):
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_FrameBg,
                                    _hsv_to_rgb(i / 7.0, 0.5, 0.5),
                                )
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_SliderGrab,
                                    _hsv_to_rgb(i / 7.0, 0.9, 0.9),
                                )
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_FrameBgActive,
                                    _hsv_to_rgb(i / 7.0, 0.7, 0.5),
                                )
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_FrameBgHovered,
                                    _hsv_to_rgb(i / 7.0, 0.6, 0.5),
                                )

                        dpg.add_slider_float(
                            label=" ",
                            default_value=values[i],
                            vertical=True,
                            max_value=1.0,
                            height=160,
                        )
                        dpg.bind_item_theme(
                            dpg.last_item(), "__spk_theme2_" + str(i)
                        )

                with dpg.group():
                    for i in range(3):
                        with dpg.group(horizontal=True):
                            values = [0.20, 0.80, 0.40, 0.25]
                            for j in range(4):
                                dpg.add_slider_float(
                                    label=" ",
                                    default_value=values[j],
                                    vertical=True,
                                    max_value=1.0,
                                    height=50,
                                )

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(
                        label=" ",
                        vertical=True,
                        max_value=1.0,
                        height=160,
                        width=40,
                    )
                    dpg.add_slider_float(
                        label=" ",
                        vertical=True,
                        max_value=1.0,
                        height=160,
                        width=40,
                    )
                    dpg.add_slider_float(
                        label=" ",
                        vertical=True,
                        max_value=1.0,
                        height=160,
                        width=40,
                    )
                    dpg.add_slider_float(
                        label=" ",
                        vertical=True,
                        max_value=1.0,
                        height=160,
                        width=40,
                    )
