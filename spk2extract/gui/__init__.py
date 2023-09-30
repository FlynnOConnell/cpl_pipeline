import dearpygui.dearpygui as dpg
from run_gui import show_gui
from . import dark_theme

dpg.create_context()
dpg.create_viewport(title='Spike Sorter', width=600, height=600)

dark_theme = dark_theme.get_dark_theme()
dpg.bind_theme(dark_theme)
show_gui()

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

