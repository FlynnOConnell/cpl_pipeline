import dearpygui.dearpygui as dpg
from run_gui import show_gui

dpg.create_context()
dpg.create_viewport(title='Spike Sorter', width=600, height=600)

show_gui()

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

