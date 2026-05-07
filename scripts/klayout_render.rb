app = RBA::Application.instance
mw = app.main_window

BASE = '/openfma/openlane/fma_top/runs/RUN_2026.05.05_03.57.29/results'
OUT = '/openfma/delivery/images'

gds = BASE + '/final/gds/fma_top_stage4.gds'

# === 1. Final GDS — overview ===
puts '=== 1. Final GDS overview ==='
mw.load_layout(gds, 0)
lv = mw.current_view
cv = lv.active_cellview
cell = cv.layout.top_cell
lv.select_cell(cv.cell_index, 0)
lv.zoom_fit
lv.save_image(OUT + '/final_layout.png', 2000, 1500)
puts 'OK final_layout.png'

bbox = cell.bbox
w = bbox.width
h = bbox.height

# === 2. Detail: bottom-left corner ===
puts '=== 2. Detail zoom ==='
box = RBA::DBox.new(bbox.left + w*0.0, bbox.bottom + h*0.0,
                    bbox.left + w*0.25, bbox.bottom + h*0.25)
lv.zoom_box(box)
lv.save_image(OUT + '/layout_detail.png', 2000, 1500)
puts 'OK layout_detail.png'

# === 3. Detail: center ===
puts '=== 3. Center ==='
box = RBA::DBox.new(bbox.left + w*0.35, bbox.bottom + h*0.35,
                    bbox.left + w*0.65, bbox.bottom + h*0.65)
lv.zoom_box(box)
lv.save_image(OUT + '/layout_center.png', 2000, 1500)
puts 'OK layout_center.png'

# === 4. Top-level overview (wider) ===
puts '=== 4. Top overview ==='
lv.zoom_fit
lv.save_image(OUT + '/chip_full.png', 2400, 1800)
puts 'OK chip_full.png'
