import re

file_path = "/home/kukoboshi/Artificial Intelligence/IP1/trust-region-irl/trust_region_irl/environments/pusht_mjx/viewer.py"
with open(file_path, "r") as f:
    content = f.read()

# Add reset_requested property
content = content.replace("self.hide_menu = False", "self.hide_menu = False\n        self.reset_requested = False")

# Add KEY_R handling
content = content.replace("elif key == glfw.KEY_H:", "elif key == glfw.KEY_R:\n            self.reset_requested = True\n        elif key == glfw.KEY_H:")

# Add overlay text
content = content.replace('self.overlay[topleft][0] += "Press SPACE to pause.\\n"\n        self.overlay[topleft][1] += "\\n"', 'self.overlay[topleft][0] += "Press SPACE to pause.\\n"\n        self.overlay[topleft][1] += "\\n"\n        self.overlay[topleft][0] += "Press R to switch position.\\n"\n        self.overlay[topleft][1] += "\\n"')

with open(file_path, "w") as f:
    f.write(content)
print("Updated viewer.")

