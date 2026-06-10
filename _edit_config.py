path = r"D:\The_course_of_Grade_2\AI\2048\config.py"
with open(path, "r", encoding="utf-8", newline="") as f:
    content = f.read()
idx = content.find("MODEL_PATH = ")
after_models = content.find("\n\n", idx)
insertion_point = after_models + 2
new_content = (
    content[:insertion_point]
    + "\n# 监督学习模型路径\n"
    + 'SUPERVISED_MODEL_PATH = "2048_supervised_model.pth"\n'
    + "\n"
    + content[insertion_point:]
)
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(new_content)
print("Success")
