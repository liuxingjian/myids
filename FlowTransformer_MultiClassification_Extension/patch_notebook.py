import json

with open("multi_classification_demo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("1. 临时导出" in line for line in source):
            # remove starting from # 1.5
            new_source = []
            for line in source:
                if "# 1.5 替换 ONNX 中的" in line:
                    break
                new_source.append(line)
            
            # Now we find where 2. 转换 starts
            resume_source = []
            resume = False
            for line in source:
                if "2. 将 ONNX" in line:
                    resume = True
                if resume:
                    resume_source.append(line)
            
            cell["source"] = new_source + ["\n"] + resume_source
            break

with open("multi_classification_demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
