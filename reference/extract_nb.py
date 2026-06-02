import json, sys

nb = json.load(open("ISLR_1st_place_Hoyeol_Sohn.ipynb", "r", encoding="utf-8"))
with open("nb_extracted.txt", "w", encoding="utf-8") as f:
    for i, c in enumerate(nb["cells"]):
        f.write(f"\n=== CELL {i} [{c['cell_type']}] ===\n")
        f.write("".join(c["source"]))
        f.write("\n")
print("Done. Written to nb_extracted.txt")
