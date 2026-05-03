import csv
import os
import xml.etree.ElementTree as ET


def parse_mesh_descriptors(xml_path):
    xml_path = os.path.expanduser(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mesh_dict = {}
    for desc_record in root.findall(".//DescriptorRecord"):
        ui_tag = desc_record.find("./DescriptorUI")
        if ui_tag is None or ui_tag.text is None:
            continue
        descriptor_ui = ui_tag.text.strip()

        name_tag = desc_record.find("./DescriptorName/String")
        descriptor_name = (
            name_tag.text.strip()
            if name_tag is not None and name_tag.text is not None
            else ""
        )

        tree_numbers = []
        for tn in desc_record.findall("./TreeNumberList/TreeNumber"):
            if tn.text is not None:
                tree_numbers.append(tn.text.strip())

        scope_note_tag = desc_record.find("./ConceptList/Concept/ScopeNote")
        scope_note = (
            scope_note_tag.text.strip()
            if scope_note_tag is not None and scope_note_tag.text is not None
            else ""
        )

        synonyms = []
        concept_list = desc_record.findall("./ConceptList/Concept")
        for concept in concept_list:
            term_list = concept.findall("./TermList/Term")
            for term_elem in term_list:
                term_str_tag = term_elem.find("./String")
                if term_str_tag is not None and term_str_tag.text is not None:
                    synonyms.append(term_str_tag.text.strip())

        mesh_dict[descriptor_ui] = {
            "descriptor_name": descriptor_name,
            "tree_numbers": tree_numbers,
            "scope_note": scope_note,
            "synonyms": list(set(synonyms)),
        }
    return mesh_dict


def build_label_text(descriptor_info, min_synonyms=1):
    name_part = descriptor_info["descriptor_name"]
    scope_part = descriptor_info["scope_note"]
    synonyms = [
        s for s in descriptor_info["synonyms"] if s.lower() != name_part.lower()
    ]
    synonyms_text = "; ".join(synonyms[:min_synonyms])
    text_parts = []
    text_parts.append(name_part)
    if synonyms_text:
        text_parts.append(f"Synonyms: {synonyms_text}")
    if scope_part:
        text_parts.append(f"Scope note: {scope_part}")
    full_text = ". ".join(text_parts)
    return full_text


def main():
    xml_path = "./data/raw/desc2026.xml"
    mesh_dict = parse_mesh_descriptors(xml_path)

    label_text_map = {}
    for ui, info in mesh_dict.items():
        text_for_bert = build_label_text(info, min_synonyms=2)
        label_text_map[ui] = text_for_bert

    # Output CSV path under ./data/
    output_csv = os.path.expanduser("./data/mesh/mesh_descriptors.csv")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["descriptor_ui", "descriptor_name", "label_text"])  # Header row

        for ui, info in mesh_dict.items():
            descriptor_name = info["descriptor_name"]
            label_text = label_text_map[ui]  # Text assembled by build_label_text
            writer.writerow([ui, descriptor_name, label_text])

    print(f"CSV written to: {output_csv}")

    # Sanity check: print the first 5 entries
    i = 0
    for ui, text in label_text_map.items():
        print(f"{ui} -> {text}")
        i += 1
        if i >= 5:
            break


if __name__ == "__main__":
    main()