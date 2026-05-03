#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import xml.etree.ElementTree as ET


def parse_mesh_descriptors(xml_path: str) -> dict:
    """
    Parse MeSH DescriptorRecord XML and return a dict keyed by DescriptorUI.

    Returns:
      mesh_dict[DescriptorUI] = {
          "descriptor_name": str,
          "tree_numbers": list[str],
          "scope_note": str,
          "synonyms": list[str],
      }
    """
    xml_path = os.path.expanduser(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mesh_dict: dict[str, dict] = {}

    for desc_record in root.findall(".//DescriptorRecord"):
        ui_tag = desc_record.find("./DescriptorUI")
        if ui_tag is None or ui_tag.text is None:
            continue
        descriptor_ui = ui_tag.text.strip()
        if not descriptor_ui:
            continue

        name_tag = desc_record.find("./DescriptorName/String")
        descriptor_name = (
            name_tag.text.strip()
            if name_tag is not None and name_tag.text is not None
            else ""
        )

        tree_numbers: list[str] = []
        for tn in desc_record.findall("./TreeNumberList/TreeNumber"):
            if tn.text:
                t = tn.text.strip()
                if t:
                    tree_numbers.append(t)

        scope_note_tag = desc_record.find("./ConceptList/Concept/ScopeNote")
        scope_note = (
            scope_note_tag.text.strip()
            if scope_note_tag is not None and scope_note_tag.text is not None
            else ""
        )

        synonyms: list[str] = []
        for concept in desc_record.findall("./ConceptList/Concept"):
            for term_elem in concept.findall("./TermList/Term"):
                term_str_tag = term_elem.find("./String")
                if term_str_tag is not None and term_str_tag.text is not None:
                    s = term_str_tag.text.strip()
                    if s:
                        synonyms.append(s)

        # Deduplicate synonyms (case-insensitive), keep first occurrence
        seen = set()
        uniq_synonyms = []
        for s in synonyms:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq_synonyms.append(s)

        mesh_dict[descriptor_ui] = {
            "descriptor_name": descriptor_name,
            "tree_numbers": tree_numbers,
            "scope_note": scope_note,
            "synonyms": uniq_synonyms,
        }

    return mesh_dict


def build_label_text(descriptor_info: dict, min_synonyms: int = 1) -> str:
    """
    Build BERT input text from descriptor_name + (optional) a few synonyms + (optional) scope note.
    NOTE: TreeNumber is intentionally NOT included in label_text.
    """
    name_part = (descriptor_info.get("descriptor_name") or "").strip()
    scope_part = (descriptor_info.get("scope_note") or "").strip()

    synonyms_all = descriptor_info.get("synonyms", []) or []
    synonyms = [s for s in synonyms_all if s.lower() != name_part.lower()]

    k = max(0, int(min_synonyms))
    synonyms_text = "; ".join(synonyms[:k])

    text_parts: list[str] = []
    if name_part:
        text_parts.append(name_part)
    if synonyms_text:
        text_parts.append(f"Synonyms: {synonyms_text}")
    if scope_part:
        text_parts.append(f"Scope note: {scope_part}")

    return ". ".join(text_parts).strip()


def classify_tree_numbers(tree_numbers: list[str]) -> int:
    """
    Label tree_numbers by whether any of them contains a '.'.
    Returns: 1 if at least one '.' is present (i.e., depth > 0 exists),
             0 if there are no '.'s (root-level only, or no TreeNumber).
    """
    dot_count = sum(tn.count(".") for tn in (tree_numbers or []))
    return 1 if dot_count > 0 else 0


def main() -> None:
    xml_path = "./data/raw/desc2026.xml"
    output_csv = "./data/mesh/mesh_descriptors_withTree.csv"

    mesh_dict = parse_mesh_descriptors(xml_path)

    # Pre-build label_text (does not include TreeNumber)
    label_text_map: dict[str, str] = {}
    for ui, info in mesh_dict.items():
        label_text_map[ui] = build_label_text(info, min_synonyms=2)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Place tree_numbers in a separate column and add a label column for presence of '.'
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "descriptor_ui",
                "descriptor_name",
                "tree_numbers",
                "tree_has_dot_label",
                "label_text",
            ]
        )

        for ui, info in mesh_dict.items():
            descriptor_name = info.get("descriptor_name", "")
            tree_numbers = info.get("tree_numbers", []) or []
            tree_numbers_str = ";".join(tree_numbers)

            tree_has_dot_label = classify_tree_numbers(tree_numbers)

            label_text = label_text_map.get(ui, "")
            writer.writerow(
                [ui, descriptor_name, tree_numbers_str, tree_has_dot_label, label_text]
            )

    print(f"CSV written to: {output_csv}")

    # Sanity check: print the first 5 entries
    i = 0
    for ui, info in mesh_dict.items():
        trees = info.get("tree_numbers", []) or []
        label = classify_tree_numbers(trees)
        print(
            f"{ui} -> trees={';'.join(trees)} | tree_has_dot_label={label} | text={label_text_map[ui]}"
        )
        i += 1
        if i >= 5:
            break


if __name__ == "__main__":
    main()
