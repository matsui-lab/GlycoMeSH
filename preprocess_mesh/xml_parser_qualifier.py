import csv
import os
import xml.etree.ElementTree as ET


def _safe_text(elem, default=""):
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def parse_mesh_qualifiers(xml_path: str):
    xml_path = os.path.expanduser(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    qual_dict = {}

    for qrec in root.findall(".//QualifierRecord"):
        ui = _safe_text(qrec.find("./QualifierUI"))
        if not ui:
            continue

        qualifier_name = _safe_text(qrec.find("./QualifierName/String"))

        tree_numbers = []
        for tn in qrec.findall("./TreeNumberList/TreeNumber"):
            t = _safe_text(tn)
            if t:
                tree_numbers.append(t)

        scope_note = ""
        preferred_concept = qrec.find('./ConceptList/Concept[@PreferredConceptYN="Y"]')
        if preferred_concept is not None:
            scope_note = _safe_text(preferred_concept.find("./ScopeNote"))
        if not scope_note:
            any_concept = qrec.find("./ConceptList/Concept")
            if any_concept is not None:
                scope_note = _safe_text(any_concept.find("./ScopeNote"))

        synonyms = []
        abbreviations = set()

        for concept in qrec.findall("./ConceptList/Concept"):
            for term in concept.findall("./TermList/Term"):
                term_str = _safe_text(term.find("./String"))
                if term_str:
                    synonyms.append(term_str)

                abbr = _safe_text(term.find("./Abbreviation"))
                if abbr:
                    abbreviations.add(abbr)

        synonyms = list({s.strip() for s in synonyms if s.strip()})

        qual_dict[ui] = {
            "qualifier_name": qualifier_name,
            "tree_numbers": tree_numbers,
            "scope_note": scope_note,
            "synonyms": synonyms,
            "abbreviations": sorted(abbreviations),
        }

    return qual_dict


def build_qual_label_text(qual_info, min_synonyms=1):
    name_part = qual_info["qualifier_name"]
    scope_part = qual_info.get("scope_note", "")
    abbreviations = qual_info.get("abbreviations", [])

    synonyms = [s for s in qual_info.get("synonyms", []) if s.lower() != name_part.lower()]
    synonyms_text = "; ".join(synonyms[:min_synonyms])

    text_parts = [name_part]

    if abbreviations:
        text_parts.append(f"Abbrev: {', '.join(abbreviations)}")

    if synonyms_text:
        text_parts.append(f"Synonyms: {synonyms_text}")

    if scope_part:
        text_parts.append(f"Scope note: {scope_part}")

    return ". ".join(text_parts)


def main():
    xml_path = "./data/raw/qual2026.xml"
    qual_dict = parse_mesh_qualifiers(xml_path)

    label_text_map = {}
    for ui, info in qual_dict.items():
        label_text_map[ui] = build_qual_label_text(info, min_synonyms=2)

    output_csv = os.path.expanduser("./data/mesh/mesh_qualifiers.csv")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qualifier_ui", "qualifier_name", "label_text"])

        for ui, info in qual_dict.items():
            writer.writerow([ui, info["qualifier_name"], label_text_map[ui]])

    print(f"CSV written to: {output_csv}")

    # Sanity check: print the first 5 entries
    for i, (ui, text) in enumerate(label_text_map.items()):
        print(f"{ui} -> {text}")
        if i >= 4:
            break


if __name__ == "__main__":
    main()
