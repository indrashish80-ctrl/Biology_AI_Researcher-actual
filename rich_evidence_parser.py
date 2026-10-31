"""
rich_evidence_parser.py
-----------------------
Extracts figures, captions, supplementary datasets, and experimental tables
directly from JATS XML documents.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict


def text_of(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class RichEvidenceParser:
    @staticmethod
    def parse_rich_evidence(xml_path: str) -> Dict[str, Any]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        figures = []
        supplementary = []
        tables = []

        for el in root.iter():
            tag = lname(el.tag)

            # Figures & Captions
            if tag == "fig":
                fig_id = el.attrib.get("id", "")
                label = next((text_of(x) for x in el if lname(x.tag) == "label"), "")
                caption = next((text_of(x) for x in el if lname(x.tag) == "caption"), "")
                graphic = next((x.attrib.get("{http://www.w3.org/1999/xlink}href", "") for x in el.iter() if lname(x.tag) == "graphic"), "")
                figures.append({
                    "id": fig_id,
                    "label": label,
                    "caption": caption,
                    "image_href": graphic,
                })

            # Supplementary Materials
            elif tag == "supplementary-material":
                sup_id = el.attrib.get("id", "")
                label = next((text_of(x) for x in el if lname(x.tag) == "label"), "")
                caption = next((text_of(x) for x in el if lname(x.tag) == "caption"), "")
                href = next((x.attrib.get("{http://www.w3.org/1999/xlink}href", "") for x in el.iter() if lname(x.tag) == "media"), "")
                supplementary.append({
                    "id": sup_id,
                    "label": label,
                    "caption": caption,
                    "file_link": href,
                })

            # Tables
            elif tag == "table-wrap":
                tbl_id = el.attrib.get("id", "")
                label = next((text_of(x) for x in el if lname(x.tag) == "label"), "")
                caption = next((text_of(x) for x in el if lname(x.tag) == "caption"), "")
                tables.append({"id": tbl_id, "label": label, "caption": caption})

        root.clear()
        return {
            "figures_count": len(figures),
            "figures": figures[:8],
            "tables_count": len(tables),
            "tables": tables[:8],
            "supplementary_materials": supplementary[:5],
        }