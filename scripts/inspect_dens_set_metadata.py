"""Read a bounded, release-bound DENS SET header without decoding outcomes/data.

This is a deliberately selective MATLAB-v5 reader, not scipy.io.loadmat.
Only whitelisted acquisition metadata are decoded. Unknown fields stay opaque.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

import requests

COMMIT = "5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c"
SUBJECTS = ("sub-mit003", "sub-mit081", "sub-mit108")
MAX_TOTAL = 12 * 1024 * 1024
MAX_INFLATED = 64 * 1024 * 1024
SCALARS = {"nbchan", "trials", "pnts", "srate", "xmin", "xmax"}
NUMERIC = {1: ("b", 1), 2: ("B", 1), 3: ("h", 2), 4: ("H", 2),
           5: ("i", 4), 6: ("I", 4), 7: ("f", 4), 9: ("d", 8),
           12: ("q", 8), 13: ("Q", 8)}


def tag(buf, pos, endian="<"):
    if pos + 8 > len(buf):
        raise ValueError("Truncated MAT tag")
    word, second = struct.unpack_from(endian + "II", buf, pos)
    small = word >> 16
    if small:
        if small > 4:
            raise ValueError("Invalid small element")
        return word & 0xFFFF, buf[pos + 4:pos + 4 + small], pos + 8
    end = pos + 8 + second
    if end > len(buf):
        raise ValueError("Truncated MAT payload")
    # miCOMPRESSED elements are not padded in common MATLAB/scipy writers.
    next_pos = end if word == 15 else end + ((-second) % 8)
    return word, buf[pos + 8:end], next_pos


class Matrix:
    def __init__(self, payload, endian):
        self.buf, self.endian = payload, endian
        typ, flags, pos = tag(payload, 0, endian)
        if typ != 6 or len(flags) < 8:
            raise ValueError("Invalid array flags")
        self.cls = struct.unpack_from(endian + "I", flags)[0] & 0xFF
        typ, dims, pos = tag(payload, pos, endian)
        if typ != 5 or len(dims) % 4:
            raise ValueError("Invalid dimensions")
        self.shape = list(struct.unpack(endian + "i" * (len(dims) // 4), dims))
        if any(n < 0 for n in self.shape):
            raise ValueError("Negative dimensions")
        typ, name, pos = tag(payload, pos, endian)
        if typ not in (1, 2, 16):
            raise ValueError("Invalid array name")
        self.name = bytes(name).decode("ascii", errors="strict")
        self.pos = pos
        self.count = 1
        for n in self.shape:
            self.count *= n
        self.fields = []
        if self.cls == 2:
            typ, field_width, pos = tag(payload, pos, endian)
            if typ != 5 or len(field_width) != 4:
                raise ValueError("Invalid struct field width")
            width = struct.unpack(endian + "i", field_width)[0]
            typ, names, pos = tag(payload, pos, endian)
            if typ not in (1, 2) or width < 1 or len(names) % width:
                raise ValueError("Invalid struct field names")
            self.fields = [bytes(names[i:i + width]).split(b"\0", 1)[0].decode("ascii")
                           for i in range(0, len(names), width)]
            self.pos = pos

    def members(self):
        if self.cls != 2:
            raise ValueError("Not a struct")
        pos = self.pos
        for row in range(self.count):
            for field in self.fields:
                typ, child, pos = tag(self.buf, pos, self.endian)
                if typ != 14:
                    raise ValueError("Struct member is not matrix")
                # Lazy children: unused fields (including legal empty miMATRIX
                # values) are never parsed or decoded.
                yield row, field, child

    def scalar(self):
        if self.count != 1 or self.cls in (1, 2, 4, 5):
            raise ValueError("Expected acquisition scalar")
        typ, data, _ = tag(self.buf, self.pos, self.endian)
        if typ not in NUMERIC:
            raise ValueError("Unsupported acquisition scalar type")
        code, size = NUMERIC[typ]
        if len(data) != size:
            raise ValueError("Invalid acquisition scalar size")
        value = struct.unpack(self.endian + code, data)[0]
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise ValueError("Nonfinite acquisition scalar")
        return value

    def text(self):
        if self.cls != 4 or self.count > 256:
            raise ValueError("Expected bounded metadata text")
        typ, data, _ = tag(self.buf, self.pos, self.endian)
        codec = {1: "ascii", 2: "ascii", 16: "utf-8",
                 4: "utf-16-le" if self.endian == "<" else "utf-16-be",
                 17: "utf-16-le" if self.endian == "<" else "utf-16-be",
                 18: "utf-32-le" if self.endian == "<" else "utf-32-be"}.get(typ)
        if codec is None:
            raise ValueError("Unknown metadata text type")
        return bytes(data).decode(codec, errors="strict").rstrip("\0")


def inspect_bytes(body):
    if len(body) < 128 or not body.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("Only MATLAB-v5 files are permitted")
    marker = body[126:128]
    endian = "<" if marker == b"IM" else ">" if marker == b"MI" else None
    if endian is None:
        raise ValueError("Invalid MAT endian marker")
    buf, pos = memoryview(body), 128
    roots = []
    inflated = 0
    while pos + 8 <= len(buf):
        typ, payload, pos = tag(buf, pos, endian)
        if typ == 15:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, MAX_INFLATED + 1)
            if len(raw) > MAX_INFLATED or decoder.unconsumed_tail or not decoder.eof:
                raise ValueError("Compressed metadata exceeds bound or is incomplete")
            inflated += len(raw)
            if inflated > MAX_INFLATED:
                raise ValueError("Total inflated metadata exceeds bound")
            inner = memoryview(raw)
            ityp, ipayload, ipos = tag(inner, 0, endian)
            if ityp != 14 or ipos != len(inner):
                raise ValueError("Unexpected compressed matrix geometry")
            roots.append(Matrix(ipayload, endian))
        elif typ == 14:
            roots.append(Matrix(payload, endian))
        else:
            raise ValueError("Unexpected top-level MAT type")
    if len(roots) == 1 and roots[0].name == "EEG":
        eeg = roots[0]
    else:
        raise ValueError("Expected a single EEG root; do not inspect other variables")
    if eeg.cls != 2 or eeg.count != 1:
        raise ValueError("Expected scalar EEG structure")
    result = {"mat_endian": endian, "top_level_names": [m.name for m in roots],
              "eeg_field_names": eeg.fields, "inflated_container_bytes": inflated,
              "acquisition": {}, "schema": {}, "skipped_fields": [],
              "numerical_waveform_values_decoded": 0,
              "behavioral_outcome_values_decoded": 0,
              "event_values_decoded": 0}
    for row, field, child in eeg.members():
        if field not in SCALARS | {"data", "ref", "event", "urevent", "epoch", "chanlocs"}:
            result["skipped_fields"].append(field)
            continue
        if len(child) == 0:
            result["schema"][field] = {"empty_matrix": True}
            continue
        member = Matrix(child, endian)
        if field in SCALARS:
            result["acquisition"][field] = member.scalar()
        elif field == "data":
            # Never inspect a numeric embedded data array, even if small.
            if member.cls != 4:
                raise ValueError("Embedded EEG.data rejected without decoding")
            filename = member.text()
            if not re.fullmatch(r"sub-mit\d+_task-Emotion_eeg\.fdt", filename):
                raise ValueError("Unexpected external waveform filename")
            result["external_waveform_filename"] = filename
        elif field == "ref":
            if member.cls == 4:
                result["reference_header"] = member.text()
            else:
                result["reference_header_shape"] = member.shape
        elif field in ("event", "urevent", "epoch"):
            result["schema"][field] = {"mat_class": member.cls,
                                       "shape": member.shape,
                                       "entry_count": member.count,
                                       "field_names": member.fields}
        elif field == "chanlocs":
            if member.cls != 2 or member.count > 256:
                raise ValueError("Unexpected channel metadata")
            labels = []
            for index, key, channel_child in member.members():
                if key == "labels":
                    value = Matrix(channel_child, endian)
                    labels.append(value.text())
            result["channel_labels"] = labels
            result["channel_metadata_fields"] = member.fields
    if set(result["acquisition"]) != SCALARS or "external_waveform_filename" not in result:
        raise ValueError("Required acquisition metadata incomplete")
    if len(result.get("channel_labels", [])) != result["acquisition"]["nbchan"]:
        raise ValueError("Channel count/label geometry mismatch")
    return result


def qualify(work_dir: Path, output: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    records, total = [], 0
    for subject in SUBJECTS:
        relative = f"{subject}/eeg/{subject}_task-Emotion_eeg.set"
        pointer_url = f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/{COMMIT}/{relative}"
        response = session.get(pointer_url, timeout=45)
        response.raise_for_status()
        pointer = response.content
        match = re.search(rb"MD5E-s(\d+)--([a-f0-9]{32})\.set", pointer)
        if match is None:
            raise ValueError("Missing pinned MD5E SET pointer")
        size, md5 = int(match[1]), match[2].decode()
        if size + total > MAX_TOTAL:
            raise ValueError("Bounded SET download budget exceeded")
        object_url = f"https://s3.amazonaws.com/openneuro.org/ds003751/{relative}"
        path = work_dir / (subject + ".set")
        if path.exists():
            body = path.read_bytes()
            downloaded = 0
        else:
            chunks, count = [], 0
            with session.get(object_url, timeout=60, stream=True) as recording:
                recording.raise_for_status()
                for chunk in recording.iter_content(chunk_size=65536):
                    count += len(chunk)
                    if count > size:
                        raise ValueError("Recording body exceeds release-bound length")
                    chunks.append(chunk)
            body = b"".join(chunks)
            downloaded = len(body)
        if len(body) != size or hashlib.md5(body).hexdigest() != md5:
            raise ValueError("SET failed pinned release integrity binding")
        if not path.exists():
            path.write_bytes(body)
        total += len(body)
        inspected = inspect_bytes(body)
        records.append({"subject": subject, "relative_path": relative,
                        "pointer_url": pointer_url, "object_url": object_url,
                        "pointer_sha256": hashlib.sha256(pointer).hexdigest(),
                        "annex_md5": md5, "size": size,
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "file_body_bytes_downloaded_this_run": downloaded,
                        "release_integrity_binding": "pinned Git-annex length and MD5E passed; MD5 not strong provenance authentication",
                        "metadata": inspected})
    receipt = {"schema": 1, "snapshot_commit": COMMIT, "records": records,
               "set_file_bytes_inspected": total,
               "fdt_waveform_body_bytes_requested": 0,
               "individual_outcomes_decoded": 0, "event_values_decoded": 0,
               "new_fits": 0, "new_predictions": 0,
               "scope": "Three SET acquisition headers/schema only, no behavioral/event values"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "set_bytes": total, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.work_dir, args.output)


if __name__ == "__main__":
    main()
