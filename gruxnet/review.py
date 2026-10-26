"""Render SAM valence crops for checking the extracted graphical ratings."""
import json
from pathlib import Path


def render_sam_review(audit_dir: Path):
    import fitz
    from PIL import Image, ImageDraw
    ratings = json.loads((audit_dir / "gameemo_sam_ratings.json").read_text())
    for batch in range(0, len(ratings), 56):
        subset = ratings[batch:batch + 56]
        sheet = Image.new("RGB", (420*4, 102*((len(subset)+3)//4)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, rating in enumerate(subset):
            with fitz.open(rating["pdf"]) as doc:
                pixmap = doc[0].get_pixmap(matrix=fitz.Matrix(.7, .7), clip=fitz.Rect(0, 437, 595.32, 549), alpha=False)
                crop = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            x, y = (i%4)*420, (i//4)*102
            draw.text((x+10, y+2), f"{rating['trial_id']}   V={rating['valence']} A={rating['arousal']}", fill="black")
            sheet.paste(crop, (x, y+20))
        target = audit_dir / f"sam_valence_review_{batch//56+1}.png"
        sheet.save(target)
        print(target, flush=True)
