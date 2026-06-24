"""Generate the Geopickr icon/favicon: a particle-covered sphere (the Geometry
Picker 'sphere' output) in pastel colours on a dark rounded square."""
import math, colorsys, os
from PIL import Image, ImageDraw, ImageFilter

S = 1024                      # supersampled master
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# rounded-square dark background (transparent outside corners)
pad, rad = int(S * 0.035), int(S * 0.22)
d.rounded_rectangle([pad, pad, S - pad, S - pad], radius=rad, fill=(13, 17, 23, 255))

cx, cy, R = S / 2, S / 2, S * 0.345
n = 230
ang = math.radians(20)        # tilt for a 3-D feel
ca, sa = math.cos(ang), math.sin(ang)

pts = []
for i in range(n):
    z = 1 - 2 * (i + 0.5) / n
    r = math.sqrt(max(0.0, 1 - z * z))
    th = math.pi * (1 + 5 ** 0.5) * i           # golden-angle (Fibonacci) sphere
    x, y = math.cos(th) * r, math.sin(th) * r
    y2, z2 = y * ca - z * sa, y * sa + z * ca    # tilt about x
    hue = (i * 0.61803398875) % 1.0              # spread hues
    pts.append((x, y2, z2, hue))

pts.sort(key=lambda p: p[2])                      # back-to-front painter's order
for x, y, z, hue in pts:
    depth = (z + 1) / 2                           # 0 (back) .. 1 (front)
    px, py = cx + x * R, cy - y * R
    dotR = R * 0.082 * (0.6 + 0.55 * depth)
    rr, gg, bb = colorsys.hsv_to_rgb(hue, 0.42, 1.0)
    shade = 0.45 + 0.55 * depth                   # dim toward the back for volume
    col = (int(rr * 255 * shade), int(gg * 255 * shade), int(bb * 255 * shade), 255)
    d.ellipse([px - dotR, py - dotR, px + dotR, py + dotR], fill=col)

# soft specular highlight (upper-left)
hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
hd = ImageDraw.Draw(hl)
hr = R * 0.55
hd.ellipse([cx - R * 0.5 - hr, cy - R * 0.55 - hr,
            cx - R * 0.5 + hr, cy - R * 0.55 + hr], fill=(255, 255, 255, 60))
hl = hl.filter(ImageFilter.GaussianBlur(S * 0.04))
# clip highlight to the sphere
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
img = Image.composite(Image.alpha_composite(img, hl), img, mask)

master = img.resize((512, 512), Image.LANCZOS)

repo = "/Users/bbarad/Downloads/ChimeraX-Geopickr"
os.makedirs(repo + "/docs", exist_ok=True)
master.save(repo + "/docs/geopickr_icon.png")
master.resize((256, 256), Image.LANCZOS).save(repo + "/src/icons/geopickr.png")
master.resize((128, 128), Image.LANCZOS).save(repo + "/docs/favicon.png")
master.save(repo + "/docs/favicon.ico",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)])
master.resize((256, 256), Image.LANCZOS).save("/tmp/geopickr_icon_preview.png")
# tiny preview to check small-size readability
master.resize((32, 32), Image.LANCZOS).resize((160, 160), Image.NEAREST).save(
    "/tmp/geopickr_icon_32.png")
print("icon written")
