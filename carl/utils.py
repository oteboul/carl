import colorsys

def make_color(h):
    return "#{:02x}{:02x}{:02x}".format(
        *map(lambda x: int(255*x), colorsys.hsv_to_rgb(h, 0.8, 0.8)))
