import numpy as np

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate anchors windows by enumerating ratios and scales
    with respect to a reference (0, 0, 15, 15) base anchor
    """
    
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 # [0, 0, 15, 15]
    ratio_anchors = enumerate_ratio(base_anchor, ratios) # (3, 4)
    anchors = np.vstack([enumerate_scale(ratio_anchors[i, :], scales)
                        for i in range(ratio_anchors.shape[0])])
    return anchors # (9, 4)

def whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    ctr_x = anchor[0] + 0.5 * (w - 1)
    ctr_y = anchor[1] + 0.5 * (h - 1)

    return w, h, ctr_x, ctr_y

def make_anchors(ws, hs, ctr_x, ctr_y):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((ctr_x - 0.5 * (ws - 1),
                         ctr_y - 0.5 * (hs - 1),
                         ctr_x + 0.5 * (ws - 1),
                         ctr_y + 0.5 * (hs - 1)))
    return anchors 

def enumerate_ratio(anchor, ratios):
    w, h, ctr_x, ctr_y = whctrs(anchor)
    size = w * h 
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = make_anchors(ws, hs, ctr_x, ctr_y)
    return anchors # (3, 4)

def enumerate_scale(anchor, scales):
    w, h, ctr_x, ctr_y = whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = make_anchors(ws, hs, ctr_x, ctr_y)
    return anchors
