import numpy as np
import cv2
import math


def swap_line_pt_maybe(line):
    '''
    [x0, y0, x1, y1]
    '''
    L = line

    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        if line[0] > line[2]:
            L = [line[2], line[3], line[0], line[1], line[4]]
    else:
        if line[1] > line[3]:
            L = [line[2], line[3], line[0], line[1], line[4]]
    return L


def get_ext_lines(norm_lines, h=256, w=256, min_len=0.125):
    mu_half = min_len / 2
    ext_lines = []
    for line in norm_lines:
        x0, y0, x1, y1, cls = line
        line_len = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        nn = int(line_len / mu_half) - 1
        # print("nn: ", nn)
        if nn <= 1:
            ext_lines.append(line)
        else:
            ## y = k * x + b
            if abs(x0 - x1) > abs(y0 - y1):
                ## y = k* x + b
                k = (y1 - y0) / (x1 - x0)
                b = y1 - k * x1
                step = (x1 - x0) / (nn + 1) # 计算平均每个小段的距离
                len_step = 2 * step  # (x1 - x0) / (nn - 1)  没两个小段为一个线段
                for ix in range(nn):
                    ix0 = x0 + ix * step
                    # ix1 = x0 + (ix + 1) * step
                    ix1 = ix0 + len_step
                    iy0 = k * ix0 + b
                    iy1 = k * ix1 + b
                    ext_lines.append([ix0, iy0, ix1, iy1, cls])

            else:
                ## x = k* y + b
                k = (x1 - x0) / (y1 - y0)
                b = x1 - k * y1
                step = (y1 - y0) / (nn + 1)
                len_step = 2 * step  # (y1 - y0) / (nn - 1)
                for iy in range(nn):
                    iy0 = y0 + iy * step
                    # iy1 = y0 + (iy + 1) * step
                    iy1 = iy0 + len_step
                    ix0 = k * iy0 + b
                    ix1 = k * iy1 + b
                    ext_lines.append([ix0, iy0, ix1, iy1, cls])
    return ext_lines


def line_len_and_angle(x0, y0, x1, y1):
    if abs(x0 - x1) < 1e-6:
        ang = np.pi / 2
    else:
        ang = np.arctan(abs((y0 - y1) / (x0 - x1)))

    ang = ang / (2 * np.pi) + 0.5
    len = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return len, ang


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):  # heatmap h * w
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def cut_line_by_xmin(line, xmin):
    if line[0] > xmin and line[2] > xmin:
        return  True, line
    if line[0] <= xmin and line[2] <= xmin:
        return  False, line
    if abs(line[0] - line[2]) < 1:
        return  False, line
    # y = k*x  + b
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = line[3] - k * line[2]
    y = k * xmin + b
    p0 = [xmin, y]
    if line[0] < line[2]:
        p1 = [line[2], line[3]]
    else:
        p1 = [line[0], line[1]]
    line = [p0[0], p0[1], p1[0], p1[1], line[4]]

    return True, line


def cut_line_by_xmax(line, xmax):
    if line[0] < xmax and line[2] < xmax:
        return  True, line
    if line[0] >= xmax and line[2] >= xmax:
        return  False, line
    if abs(line[0] - line[2]) < 1:
        return  False, line
    # y = k*x  + b
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = line[3] - k * line[2]
    y = k * xmax + b
    p1 = [xmax, y]
    if line[0] > line[2]:
        p0 = [line[2], line[3]]
    else:
        p0 = [line[0], line[1]]
    return True, [p0[0], p0[1], p1[0], p1[1], line[4]]


def near_area_n(xc, yc, n=5):
    if n <= 1:
        return [[xc, yc]]
    n = n // 2
    ptss = []
    for x in range(xc - n, xc + n + 1):
        for y in range(yc - n, yc + n + 1):
            ptss.append([x, y])
    return ptss


def gen_TP_mask3(norm_lines, h=256, w=256, input_cls_dict=None, with_ext=False):
    assert input_cls_dict != 0, "input_cls_dict !=0"
    """
    centermap, displacement_map, length_map, degree_map, cls_map
    1 cengter + 4  dis + 2 + cls
    return [19, h, w]
    """
    len_divide_v = np.sqrt(h ** 2 + w ** 2)  # feature map diagonal length

    centermap = np.zeros((1, h, w), dtype=np.uint8)
    displacement_map = np.zeros((4, h, w), dtype=np.float32)
    length_map = np.zeros((1, h, w), dtype=np.float32)
    degree_map = np.zeros((1, h, w), dtype=np.float32)
    cls_num = len(input_cls_dict.keys())
    cls_map = np.zeros((cls_num, h, w), dtype=np.float32)

    for l in norm_lines:
        x0 = int(round(l[0] * w))
        y0 = int(round(l[1] * h))
        x1 = int(round(l[2] * w))
        y1 = int(round(l[3] * h))  # GT in 256 x 256 feture map

        # TODO, sort points by x
        # ....

        cls = l[4]
        for c in input_cls_dict.keys():
            if cls == c:
                cls_id = input_cls_dict[cls]

        xc = round(w * (l[0] + l[2]) / 2)
        yc = round(h * (l[1] + l[3]) / 2)  # center point GT

        xc = int(np.clip(xc, 0, w - 1))
        yc = int(np.clip(yc, 0, h - 1))

        centermap[0, yc, xc] = 255  # give value for center map

        line_len, ang = line_len_and_angle(x0, y0, x1, y1)  # get degree(normarlized) and line length
        line_len /= len_divide_v  # normarlize line length
        length_map[0, yc, xc] = line_len
        degree_map[0, yc, xc] = ang  # give value for degree map and length map with same position as center point

        x0d = x0 - xc
        y0d = y0 - yc
        x1d = x1 - xc
        y1d = y1 - yc  # get displacement value relate (xs,ys),(xe,ye) to (xc,yc)

        displacement_map[0, yc, xc] = x0d  # / 2
        displacement_map[1, yc, xc] = y0d  # / 2
        displacement_map[2, yc, xc] = x1d  # / 2
        displacement_map[3, yc, xc] = y1d  # / 2  # give value for displacement map

        # cls_map representation
        eps = 0.00001
        height, width = abs(x1 - x0) + eps, abs(y1 - y0) + eps
        if height > 0 and width > 0:
            radius = gaussian_radius((math.ceil(height), math.ceil(width)))
            radius = max(0, int(radius))
            ct = np.array([xc, yc], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(cls_map[cls_id], ct_int, radius)

        # why normalize?
        for inx, i in enumerate(cls_map):
            b = i.max() - i.min()
            if b != 0:
                cls_map[inx, :, :] = (cls_map[inx, :, :] - cls_map[inx, :, :].min()) / b

        ## walk around line
        # ptss = work_around_line(x0, y0, x1, y1, n=5, r=0.0, thickness=3)

        # extrapolated to a 3×3 window
        ptss = near_area_n(xc, yc, n=3)  # 得到center point附近的8个像素点，形成3x3的window
        for p in ptss:
            xc = round(p[0])
            yc = round(p[1])
            xc = int(np.clip(xc, 0, w - 1))
            yc = int(np.clip(yc, 0, h - 1))
            x0d = x0 - xc
            y0d = y0 - yc
            x1d = x1 - xc
            y1d = y1 - yc
            displacement_map[0, yc, xc] = x0d  # / 2
            displacement_map[1, yc, xc] = y0d  # / 2
            displacement_map[2, yc, xc] = x1d  # / 2
            displacement_map[3, yc, xc] = y1d  # / 2 # 给displacement map 赋同样的值

            length_map[0, yc, xc] = line_len
            degree_map[0, yc, xc] = ang  # 同上

        xc = round(w * (l[0] + l[2]) / 2)
        yc = round(h * (l[1] + l[3]) / 2)

        xc = int(np.clip(xc, 0, w - 1))
        yc = int(np.clip(yc, 0, h - 1))

        centermap[0, yc, xc] = 255

        line_len, ang = line_len_and_angle(x0, y0, x1, y1)
        line_len /= len_divide_v
        length_map[0, yc, xc] = line_len
        degree_map[0, yc, xc] = ang

        x0d = x0 - xc
        y0d = y0 - yc
        x1d = x1 - xc
        y1d = y1 - yc

        displacement_map[0, yc, xc] = x0d  # / 2
        displacement_map[1, yc, xc] = y0d  # / 2
        displacement_map[2, yc, xc] = x1d  # / 2
        displacement_map[3, yc, xc] = y1d  # / 2   赋值了两次，再次确认一遍，以防漏了？？？

    centermap[0, :, :] = cv2.GaussianBlur(centermap[0, :, :], (3, 3), 0.0)  # 对centermap取3x3的高斯
    centermap = np.array(centermap, dtype=np.float32) / 255.0
    b = centermap.max() - centermap.min()
    if b != 0:
        centermap = (centermap - centermap.min()) / b  # 对centermap归一化

    tp_mask = np.concatenate((centermap, displacement_map, length_map, degree_map, cls_map), axis=0)
    return tp_mask


def gen_SOL_map2(norm_lines,  h =256, w =256, min_len =0.125, with_ext= False, input_cls_dict=None):
    assert input_cls_dict != None, "input_cls_dict != None"
    """
    1 + 4 + 2 + 12
    return [19, h, w]
    """
    ext_lines = get_ext_lines(norm_lines, h, w, min_len) # 得到均分后的小线段，加入cls类别标签
    return gen_TP_mask3(ext_lines, h, w, input_cls_dict=input_cls_dict, with_ext=with_ext), ext_lines  # 像TPmap一样处理SOLmap， 并输出均分后的小线段。


def gen_junction_and_line_mask(norm_lines, h=256, w=256):
    junction_map = np.zeros((h, w, 1), dtype=np.float32)
    line_map = np.zeros((h, w, 1), dtype=np.float32)

    radius = 1
    for l in norm_lines:
        x0 = int(round(l[0] * w))
        y0 = int(round(l[1] * h))
        x1 = int(round(l[2] * w))
        y1 = int(round(l[3] * h))
        cv2.line(line_map, (x0, y0), (x1, y1), (255, 255, 255), radius)
        # cv2.circle(junction_map, (x0, y0), radius, (255, 255, 255), radius)
        # cv2.circle(junction_map, (x1, y1), radius, (255, 255, 255), radius)

        ptss = near_area_n(x0, y0, n=3)
        ptss.extend(near_area_n(x1, y1, n=3))
        for p in ptss:
            xc = round(p[0])
            yc = round(p[1])
            xc = int(np.clip(xc, 0, w - 1))
            yc = int(np.clip(yc, 0, h - 1))
            junction_map[yc, xc, 0] = 255

    junction_map[:, :, 0] = cv2.GaussianBlur(junction_map[:, :, 0], (3, 3), 0.0)
    junction_map = np.array(junction_map, dtype=np.float32) / 255.0
    b = junction_map.max() - junction_map.min()
    if b != 0:
        junction_map = (junction_map - junction_map.min()) / b  # 和centernet一样进行归一化
    # line map use binary one
    line_map = np.array(line_map, dtype=np.float32) / 255.0  # linemap 二值化，只有0和1
    #     line_map[:, :, 0] = cv2.GaussianBlur(line_map[:, :, 0], (3, 3), 0.0)
    #     line_map = np.array(line_map, dtype=np.float32) / 255.0
    #     b = line_map.max() - line_map.min()
    #     if b !=0:
    #         line_map = ( line_map - line_map.min() ) / b

    junction_map = junction_map.transpose(2, 0, 1)
    line_map = line_map.transpose(2, 0, 1)

    return junction_map, line_map


def gen_curve_mask(norm_curves, h=256, w=256):
    '''
    norm_curves: [[[point1], [point2] ..., num_points, cls](curve1), [[point1], [point2], ..., num_points, cls](curve2), ...]
    curve_map : 1 x 256 x 256 map
    '''
    curve_map = np.zeros((h, w, 1), dtype=np.float32)
    radius = 2

    lines = []
    for c in norm_curves:
        num = c[-2]
        line = []
        for i in range(num - 1):
            x1, y1 = c[i]
            x2, y2 = c[i + 1]
            x1 = int(round(x1 * w))
            y1 = int(round(y1 * h))
            x2 = int(round(x2 * w))
            y2 = int(round(y2 * h))
            line.append([x1, y1, x2, y2])
        lines.append(line)

    # print(lines)
    for l in lines:
        for ll in l:
            # eps = 0.00001
            # h, w = abs(ll[0] - ll[2]) + eps, abs(ll[1] - ll[3]) + eps
            # radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            # radius = max(1, int(radius))
            cv2.line(curve_map, (ll[0], ll[1]), (ll[2], ll[3]), (255, 255, 255), radius)

    curve_map = np.array(curve_map, dtype=np.float32) / 255.0
    curve_map = curve_map.transpose(2, 0, 1)

    return curve_map
