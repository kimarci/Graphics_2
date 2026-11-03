import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_blank(width, height):
    return Image.new("RGB", (width, height), (255, 255, 255))


def load_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def transfer_semicircle(src, dst, r, cx2, cy2, cx1, cy1):
    src_arr = np.array(src, dtype=np.uint8)
    dst_arr = np.array(dst, dtype=np.uint8)

    H1, W1 = dst_arr.shape[:2]
    H2, W2 = src_arr.shape[:2]

    dx = cx1 - cx2
    dy = cy1 - cy2

    ys, xs = np.meshgrid(np.arange(H1), np.arange(W1), indexing="ij")

    d2 = (ys - cy1) ** 2 + (xs - cx1) ** 2

    inside_mask = (d2 < r * r) & (ys >= cy1 - r)

    ring_outer = (r + 1) * (r + 1)
    border_mask = (d2 >= r * r) & (d2 <= ring_outer) & (ys >= cy1 - r)

    src_xs = xs - dx
    src_ys = ys - dy
    valid_src = (src_xs >= 0) & (src_xs < W2) & (src_ys >= 0) & (src_ys < H2)

    copy_mask = inside_mask & valid_src
    dst_arr[copy_mask] = src_arr[src_ys[copy_mask], src_xs[copy_mask]]

    black_mask = inside_mask & ~valid_src
    dst_arr[black_mask] = (0, 0, 0)

    dst_arr[border_mask] = (127, 255, 127)

    src_border_mask = border_mask & valid_src
    src_arr[src_ys[src_border_mask], src_xs[src_border_mask]] = (255, 127, 127)

    src.paste(Image.fromarray(src_arr), (0, 0))
    dst.paste(Image.fromarray(dst_arr), (0, 0))
    return dst


def draw_axes(img):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    color = (0, 0, 0)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    draw.line([(10, H - 10), (10, 10)], fill=color, width=1)
    draw.polygon([(5, 20), (10, 10), (15, 20)], fill=color)

    draw.line([(8, H // 2), (W - 10, H // 2)], fill=color, width=1)
    draw.polygon([(W - 20, H // 2 - 5), (W - 10, H // 2), (W - 20, H // 2 + 5)], fill=color)

    for i in range(0, (H - 40) // 20 + 1):
        y1 = i * 10 + H // 2
        y2 = H // 2 - i * 10
        if 0 <= y1 < H:
            draw.line([(10, y1), (8, y1)], fill=color)
        if 0 <= y2 < H:
            draw.line([(10, y2), (8, y2)], fill=color)

    for i in range(0, (W - 40) // 10 + 1):
        x = (i + 1) * 10
        if x < W:
            draw.line([(x, H // 2), (x, H // 2 + 2)], fill=color)

    draw.text((1, H // 2 + 1), "0", fill=color, font=font)
    draw.text((W - 20, H // 2 + 6), "x", fill=color, font=font)
    draw.text((20, 1), "y", fill=color, font=font)

    return img


def draw_function(img, func, label="y = x^(3/7)"):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    color = (127, 127, 255)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    draw.text((W // 2, 1), label, fill=(0, 0, 0), font=font)

    scale_y = H // 2 - 20
    prev = None
    for i in range(1, W - 20):
        arg = i / 10.0
        try:
            val = func(arg)
        except (ValueError, ZeroDivisionError, OverflowError):
            prev = None
            continue
        px = i + 10
        py = H // 2 - int(round(val * scale_y))
        if prev is not None:
            draw.line([prev, (px, py)], fill=color, width=1)
        prev = (px, py)

    return img


def func_x_3_7(x):
    if x < 0:
        raise ValueError("negative argument")
    return x ** (3.0 / 7.0)


def _to_bilevel_mask(img, threshold=128):
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return (arr < threshold).astype(np.uint8)


def save_pbm_ascii(img, path, threshold=128, comment="Lab2 variant 18, PBM P1"):
    mask = _to_bilevel_mask(img, threshold)
    H, W = mask.shape

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("P1\n")
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"{W} {H}\n")
        for row in mask:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")


def load_pbm_ascii(path):
    with open(path, "rb") as f:
        raw = f.read()

    pos = 0
    tokens = []
    while len(tokens) < 3 and pos < len(raw):
        while pos < len(raw) and raw[pos:pos + 1] in b" \t\r\n":
            pos += 1
        if raw[pos:pos + 1] == b"#":
            while pos < len(raw) and raw[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(raw) and raw[pos:pos + 1] not in b" \t\r\n":
            pos += 1
        tokens.append(raw[start:pos])

    magic = tokens[0].decode("ascii")
    W = int(tokens[1])
    H = int(tokens[2])

    if magic == "P1":
        data = raw[pos:].split()
        vals = np.array([int(b) for b in data[:W * H]], dtype=np.uint8)
        if vals.size != W * H:
            vals = np.zeros(W * H, np.uint8)
        vals = vals.reshape((H, W))
        img_arr = (1 - vals) * 255
        return Image.fromarray(img_arr.astype(np.uint8), mode="L").convert("1")

    if magic == "P4":
        if pos < len(raw) and raw[pos:pos + 1] in b" \t\r\n":
            pos += 1
        row_bytes = (W + 7) // 8
        arr = np.zeros((H, W), np.uint8)
        for y in range(H):
            row = raw[pos + y * row_bytes: pos + (y + 1) * row_bytes]
            for x in range(W):
                byte = row[x // 8]
                bit = (byte >> (7 - (x % 8))) & 1
                arr[y, x] = bit
        img_arr = (1 - arr) * 255
        return Image.fromarray(img_arr.astype(np.uint8), mode="L").convert("1")

    raise ValueError(f"Неподдерживаемый формат PBM: {magic}")


class Lab2App:
    def __init__(self, root):
        self.root = root
        root.title("Лаб. 2, вариант 18: полукруг + y = x^(3/7), PBM")

        self.src_img = None
        self.dst_img = None
        self.src_path = ""
        self.dst_path = ""

        frm = tk.Frame(root)
        frm.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        self.e_w1 = self._add_entry(frm, "Width 1:", "300", 0)
        self.e_h1 = self._add_entry(frm, "Height 1:", "300", 1)
        self.e_r  = self._add_entry(frm, "R:", "80", 2)
        self.e_x1 = self._add_entry(frm, "x1:", "150", 3)
        self.e_y1 = self._add_entry(frm, "y1:", "299", 4)
        self.e_x2 = self._add_entry(frm, "x2:", "320", 5)
        self.e_y2 = self._add_entry(frm, "y2:", "359", 6)

        btn_frm = tk.Frame(root)
        btn_frm.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        tk.Button(btn_frm, text="Создать 1", command=self.on_create).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Открыть 2", command=self.on_open).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Перенести", command=self.on_transfer).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Координаты", command=self.on_axes).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="x^(3/7)", command=self.on_plot).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Сохранить", command=self.on_save).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Сохранить PBM", command=self.on_save_pbm).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text="Открыть PBM", command=self.on_open_pbm).pack(side=tk.LEFT, padx=2)

        cv_frm = tk.Frame(root)
        cv_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas_new = tk.Canvas(cv_frm, bg="lightgray", width=360, height=360)
        self.canvas_new.pack(side=tk.LEFT, padx=4)
        tk.Label(cv_frm, text="Новое изображение").pack(side=tk.LEFT)

        self.canvas_src = tk.Canvas(cv_frm, bg="lightgray", width=360, height=360)
        self.canvas_src.pack(side=tk.RIGHT, padx=4)
        tk.Label(cv_frm, text="Исходное изображение").pack(side=tk.RIGHT)

        self.tk_new = None
        self.tk_src = None

        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def _add_entry(self, parent, label, default, col):
        tk.Label(parent, text=label).grid(row=0, column=col, padx=2)
        e = tk.Entry(parent, width=7)
        e.insert(0, default)
        e.grid(row=1, column=col, padx=2)
        return e

    def on_create(self):
        try:
            w = int(self.e_w1.get())
            h = int(self.e_h1.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Ширина/высота должны быть целыми")
            return
        self.dst_img = create_blank(w, h)
        self._refresh_canvas(self.canvas_new, self.dst_img, "new")
        self.status.set(f"Создано новое изображение {w}x{h}")

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[("Изображения", "*.bmp *.png *.jpg *.jpeg *.gif *.tif *.tiff"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.src_img = load_image(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return
        self.src_path = path
        self._refresh_canvas(self.canvas_src, self.src_img, "src")

        w2, h2 = self.src_img.size
        self.e_x2.delete(0, tk.END); self.e_x2.insert(0, str(w2 // 2))
        self.e_y2.delete(0, tk.END); self.e_y2.insert(0, str(h2 - 1))
        self.e_r.delete(0, tk.END); self.e_r.insert(0, str(min(w2, h2) // 4))

        self.status.set(f"Открыто: {os.path.basename(path)} ({w2}x{h2})")

    def on_transfer(self):
        if self.src_img is None:
            messagebox.showwarning("Внимание", "Сначала откройте исходное изображение")
            return
        if self.dst_img is None:
            messagebox.showwarning("Внимание", "Сначала создайте новое изображение")
            return
        try:
            r  = int(self.e_r.get())
            x1 = int(self.e_x1.get()); y1 = int(self.e_y1.get())
            x2 = int(self.e_x2.get()); y2 = int(self.e_y2.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Все параметры должны быть целыми")
            return
        src_copy = self.src_img.copy()
        dst_copy = self.dst_img.copy()
        transfer_semicircle(src_copy, dst_copy, r, x2, y2, x1, y1)
        self.src_img = src_copy
        self.dst_img = dst_copy
        self._refresh_canvas(self.canvas_new, self.dst_img, "new")
        self._refresh_canvas(self.canvas_src, self.src_img, "src")
        self.status.set(f"Перенесён полукруг R={r} из ({x2},{y2}) в ({x1},{y1})")

    def on_axes(self):
        if self.dst_img is None:
            messagebox.showwarning("Внимание", "Сначала создайте новое изображение")
            return
        self.dst_img = draw_axes(self.dst_img.copy())
        self._refresh_canvas(self.canvas_new, self.dst_img, "new")
        self.status.set("Оси координат построены")

    def on_plot(self):
        if self.dst_img is None:
            messagebox.showwarning("Внимание", "Сначала создайте новое изображение")
            return
        self.dst_img = draw_function(self.dst_img.copy(), func_x_3_7, "y = x^(3/7)")
        self._refresh_canvas(self.canvas_new, self.dst_img, "new")
        self.status.set("График y = x^(3/7) построен")

    def on_save(self):
        if self.dst_img is None:
            messagebox.showwarning("Внимание", "Нечего сохранять")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            self.dst_img.save(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")
            return
        self.dst_path = path
        self.status.set(f"Сохранено: {os.path.basename(path)}")

    def on_save_pbm(self):
        if self.dst_img is None:
            messagebox.showwarning("Внимание", "Сначала создайте новое изображение")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить как PBM (ASCII)",
            defaultextension=".pbm",
            filetypes=[("Portable BitMap (ASCII)", "*.pbm"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            save_pbm_ascii(self.dst_img, path, threshold=128,
                           comment="Lab2 variant 18, PBM P1")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить PBM:\n{e}")
            return
        self.status.set(f"Сохранено в PBM: {os.path.basename(path)}")

    def on_open_pbm(self):
        path = filedialog.askopenfilename(
            title="Открыть PBM",
            filetypes=[("Portable BitMap", "*.pbm"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            img = load_pbm_ascii(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать PBM:\n{e}")
            return
        self.dst_img = img.convert("RGB")
        self._refresh_canvas(self.canvas_new, self.dst_img, "new")
        self.status.set(f"Открыт PBM: {os.path.basename(path)} "
                        f"({self.dst_img.size[0]}x{self.dst_img.size[1]})")

    def _refresh_canvas(self, canvas, img, kind):
        if img is None:
            return
        cw = canvas.winfo_width() or 360
        ch = canvas.winfo_height() or 360
        im = img.copy()
        im.thumbnail((cw, ch))
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(im)
        if kind == "new":
            self.tk_new = tk_img
        else:
            self.tk_src = tk_img
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")


def main():
    root = tk.Tk()
    root.geometry("980x560")
    app = Lab2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()