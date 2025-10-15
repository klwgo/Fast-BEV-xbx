import os, glob, cv2
from natsort import natsorted  # pip install natsort (可选, 没有就用 sorted)

def even_size(w, h):
    return (w + w%2, h + h%2)

def images_to_video(img_dir, out_path, fps=10, max_w=1920, max_h=1080):
    imgs = glob.glob(os.path.join(img_dir, "*.png"))
    if not imgs: 
        return False
    imgs = natsorted(imgs) if 'natsort' in globals() else sorted(imgs)

    # 读第一张，确定目标尺寸（限制到 max_w/max_h，保持比例）
    im0 = cv2.imread(imgs[0])
    h0, w0 = im0.shape[:2]
    scale = min(max_w/float(w0), max_h/float(h0), 1.0)
    W = int(w0 * scale); H = int(h0 * scale)
    W, H = even_size(W, H)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))

    for p in imgs:
        im = cv2.imread(p)
        if im is None: continue
        if (im.shape[1], im.shape[0]) != (W, H):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(im)
    vw.release()
    print("Saved:", out_path)
    return True

if __name__ == "__main__":
    SRC_DIR = "runs/vis_fastbev_mini_offline"  # 👈 改成你的目录
    print('1')
    OUT_DIR = "runs/videos_py"
    FPS = 10
    os.makedirs(OUT_DIR, exist_ok=True)

    for d in sorted([os.path.join(SRC_DIR, x) for x in os.listdir(SRC_DIR)]):
        if not os.path.isdir(d): 
            continue
        base = os.path.basename(d)
        out = os.path.join(OUT_DIR, f"{base}.mp4")
        images_to_video(d, out, fps=FPS, max_w=1920, max_h=1080)
