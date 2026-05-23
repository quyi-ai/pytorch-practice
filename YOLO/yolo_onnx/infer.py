from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

# 当前文件：YOLO/yolo_onnx/infer.py
# ROOT 表示 YOLO/yolo_onnx 目录
ROOT = Path(__file__).resolve().parent
ONNX_PATH=ROOT / "models/best.onnx"
IMAGE_PATH = ROOT / "010.png"
SAVE_PATH=ROOT / "010result.jpg"
SHAPE=(640,640)
CONF=0.4
IOU=0.5
def preprocess(image):#处理图片，转化为np数组形式
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#BGR变RGB
    img=img.astype(np.float32)/255.0#调整数据格式，归一化
    img=np.transpose(img,(2,0,1))# HWC -> CHW
    img = np.ascontiguousarray(img)# 确保内存连续
    img = np.expand_dims(img, axis=0)    # CHW -> NCHW，加 batch 维度
    return img

def letterbox(image,new_shape,color=(114,114,114)):# 保持原图比例缩放，并用灰边补到指定尺寸
    if isinstance(new_shape,int):# 如果只传一个数字，就认为目标高宽相同
        target_h,target_w=new_shape,new_shape# 目标尺寸为正方形
    else:# 如果传入的是元组，就分别取目标高和目标宽
        target_h,target_w=new_shape# 目标尺寸：[高, 宽]
    h,w=image.shape[:2]# 取出原图高度和宽度
    ratio=min(target_h/h,target_w/w)# 计算缩放比例，让图片完整放进目标尺寸
    new_h,new_w=int(round(h*ratio)),int(round(w*ratio))# 根据缩放比例计算新的高度和宽度
    new_img=cv2.resize(image,(new_w,new_h))# 按新的宽高缩放图片
    pad_h,pad_w=target_h-new_h,target_w-new_w# 计算高和宽方向一共需要补多少边
    pad_top,pad_bottom=pad_h//2,pad_h-pad_h//2# 计算上下需要补的边
    pad_left,pad_right=pad_w//2,pad_w-pad_w//2# 计算左右需要补的边
    pad=(pad_left,pad_top)# 记录左侧和上侧 padding，后面还原坐标时会用到
    new_img=cv2.copyMakeBorder(new_img,borderType=cv2.BORDER_CONSTANT,top=pad_top,bottom=pad_bottom,left=pad_left,right=pad_right,value=color)# 用固定颜色补边
    return new_img,ratio,pad# 返回 letterbox 后的图片、缩放比例和 padding

def xywh_to_xyxy(boxes):#改变框的输出格式
    boxes=np.asarray(boxes)
    x_c=boxes[:,0]
    y_c=boxes[:,1]
    w=boxes[:,2]
    h=boxes[:,3]
    x1=x_c-w/2
    x2=x_c+w/2
    y1=y_c-h/2
    y2=y_c+h/2
    return np.stack([x1,y1,x2,y2],axis=1)

def scale_box(boxes,ratio,pad,original_shape):#把框映射回原图片
    boxes=np.asarray(boxes,dtype=np.float32).copy()
    single_box=boxes.ndim==1
    if single_box:
        boxes=np.expand_dims(boxes,axis=0)
    pad_x,pad_y=pad
    h,w=original_shape[:2]
    boxes[:,[0,2]]=(boxes[:,[0,2]]-pad_x)/ratio
    boxes[:,[1,3]]=(boxes[:,[1,3]]-pad_y)/ratio
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w-1)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h-1)
    return boxes[0] if single_box else boxes

def draw_box(image, boxes, scores):#在原图画框
    boxes=np.asarray(boxes)
    scores=np.asarray(scores)
    if boxes.ndim==1:
        boxes=np.expand_dims(boxes,axis=0)
        scores=np.expand_dims(scores,axis=0)
    for box,score in zip(boxes,scores):
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            4
        )
        text = f"score:{score:.2f}"
        # 文字位置，放在框的左上方；如果太靠上，就放在 y=0
        text_position = (x1, max(y1 - 20, 40))
        cv2.putText(
            image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 0, 0),
            2
        )

    return image

def filter_by_score(boxes,scores,conf):    # 这个函数负责置信度过滤。
    # YOLO 会输出很多候选框，大部分框的 score 很低，基本可以认为不是目标。
    # 所以在做 NMS 之前，先把低分框去掉，减少后处理计算量。
    # 当前 boxes 仍然是 xywh 格式，先筛掉框再画出来，所以不在这里做坐标转换。
    # 输入：
    #   boxes: shape [N, 4]
    #   scores: shape [N]
    #   conf_threshold: 分数阈值
    # 输出：
    #   filtered_boxes: 分数大于阈值的框
    #   filtered_scores: 对应的分数
    mask=scores>conf
    boxes=boxes[mask]
    scores=scores[mask]
    return boxes,scores

def compute_iou(box,boxes):# 这个函数用于计算 IoU。IoU 表示两个框的重叠程度，值越大说明两个框越重叠。NMS 会用 IoU 判断哪些框是重复框。
    # 输入 box 是一个 xyxy 格式的框。[4]
    # 输入 boxes 是多个 xyxy 格式的框。[N,4]
    # 返回 box 和 boxes 中每个框的 IoU。[N]
    boxes=np.asarray(boxes)
    x1,y1,x2,y2=box
    a1,b1,a2,b2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x=np.maximum(x1,a1)
    y=np.maximum(y1,b1)
    a=np.minimum(x2,a2)
    b=np.minimum(y2,b2)
    w=np.maximum(a-x,0)
    h=np.maximum(b-y,0)
    s0=w*h
    s1=(x2-x1)*(y2-y1)
    s2=(a2-a1)*(b2-b1)
    union=s1+s2-s0
    iou=np.zeros_like(union,dtype=np.float32)
    np.divide(s0,union,out=iou,where=union>0)
    return iou

def nms(boxes,scores,iou_threshold):# NMS 的目标是去掉重复检测框。
    # 输入 boxes 必须是 xyxy 格式。
    # 输入 scores 是每个框的置信度。
    # 算法会优先保留最高分框，然后删除和它 IoU 过高的低分框。
    # 返回最终保留框的下标 keep_indices。
    keep=np.empty(len(scores),dtype=np.int64)
    keep_count=0
    order=np.argsort(scores)[::-1]
    while len(order)>0:
        current=order[0]
        keep[keep_count]=current
        keep_count+=1
        ious=compute_iou(boxes[current],boxes[order[1:]])
        mask=ious<iou_threshold
        order=order[1:][mask]
    return keep[:keep_count]

def main():
    image=cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    original_shape=image.shape
    image_lb,ratio,pad=letterbox(image,SHAPE)
    input_tensor=preprocess(image_lb)
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    pred=outputs[0][0]
    if pred.shape[0]<pred.shape[1]:
        pred=pred.T
    box_xywh=pred[:,:4]
    scores=pred[:,4]
    fboxes_xywh,fscore=filter_by_score(box_xywh,scores,CONF)
    if len(fscore)==0:
        cv2.imwrite(str(SAVE_PATH),image)
        print("no boxes found")
        print(f"result saved to: {SAVE_PATH}")
        return
    fboxes_xxyy=xywh_to_xyxy(fboxes_xywh)
    indices=nms(fboxes_xxyy,fscore,IOU)
    final_boxes=fboxes_xxyy[indices]
    final_scores=fscore[indices]
    boxes=scale_box(final_boxes,ratio,pad,original_shape)
    image=draw_box(image,boxes,final_scores)
    cv2.imwrite(str(SAVE_PATH),image)
    print(f"result saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
