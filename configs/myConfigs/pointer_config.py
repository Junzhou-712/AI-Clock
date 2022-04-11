# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)
    )
)

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('pointer','scale')
data = dict(
    train=dict(
        img_prefix='cocome/train/',
        classes=classes,
        ann_file='cocome/annotations/instance_train.json'),
    val=dict(
        img_prefix='cocome/val/',
        classes=classes,
        ann_file='cocome/annotations/instance_val.json'),
    test=dict(
        img_prefix='cocome/test/',
        classes=classes,
        ann_file='cocome/annotations/instance_test.json'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = '/home/ubuntu/Software/mmdetection/checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'