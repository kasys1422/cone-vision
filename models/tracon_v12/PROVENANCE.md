# Model provenance

The detector uses the torchvision SSDlite320-MobileNetV3-Large architecture and COCO_V1 initialization. The 91-class classifier was replaced with a two-class cone/background head. See the [torchvision model specification](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) and [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE).

The landmark model derives from the V11 cone landmark checkpoint. Training used the frozen TraCon real labels and synthetic v6 records built from Mugicha3G's cone asset. The code and weights have distinct upstream provenance. Existing attribution and synthetic asset terms are described in the parent THIRD_PARTY_NOTICES.md and in the V11 source package. The archives exclude original TraCon photographs.
