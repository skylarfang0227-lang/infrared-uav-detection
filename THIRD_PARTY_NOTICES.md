# Third-Party Notices

- **Ultralytics YOLOv5u, YOLOv8 and YOLO11**: [upstream repository](https://github.com/ultralytics/ultralytics), version 8.3.161, AGPL-3.0. Dependencies are declared rather than bundled; source notices are retained. The native model YAML definitions are included with their source fingerprints.
- **YOLO11n-P2 configuration**: [Y-T-G community discussion](https://github.com/Y-T-G/community/discussions/60), used with the Ultralytics architecture and retained as a complete comparison model. See `benchmarks/detectors/protocol.json`.
- **MSDA-IRSTD**: [paper](https://doi.org/10.1007/s11760-026-05647-1). The reference is independently reconstructed from the paper; the full author configuration was not available. No paper PDF is included.
- **Sub-pixel convolution**: [ESPCN, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html) and [related detection work](https://doi.org/10.3390/s21165656). The local adaptation targets P2 classification/DFL projections; no first-use claim is made.
- **Multiple-padding input component**: [MP-YOLO author repository](https://github.com/zhangjun2636808827/MP-YOLO), reviewed commit `e53657f2dddbe313976ed0d0b5c1ad49eaede175`, AGPL-3.0. The local implementation partitions a common scratch-initialized stem across four padding patterns. It is not a full MP-YOLO reproduction.
- **SIRST-UAVB**: [dataset source](https://github.com/JN-Yang/PConv-SDloss-Data). Images and original annotation files are not distributed. Dataset usage conditions remain those of the source.
- **SIDD**: [author dataset repository](https://github.com/Dang-zy/SIDD), [originating study](https://doi.org/10.3390/drones7060393). Images are not bundled. The retained normalized box records support verification of the reported bounding-box assessment; dataset rights and terms remain with the source.

The AGPL-3.0 text is retained in [LICENSE](LICENSE), copied from the reviewed MP-YOLO repository. Its SHA256 is `4df3c306dddaaf4baffdff5ca820cc679ac8cd6dc263c6a74517783e42fa7a3b`. No ownership of third-party papers, datasets or methods is asserted.
