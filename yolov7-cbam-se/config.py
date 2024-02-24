config={
    "name_project":'Detect passenger density.',

    # model detection
    "model":"model/cbam-yolov7.pt",   #"model/cbam-yolov7.pt" | "model/se-yolov7.pt"
    "class":{0: 'Body', 1: 'Face'},
    "color_class":[(243, 150, 31),(31, 233, 243)],
    "detection": [True,True],
    "color":[(121, 243, 31),(31, 243, 243),(243, 159, 31),(31, 31, 31)],
    "rating":[30,70,90],
    "max_seat": 28,

    # video
    "mode":1,
    "video":"video/12-13.mp4",
    "video_size":[640,480],
    "scope":[(257, 204),(349, 205),(636, 388),(640, 480),(0, 480),(4, 316)],
    "is_scope":True,
    "frame_interval_seconds":3,
    "confidence_threshold":0.5

}