from ultralytics import YOLO

if __name__ == '__main__':

    # 使用预训练权重+配置文件创建模型
    # model = YOLO(r'yolo11n.pt')
    model = YOLO(r'/YOLO11/ultralytics/cfg/models/11/yolo11.yaml')
    # model.load('/workspace/YOLO11/ultralytics/cfg/models/11/yolo11.yaml')
    model.train(cfg='ultralytics/cfg/default.yaml',
                data=r"/workspace/data.yaml",
                optimizer='SGD',
                epochs=200,
                project=r"/YOLO11/runs",
                name="train-200epoch-v11n.yaml",
                batch=24,
                workers=8,
                split='val'
                )

    # #恢复中断的训练
    # Load a model
    # model = YOLO(r"E:\knowledge\YOLO-v8\ultralytics-main\runs\detected\Packages\train-200epoch-v8s\weights\last.pt")  # load a partially trained model
    # # Resume training
    # result = model.train(resume=True,
    #                      data=r"E:\knowledge\YOLO-v8\ultralytics-main\runs\detected\Packages\train-200epoch-v8s\weights\last.pt"
    #                      )

    # # 模型验证
    # model = YOLO(r"D:\Dataset\All\CarDetection\Udacity Self Driving Car.v2-fixed-large\PedestrianAndVehicles-13146\YOLO-v8-GhostNetV2\runs\detect\train-300epoch-GhostNetV2\weights\best.pt")
    # model.val( data= r"D:\Dataset\All\CarDetection\Udacity Self Driving Car.v2-fixed-large\PedestrianAndVehicles-13146\YOLO-v8-GhostNetV2\runs\detect\train-300epoch-GhostNetV2\weights\best.pt",
    #            cfg = 'ultralytics/cfg/default.yaml',
    #            split = 'val',
    #            project=r"D:\Dataset\All\CarDetection\Udacity Self Driving Car.v2-fixed-large\PedestrianAndVehicles-13146\YOLO-v8-GhostNetV2\runs\detect",
    #            name='val_GhostNetV2')

    # # 模型推理
    # model = YOLO(r"/workspace/YOLO11/yolo11n.pt")
    # model.predict(source=r"/workspace/YOLO11/ultralytics/assets",
    #               save=True,
    #               project=r"/workspace/YOLO11/runs",
    #               name='predict'
    #               )

    # # 模型转换
    # # Load a model
    # model = YOLO(r"E:\knowledge\YOLO-v8\ultralytics-main\best.pt")  # load a custom trained model
    # model.export(device='cuda', format="onnx", dynamic=False, simplify=True, opset=11)