import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('rtdetr-GhostnetLECA.yaml')
    model.train(data=r'VisDrone.yaml',
                cache=False,
                imgsz=1024,
                epochs=400,
                batch=6,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='visdrone_rtdetr',
                name='exp',
                )

