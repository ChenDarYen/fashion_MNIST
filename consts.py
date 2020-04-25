LR = .0001
IMAGE_SIZE = 32
BATCH_SIZE = 50
EPOCHS = 8
STEPS_RECORD = 10
FASHION_MNIST = True
VGG16_HALF_CONFIG = [32, 32, 'P',
                     64, 64, 'P',
                     128, 128, 128, 'P',
                     256, 256, 256, 'P',
                     256, 256, 256, 'P']
VGG16_CONFIG = [64, 64, 'P',
                128, 128, 'P',
                256, 256, 256, 'P',
                512, 512, 512, 'P',
                512, 512, 512, 'P']

RECORD_PATH = 'records.pkl'
COLORS = ['blue', 'red', 'green', 'skyblue']
